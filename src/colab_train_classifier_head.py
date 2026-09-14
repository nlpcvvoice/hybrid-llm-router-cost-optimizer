#%% md
# Train Qwen2.5-1.5B-Instruct + LoRA classifier head (free Colab)

**Goal**: fine-tune the 1.5B base into a LOW/HIGH difficulty router, so the cost-saving gateway can decide which requests go to which model.

**Usage (2 interactions only)**
1. Menu: Runtime ▶ Run all (data is embedded in the notebook's DATA cell, no upload needed)
2. Training takes ~30–60 min, then auto-downloads `best_adapter.zip` and `metrics.json` back to the local `outputs/` dir

**Notes**: fp16 LoRA (no bitsandbytes dependency, most version-stable). 560 training samples / 6 epochs, fits a 16G T4 comfortably. Re-run after disconnect: just ▶ Run all — data lives in the notebook, results are deterministically reproducible (fixed seed 42).

#%% python
import importlib.util
import subprocess
import sys

_MISSING = [p for p in ("peft", "transformers", "datasets", "accelerate", "mlflow")
            if importlib.util.find_spec(p) is None]
if _MISSING:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_MISSING])

try:
    from importlib.metadata import version as _ver
    _tv = tuple(int(x) for x in _ver("torchao").split(".")[:2])
    if _tv < (0, 16):
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
        print("torchao 版本过低,已卸载(本训练不使用)")
except Exception:
    pass
print("deps ok:", _MISSING if _MISSING else "all present")

#%% python
import os
import json
import shutil
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")
print("Python ok")

#%% python
if not all(os.path.exists(f) for f in ("train.jsonl", "val.jsonl", "test.jsonl")):
    from google.colab import files
    up = files.upload()
    for name, raw in up.items():
        with open(name, "wb") as fh:
            fh.write(raw)

def read_jsonl(name):
    with open(name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train_raw = read_jsonl("train.jsonl")
val_raw = read_jsonl("val.jsonl")
test_raw = read_jsonl("test.jsonl")
from collections import Counter
print("train:", len(train_raw), dict(Counter(r["audited_label"] for r in train_raw)))
print("val  :", len(val_raw), dict(Counter(r["audited_label"] for r in val_raw)))
print("test :", len(test_raw), dict(Counter(r["audited_label"] for r in test_raw)))

#%% python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = int(os.environ.get("EPOCHS", "6"))
DEBUG_STEPS = int(os.environ.get("DEBUG_STEPS", "0"))
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42
LABEL_TO_ID = {"LOW": 0, "HIGH": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
print("torch", torch.__version__, "device", DEVICE)

#%% python
class RouterDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.labels = torch.tensor([LABEL_TO_ID[r["audited_label"]] for r in rows])
        enc = tok([r["prompt"] for r in rows],
                  padding="max_length", truncation=True, max_length=max_len,
                  return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]

class QwenClassifier(nn.Module):
    def __init__(self, base_id, lora_cfg):
        super().__init__()
        self.lora_model = get_peft_model(
            AutoModel.from_pretrained(base_id, torch_dtype=DTYPE), lora_cfg)
        self.lora_model.config.use_cache = False
        h = self.lora_model.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, len(LABEL_TO_ID)))
        for p in self.lora_model.parameters():
            p.requires_grad = False
        for n, p in self.lora_model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.lora_model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        lens = attention_mask.sum(1) - 1
        pooled = last[torch.arange(last.size(0)), lens]
        return self.head(pooled)

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none")
print("lora config ok")

#%% python
def autocast_ctx():
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)

def evaluate(model, dl):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for ids, am, y in dl:
            with autocast_ctx():
                logits = model(ids.to(DEVICE), am.to(DEVICE))
            ys.append(y)
            ps.append(torch.softmax(logits.float().cpu(), 1)[:, 1])
    return torch.cat(ys), torch.cat(ps)

def metrics(y, p, thr=0.5):
    pred = (p >= thr).long()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = float((pred == y).float().mean())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)

def best_threshold(y, p):
    best, bt = -1.0, 0.5
    for t in [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]:
        _, _, _, f1 = metrics(y, p, t)
        if f1 > best:
            best, bt = f1, t
    return bt, best

#%% python
tok = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
train_ds = RouterDataset(train_raw, tok, MAX_LEN)
val_ds = RouterDataset(val_raw, tok, MAX_LEN)
test_ds = RouterDataset(test_raw, tok, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = QwenClassifier(BASE_MODEL, lora_cfg).to(DEVICE)
cnt = Counter(int(x) for x in train_ds.labels)
n = len(train_ds)
w = torch.tensor([n / (2 * cnt[0]), n / (2 * cnt[1])], dtype=torch.float32).to(DEVICE)
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
print("class weights:", [round(float(x), 3) for x in w])
print("trainable params:", sum(p.numel() for p in trainable))

#%% python
hist = []
best_f1, best_ep, best_t = -1.0, -1, 0.5
for ep in range(EPOCHS):
    model.train()
    run_loss, n_steps = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for step, (ids, am, y) in enumerate(train_dl):
        ids, am, y = ids.to(DEVICE), am.to(DEVICE), y.to(DEVICE)
        with autocast_ctx():
            logits = model(ids, am)
            loss = nn.functional.cross_entropy(logits, y, weight=w) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        run_loss += float(loss.detach()) * GRAD_ACCUM
        n_steps += 1
        if DEBUG_STEPS and n_steps >= DEBUG_STEPS:
            break
    yv, pv = evaluate(model, val_dl)
    acc, prec, rec, f1 = metrics(yv, pv, 0.5)
    bt, bf = best_threshold(yv, pv)
    hist.append({"epoch": ep + 1, "train_loss": round(run_loss / n_steps, 4),
                 "val_acc": acc, "val_prec": prec, "val_rec": rec, "val_f1": f1,
                 "best_thr": bt, "best_thr_f1": round(bf, 4)})
    print(f"[epoch {ep + 1}] loss={run_loss / n_steps:.4f}  "
          f"val acc={acc} prec={prec} rec={rec} f1={f1}  thr={bt}(f1={bf:.4f})")
    if bf >= best_f1:
        best_f1, best_ep, best_t = bf, ep + 1, bt
    sched.step()
print(f"best epoch={best_ep}  val@0.5 f1={max(h['val_f1'] for h in hist)}  "
      f"thr={best_t} f1={best_f1:.4f}")

#%% python
import re
from transformers import AutoModelForCausalLM

JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
print("[baseline] judge =", JUDGE_MODEL, flush=True)
jtok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
jmod = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL).to(DEVICE)

J_PROMPTS = {
    "binary": ("You are a request router. Classify the following user request as "
               "LOW or HIGH difficulty for a small language model to answer correctly. "
               'Reply with exactly one word: LOW or HIGH.\n\nRequest: {prompt}'),
    "score": ("You are a request router. Rate the complexity of the following "
              "request on a scale of 1 (very simple) to 5 (very hard) "
              'for a small language model. Reply in JSON: {{"score": <int>}}.\n\n'
              "Request: {prompt}"),
}

def j_gen(prompt):
    msgs = [{"role": "system",
             "content": "You are a strict request router. Follow the user's "
                        "formatting instructions exactly."},
            {"role": "user", "content": prompt}]
    enc = jtok.apply_chat_template(msgs, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), autocast_ctx():
        out = jmod.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            max_new_tokens=8, do_sample=False,
                            pad_token_id=jtok.eos_token_id)
    return jtok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

def j_binary(raw):
    m = re.search(r"LOW|HIGH", raw.upper() or "")
    return m.group(0) if m else "LOW"

def j_score(raw):
    m = re.search(r"\d", raw)
    return (max(1, min(5, int(m.group(0)))) - 1) / 4.0 if m else 0.5

base_report = {}
for s, rows in (("val", val_raw), ("test", test_raw)):
    print(f"\n=== zero-shot judge @ {s} n={len(rows)} ===", flush=True)
    y = torch.tensor([LABEL_TO_ID[r["audited_label"]] for r in rows])
    base_report[s] = {}
    for vname, tmpl in J_PROMPTS.items():
        preds, scores = [], []
        for r in rows:
            raw = j_gen(tmpl.format(prompt=r["prompt"]))
            if vname == "binary":
                preds.append(1.0 if j_binary(raw) == "HIGH" else 0.0)
            else:
                scores.append(j_score(raw))
        pt = torch.tensor(preds if vname == "binary" else scores)
        a, p_, r_, f_ = metrics(y, pt, 0.5)
        entry = {"at_0.5": [a, p_, r_, f_]}
        if vname != "binary":
            best, bt = -1.0, 0.5
            for t in [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]:
                _, _, _, f = metrics(y, pt, t)
                if f > best:
                    best, bt = f, t
            a2, p2_, r2_, f2_ = metrics(y, pt, bt)
            entry["best_threshold"] = bt
            entry["at_best_threshold"] = [a2, p2_, r2_, f2_]
        base_report[s][vname] = entry
        print(f"  {vname:6s} acc={a:.4f} prec={p_:.4f} rec={r_:.4f} f1={f_:.4f}", end="")
        if vname != "binary":
            print(f" | best thr={bt} f1={f2_:.4f}")
        else:
            print()

with open("baselines.json", "w", encoding="utf-8") as f:
    json.dump({"judge_model": JUDGE_MODEL, "device": DEVICE, "sets": base_report},
              f, ensure_ascii=False, indent=2)
print("[done] baselines.json")

#%% python
os.makedirs("best_adapter", exist_ok=True)
model.lora_model.save_pretrained("best_adapter")
torch.save(model.head.state_dict(), "best_adapter/head.pt")

yv, pv = evaluate(model, val_dl)
yt, pt = evaluate(model, test_dl)
report = {
    "base_model": BASE_MODEL,
    "hyperparams": {"epochs": EPOCHS, "batch": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                    "lr": LR, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "seed": SEED},
    "class_weight": [round(float(x), 3) for x in w],
    "device": DEVICE,
    "threshold_chosen": best_t,
    "val": {"at_0.5": metrics(yv, pv, 0.5), "at_best_thr": metrics(yv, pv, best_t)},
    "test": {"at_0.5": metrics(yt, pt, 0.5), "at_best_thr": metrics(yt, pt, best_t)},
    "history": hist,
}
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("test @0.5  :", report["test"]["at_0.5"])
print("test @thr  :", report["test"]["at_best_thr"])

shutil.make_archive("best_adapter", "zip", "best_adapter")
try:
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("router-lora-classifier")
    with mlflow.start_run(run_name="lora-classifier") as _run:
        mlflow.log_params({**report["hyperparams"], "base_model": BASE_MODEL,
                           "class_weight": report["class_weight"],
                           "device": DEVICE, "debug_steps": DEBUG_STEPS})
        for _h in hist:
            mlflow.log_metrics({_k: float(_h[_k]) for _k in
                                ("train_loss", "val_acc", "val_prec", "val_rec", "val_f1")},
                               step=_h["epoch"])
        mlflow.log_metrics({"best_thr": float(best_t), "best_thr_f1": float(best_f1)})
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("baselines.json")
        mlflow.log_artifact("best_adapter.zip")
    print("mlflow run:", _run.info.run_id)
except Exception as _e:
    print("mlflow skipped:", type(_e).__name__)
print("[done] metrics.json + baselines.json + best_adapter.zip")

#%% python
try:
    from google.colab import files
    files.download("best_adapter.zip")
    files.download("metrics.json")
    files.download("baselines.json")
    print("下载完成: best_adapter.zip + metrics.json + baselines.json 请存回 outputs/")
except ImportError:
    print("非 Colab 环境:跳过自动下载,产物已在本目录(可手动拷入 outputs/)")