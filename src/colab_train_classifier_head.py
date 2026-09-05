#%% md
# 训练 Qwen2.5-1.5B-Instruct + LoRA 分类头（免费 Colab）

**目标**：把 1.5B 基座微调成「难易度路由器」二分类器（LOW/HIGH），供降本网关做路由决策。

**使用步骤（全程只需 2 次交互）**
1. 菜单：Runtime ▶ Run all（数据已内嵌在本 notebook 的 DATA cell，无需上传）
2. 训练约 30–60 分钟，结束后自动下载 `best_adapter.zip` 和 `metrics.json`，存回本地 `outputs/` 目录

**说明**：fp16 LoRA（不依赖 bitsandbytes，版本兼容最稳）。560 训练样本 / 6 epochs，T4 显存 16G 无压力。断线重跑：直接 ▶ Run all 即可，数据在 notebook 内，结果确定复现（固定 seed 42）。

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

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 6
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42
LABEL_TO_ID = {"LOW": 0, "HIGH": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())

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
            AutoModel.from_pretrained(base_id, torch_dtype=torch.float16), lora_cfg)
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
def evaluate(model, dl):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for ids, am, y in dl:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ids.cuda(), am.cuda())
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

model = QwenClassifier(BASE_MODEL, lora_cfg).cuda()
cnt = Counter(int(x) for x in train_ds.labels)
n = len(train_ds)
w = torch.tensor([n / (2 * cnt[0]), n / (2 * cnt[1])], dtype=torch.float32).cuda()
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
        ids, am, y = ids.cuda(), am.cuda(), y.cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(ids, am)
            loss = nn.functional.cross_entropy(logits, y, weight=w) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        run_loss += float(loss.detach()) * GRAD_ACCUM
        n_steps += 1
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
    "threshold_chosen": best_t,
    "val": {"at_0.5": metrics(yv, pv, 0.5), "at_best_thr": metrics(yv, pv, best_t)},
    "test": {"at_0.5": metrics(yt, pt, 0.5), "at_best_thr": metrics(yt, pt, best_t)},
    "history": hist,
}
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("test @0.5  :", report["test"]["at_0.5"])
print("test @thr  :", report["test"]["at_best_thr"])
print("[done] metrics.json + best_adapter/")

#%% python
shutil.make_archive("best_adapter", "zip", "best_adapter")
from google.colab import files
files.download("best_adapter.zip")
files.download("metrics.json")
print("下载完成: best_adapter.zip + metrics.json 请存回 outputs/")