# Hybrid LLM Router & Cost Optimizer

智能路由 + 降本网关：微调一个小型路由模型（Qwen-1.5B），一眼区分请求难度，简单→本地 7B、困难→OpenRouter 免费模型，配合级联容错与 FinOps 计费剖析，帮企业降低大模型 API 成本。全程用免费模型落地，接口预留可扩展为任意付费模型。

> 详细项目计划见 `reports/20260904_0050_LLMRouter_ProjectPlan.md`（reports 目录位于上级 workspace，GitHub 版 README 将于 Phase 6 补齐）。

## 状态

- 项目骨架已建立（见目录结构）
- 开发进行中：Phase 1 数据工程

## 目录结构

```
src/                    核心代码
  openrouter_client.py   OpenRouter 免费模型封装 + 自动轮换 + 付费扩展位
configs/                计费表、模型列表配置
data/                   种子数据 / 黄金集
scripts/                启动脚本
tests/                  测试
```
