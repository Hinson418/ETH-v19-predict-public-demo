# ETH V19 Public Demos (Feature Engineering + Risk Guardrails)

This repository contains **two standalone, runnable demos** distilled from my iterative ETH research workflow (V1 → V19.4).
It is intentionally **public-safe**: no private datasets, no exchange keys, and no proprietary model weights.

## What’s inside

- `feature_demo.py`  
  A compact example of **feature engineering** producing an **80-dim feature vector** from a small synthetic input.

- `risk_guardrails_demo.py`  
  A “risk-first” **validation / guardrails layer** that turns model output into an execution decision:
  **APPROVED / CONDITIONAL / REJECTED**, with explicit rule-based reasons.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
