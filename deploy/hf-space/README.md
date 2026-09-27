---
title: Statim
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Fast typed decisions from text or JSON
models:
  - Beko2210/statim-decide-multilingual-base
---

# Statim

This Space is a free public demo of [Statim](https://github.com/BEKO2210/statim), a native C++ decision engine for text and JSON. It answers several typed questions in one model pass:

- **choice** selects one option.
- **score** places the input on an ordered scale.
- **noul** answers a yes/no statement with a probability.

Open the playground above and try a decision without an API key. The demo runs the [multilingual Statim model](https://huggingface.co/Beko2210/statim-decide-multilingual-base) on Hugging Face's free 2-vCPU hardware. Requests are intentionally small and concurrent work is limited, so cold starts and complex decisions can take longer or return a busy/limit response. This public instance is for evaluation, not production workloads or sensitive data.
