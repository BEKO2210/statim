# Soak test: PASS

- Command: `./statim serve -m m=models/laya-multilingual-v9-f32.gguf --port 38995 --workers 2 --max-concurrent 16 --no-access-log --adapter m:pii=models/pii.lora.gguf --adapter m:emotion=models/emotion.lora.gguf --threads 4`
- Duration: 72.006 h of 72.0 h, 2 clients, 72782 requests
- Statuses: {'200': 65361, 'cancelled': 7421}
- RSS: first hour 2931.3 MiB, last hour 2997.0 MiB, max 3427.7 MiB
- Latency (200s): first hour {'n': 1003, 'p50': 1404.860019683838, 'p95': 35609.8198890686}, last hour {'n': 932, 'p50': 1448.378562927246, 'p95': 39896.04330062866}
- Exit: {'sigterm_to_exit_s': 0.31, 'returncode': 0}

| Check | Result |
|---|---|
| only 200 answers (cancellations excepted) | pass |
| server stayed up | pass |
| /ready and /health always 200 | pass |
| RSS growth <= 5% (first vs last hour median) | pass |
| RSS max <= 6144 MiB | pass |
| open fds stable (+10) | pass |
| p95 drift <= 1.25x | pass |
| SIGTERM exits 0 | pass |
| no sanitizer report | pass |
