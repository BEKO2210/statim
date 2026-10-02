# Soak test: FAIL

- Command: `./statim serve -m m=models/laya-multilingual-v9-f32.gguf --port 43113 --workers 2 --max-concurrent 16 --no-access-log --adapter m:pii=models/pii.lora.gguf --adapter m:emotion=models/emotion.lora.gguf --threads 4`
- Duration: 24.023 h of 24.0 h, 6 clients, 24153 requests
- Statuses: {'200': 21695, 'cancelled': 2428, '422': 30}
- RSS: first hour 2740.9 MiB, last hour 2953.2 MiB, max 3392.4 MiB
- Latency (200s): first hour {'n': 892, 'p50': 18821.324348449707, 'p95': 61492.99168586731}, last hour {'n': 944, 'p50': 17771.61169052124, 'p95': 60294.09742355347}
- Exit: {'sigterm_to_exit_s': 0.26, 'returncode': 0}

| Check | Result |
|---|---|
| only 200 answers (cancellations excepted) | FAIL |
| server stayed up | pass |
| /ready and /health always 200 | pass |
| RSS growth <= 5% (first vs last hour median) | FAIL |
| RSS max <= 6144 MiB | pass |
| open fds stable (+10) | pass |
| p95 drift <= 1.25x | pass |
| SIGTERM exits 0 | pass |
| no sanitizer report | pass |

First unexpected responses:

- `{"t": 1790851917.8800464, "kind": "batch+adapter", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790857718.8106585, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790858225.9144406, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790858804.9112105, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790870995.9232368, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790871921.718642, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790879567.0715501, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790879768.8566852, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790885331.0535007, "kind": "batch+adapter", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790890729.1486294, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790893917.4643571, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790897455.239956, "kind": "batch+adapter", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790897895.7378516, "kind": "batch+adapter", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790901521.4577682, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790902144.6037202, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790902173.9917245, "kind": "batch+adapter", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790909611.35366, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790910731.1024265, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790910763.7152195, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
- `{"t": 1790911184.5857666, "kind": "batch", "status": 422, "body": "{\"detail\":\"inference deadline exceeded\"}"}`
