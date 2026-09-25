"""Latency of the official Laya package on this machine (CPU, fp32), same inputs as golden_inputs.json."""
import json, os, sys, time
import torch, laya
n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
torch.set_num_threads(n_threads)
inp = json.load(open("tests/data/golden_inputs.json"))
agent = laya.load(os.path.abspath("models/" + (sys.argv[1] if len(sys.argv) > 1 else "laya-multilingual")), device="cpu")
states = inp["states"][: int(os.environ.get("N_STATES", "30"))]
agent.system_one(states[0], inp["questions"])  # warm-up
ts = []
for s in states:
    t = time.perf_counter(); agent.system_one(s, inp["questions"]); ts.append((time.perf_counter() - t) * 1000)
ts_sorted = sorted(ts)
print(json.dumps({"engine": "laya-python", "threads": n_threads, "states": len(ts), "mean_ms": sum(ts) / len(ts),
                  "p50_ms": ts_sorted[len(ts) // 2], "per_state_ms": [round(x, 1) for x in ts]}))
