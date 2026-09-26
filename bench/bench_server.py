#!/usr/bin/env python3
"""HTTP load test for any /v1/systemone server (Statim or laya-serve).

    python bench/bench_server.py --url http://127.0.0.1:8412 --concurrency 1 4 --requests 60
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--inputs", default="tests/data/golden_inputs.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    inp = json.load(open(a.inputs))
    states = [s for s in inp["states"] if s != ""][:25]
    extra = {"model": a.model} if a.model else {}
    bodies = [json.dumps({"state": s, "questions": inp["questions"], **extra}).encode() for s in states]
    urllib.request.urlopen(urllib.request.Request(a.url + "/v1/systemone", data=bodies[0]), timeout=600).read()  # warm-up
    for c in a.concurrency:
        lat, errors, lock, idx = [], 0, threading.Lock(), [0]

        def worker():
            nonlocal errors
            while True:
                with lock:
                    if idx[0] >= a.requests:
                        return
                    i = idx[0]
                    idx[0] += 1
                t = time.perf_counter()
                try:
                    urllib.request.urlopen(urllib.request.Request(a.url + "/v1/systemone", data=bodies[i % len(bodies)]),
                                           timeout=600).read()
                    with lock:
                        lat.append((time.perf_counter() - t) * 1000)
                except Exception:
                    with lock:
                        errors += 1

        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker) for _ in range(c)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - t0
        lat.sort()
        print(json.dumps({"server": a.label, "concurrency": c, "requests": len(lat), "errors": errors,
                          "throughput_rps": round(len(lat) / wall, 3), "p50_ms": round(statistics.median(lat), 1),
                          "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1)}), flush=True)


if __name__ == "__main__":
    main()
