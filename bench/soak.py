#!/usr/bin/env python3
"""Soak test: one `statim serve` under mixed load for hours, watched for leaks, drift and errors.

    python3 bench/soak.py --binary build/statim --model models/laya-multilingual-f32.gguf \\
        --inputs tests/data/golden_inputs.json --hours 24 --out soak.json --report soak.md
    # optional: --adapter pii=models/pii.lora.gguf (adds requests that switch weight sets)

Load: `--clients` threads in a closed loop, each picking at random:
- single decisions with a random state and subset of questions;
- batches of 2-32 states;
- the same through an adapter (with `--adapter`), so engines switch weight sets;
- cancellations: a request whose connection is closed before the answer arrives.

Every `--sample-sec` the monitor records RSS, threads and open file descriptors of the server
process and polls /ready and /health. The run passes when:
- every answer is 200 (cancellations excepted) and the server never exits on its own;
- /ready and /health always answer 200 (the clients stay below --max-concurrent);
- RSS stays under --rss-ceiling-mib, and the median of the last hour exceeds the median of the
  first hour after warm-up by at most --rss-growth (default 5 %);
- open file descriptors and threads return to their warm-up level (+10 fds);
- p95 latency of the last hour is at most --latency-drift (default 1.25) times the first hour's;
- SIGTERM at the end exits 0, and stderr has no sanitizer report.
The JSON (--out) is rewritten every sample, so an interrupted run keeps its data. Standard library
only; Linux (/proc).
"""
import argparse
import http.client
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

SANITIZER_MARKERS = ("ERROR: AddressSanitizer", "ERROR: LeakSanitizer", "runtime error:")


def reserve_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def proc_stats(pid):
    rss = threads = None
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
            elif line.startswith("Threads:"):
                threads = int(line.split()[1])
    fds = len(os.listdir(f"/proc/{pid}/fd"))
    return rss, threads, fds


def get_status(port, path):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", path)
        status = c.getresponse().status
        c.close()
        return status
    except OSError:
        return None


class Load:
    def __init__(self, port, states, questions, adapters, seed):
        self.port, self.states, self.questions, self.adapters = port, states, questions, adapters
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.samples = []      # (t, kind, status, ms)
        self.unexpected = []   # first few unexpected responses, for the report

    def payload(self, rng):
        qids = rng.sample(list(self.questions), rng.randint(1, len(self.questions)))
        questions = {q: self.questions[q] for q in qids}
        kind = rng.choices(["single", "batch", "cancel"], [0.6, 0.3, 0.1])[0]
        adapter = rng.choice(self.adapters) if self.adapters and rng.random() < 0.4 else None
        if kind == "batch":
            body = {"states": [rng.choice(self.states) for _ in range(rng.randint(2, 32))], "questions": questions}
            path = "/v1/systemone/batch"
        else:
            body = {"state": rng.choice(self.states), "questions": questions}
            path = "/v1/systemone"
        if adapter:
            body["adapter"] = adapter
        return kind + ("+adapter" if adapter else ""), path, body

    def one(self, rng):
        kind, path, body = self.payload(rng)
        t0 = time.time()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=300)
        try:
            conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            if kind.startswith("cancel"):
                time.sleep(rng.uniform(0, 0.2))
                conn.close()  # the client goes away before reading the answer
                status = "cancelled"
            else:
                res = conn.getresponse()
                data = res.read()
                status = res.status
                if status != 200 or not json.loads(data):
                    with self.lock:
                        if len(self.unexpected) < 20:
                            self.unexpected.append({"t": t0, "kind": kind, "status": status, "body": data[:300].decode("utf-8", "replace")})
        except (OSError, ValueError, http.client.HTTPException) as e:
            status = "error:" + type(e).__name__
            with self.lock:
                if len(self.unexpected) < 20:
                    self.unexpected.append({"t": t0, "kind": kind, "status": status, "body": str(e)[:300]})
        finally:
            conn.close()
        with self.lock:
            self.samples.append((t0, kind, status, (time.time() - t0) * 1000))

    def worker(self, idx, stop):
        rng = random.Random(self.rng.random() + idx)
        while not stop.is_set():
            self.one(rng)


def window(samples, t0, t1):
    ms = [s[3] for s in samples if t0 <= s[0] < t1 and s[2] == 200]
    if len(ms) < 20:
        return None
    ms.sort()
    return {"n": len(ms), "p50": ms[len(ms) // 2], "p95": ms[int(len(ms) * 0.95)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True, help="GGUF served as model 'm'")
    ap.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--inputs", required=True, help="tests/data/golden_inputs.json")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-concurrent", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0, help="statim --threads (0: its default)")
    ap.add_argument("--sample-sec", type=float, default=30.0)
    ap.add_argument("--warmup-min", type=float, default=10.0)
    ap.add_argument("--rss-ceiling-mib", type=float, default=6144.0)
    ap.add_argument("--rss-growth", type=float, default=0.05)
    ap.add_argument("--latency-drift", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    if a.clients >= a.max_concurrent:
        ap.error("--clients must stay below --max-concurrent, or /ready legitimately answers 503")

    fixture = json.load(open(a.inputs, encoding="utf-8"))
    states = [s for s in fixture["states"] if s != ""]
    questions = fixture["questions"]
    port = reserve_port()
    cmd = [a.binary, "serve", "-m", f"m={a.model}", "--port", str(port), "--workers", str(a.workers),
           "--max-concurrent", str(a.max_concurrent), "--no-access-log"]
    adapters = []
    for spec in a.adapter:
        name, path = spec.split("=", 1)
        cmd += ["--adapter", f"m:{name}={path}"]
        adapters.append(name)
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    err_path = os.path.splitext(a.out)[0] + ".stderr.log"
    err = open(err_path, "w")
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
    for _ in range(300):
        if get_status(port, "/ready") == 200:
            break
        if srv.poll() is not None:
            sys.exit(f"soak: server exited during start-up ({srv.returncode}); see {err_path}")
        time.sleep(1)
    else:
        sys.exit("soak: server not ready after 300 s")

    load = Load(port, states, questions, adapters, a.seed)
    stop = threading.Event()
    threads = [threading.Thread(target=load.worker, args=(i, stop), daemon=True) for i in range(a.clients)]
    start = time.time()
    end = start + a.hours * 3600
    for t in threads:
        t.start()
    series, probe_failures, died = [], [], None
    print(f"soak: {a.hours} h, {a.clients} clients, server pid {srv.pid}, port {port}", flush=True)

    def summary(final=False):
        warm = start + a.warmup_min * 60
        hour = 3600 if a.hours >= 2 else max(a.hours * 3600 / 4, 60)
        after = [s for s in series if s["t"] >= warm]
        first = [s for s in after if s["t"] < warm + hour]
        last = [s for s in after if s["t"] >= series[-1]["t"] - hour] if series else []
        with load.lock:
            samples = list(load.samples)
            unexpected = list(load.unexpected)
        statuses = {}
        for s in samples:
            statuses[str(s[2])] = statuses.get(str(s[2]), 0) + 1
        res = {"command": cmd, "hours": a.hours, "elapsed_h": round((time.time() - start) / 3600, 3),
               "clients": a.clients, "requests": len(samples), "statuses": statuses,
               "unexpected": unexpected, "probe_failures": probe_failures[:50], "server_died": died,
               "series": series}
        checks = {}
        bad_status = sum(n for k, n in statuses.items() if k not in ("200", "cancelled"))
        checks["only 200 answers (cancellations excepted)"] = bad_status == 0
        checks["server stayed up"] = died is None
        checks["/ready and /health always 200"] = not probe_failures
        if first and last:
            r1 = statistics.median(s["rss_mib"] for s in first)
            r2 = statistics.median(s["rss_mib"] for s in last)
            res["rss_first_hour_mib"], res["rss_last_hour_mib"] = round(r1, 1), round(r2, 1)
            checks[f"RSS growth <= {a.rss_growth:.0%} (first vs last hour median)"] = r2 <= r1 * (1 + a.rss_growth)
            checks[f"RSS max <= {a.rss_ceiling_mib:.0f} MiB"] = max(s["rss_mib"] for s in series) <= a.rss_ceiling_mib
            f1, f2 = first[0]["fds"], last[-1]["fds"]
            checks["open fds stable (+10)"] = f2 <= f1 + 10
            w1 = window(samples, first[0]["t"], first[-1]["t"])
            w2 = window(samples, last[0]["t"], last[-1]["t"] + 1)
            res["latency_first_hour"], res["latency_last_hour"] = w1, w2
            if w1 and w2:
                checks[f"p95 drift <= {a.latency_drift}x"] = w2["p95"] <= w1["p95"] * a.latency_drift
        res["checks"] = checks
        res["final"] = final
        return res

    def write(res):
        tmp = a.out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)

    try:
        while time.time() < end:
            if srv.poll() is not None:
                died = {"t": time.time(), "returncode": srv.returncode}
                break
            rss, nthreads, fds = proc_stats(srv.pid)
            ready, health = get_status(port, "/ready"), get_status(port, "/health")
            now = time.time()
            if ready != 200 or health != 200:
                probe_failures.append({"t": now, "ready": ready, "health": health})
            with load.lock:
                n = len(load.samples)
            series.append({"t": now, "rss_mib": round(rss, 1), "threads": nthreads, "fds": fds,
                           "ready": ready, "health": health, "requests": n})
            write(summary())
            time.sleep(a.sample_sec)
    except KeyboardInterrupt:
        print("soak: interrupted, writing what was measured", flush=True)
    stop.set()
    for t in threads:
        t.join(timeout=320)
    exit_info = {}
    if srv.poll() is None:
        t0 = time.time()
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
            exit_info["killed_after_timeout"] = True
        exit_info["sigterm_to_exit_s"] = round(time.time() - t0, 2)
    exit_info["returncode"] = srv.returncode
    err.close()
    stderr = open(err_path, encoding="utf-8", errors="replace").read()
    res = summary(final=True)
    res["exit"] = exit_info
    res["checks"]["SIGTERM exits 0"] = srv.returncode == 0 and not exit_info.get("killed_after_timeout")
    res["checks"]["no sanitizer report"] = not any(m in stderr for m in SANITIZER_MARKERS)
    res["pass"] = all(res["checks"].values())
    write(res)

    lines = [f"# Soak test: {'PASS' if res['pass'] else 'FAIL'}", "",
             f"- Command: `{' '.join(cmd)}`",
             f"- Duration: {res['elapsed_h']} h of {a.hours} h, {a.clients} clients, {res['requests']} requests",
             f"- Statuses: {res['statuses']}",
             f"- RSS: first hour {res.get('rss_first_hour_mib')} MiB, last hour {res.get('rss_last_hour_mib')} MiB, "
             f"max {max((s['rss_mib'] for s in series), default=None)} MiB",
             f"- Latency (200s): first hour {res.get('latency_first_hour')}, last hour {res.get('latency_last_hour')}",
             f"- Exit: {exit_info}", "", "| Check | Result |", "|---|---|"]
    lines += [f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in res["checks"].items()]
    if res["unexpected"]:
        lines += ["", "First unexpected responses:", ""] + [f"- `{json.dumps(u)[:300]}`" for u in res["unexpected"]]
    open(a.report, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
