#!/usr/bin/env python3
"""Server lifecycle tests: overload (P1 #24), client drop (P1 #27),
SIGTERM drain (P1 #26), and exit leaks / sanitizer checks (P1 #25).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time


# Shutdown bound derivation:
# Each slow batch request consists of 8 states and 3 questions. On CPU with 2 worker
# engines (--workers 2), an 8-state batch takes ~2.5 s in Release (~11 s under ASan).
# When 4 slow batch requests are in flight, 2 requests run concurrently per wave.
# Draining all 4 requests therefore requires 2 consecutive waves: ~5 s in Release
# (~22 s under ASan).
# We bound graceful shutdown to 45.0 s, which provides comfortable headroom above the
# ~22 s ASan drain time while remaining well below the 120 s default inference timeout.
# Bounds scale with the measured time of one unloaded slow batch (SLOW_SECONDS, set by the
# overload test, which runs first): a sanitizer build on a shared CI runner is 3-6x slower than a
# release build on a workstation, and fixed seconds made the test flaky there. A real hang still
# fails: it never ends, while every bound here is a small multiple of a measured request.
SLOW_SECONDS = None


def client_timeout():
    # a request may wait for a dropped or queued slow batch ahead of it on the same worker
    return max(30.0, 6 * (SLOW_SECONDS or 10.0) + 30)


def shutdown_bound():
    # 4 slow batches on 2 workers drain in 2 waves; allow 2x for contention between them
    return max(45.0, 4 * (SLOW_SECONDS or 10.0) + 30)


def drain_inference_timeout():
    # The drain cases test shutdown, not the deadline: give admitted requests room to finish, so a
    # slow sanitizer build (ThreadSanitizer is ~12x slower) does not turn a drained request into a
    # 422 "inference deadline exceeded". Never below the server default of 120 s.
    return int(max(120, 8 * (SLOW_SECONDS or 10.0) + 60))


def reserve_port(host='127.0.0.1'):
    probe = socket.socket()
    try:
        probe.bind((host, 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


def assert_answers_equal(a, b):
    assert a['model'] == b['model'] and a['routing'] == b['routing'] and a['usage'] == b['usage']
    assert a['answers'].keys() == b['answers'].keys()
    for qid in a['answers']:
        left, right = a['answers'][qid], b['answers'][qid]
        assert left['type'] == right['type']
        if left['type'] == 'choice':
            assert left['choice'] == right['choice']
        for field in ('probabilities',):
            if field in left:
                assert left[field].keys() == right[field].keys()
                for key in left[field]:
                    assert abs(left[field][key] - right[field][key]) <= 1e-4, (qid, key, left, right)
        for field in ('score', 'noul', 'confidence', 'answer_confidence'):
            if field in left:
                assert abs(left[field] - right[field]) <= 1e-4, (qid, field, left, right)
        assert abs(left['action']['act_probability'] - right['action']['act_probability']) <= 1e-4
        if 'probabilities' in left:
            assert max(left['probabilities'], key=left['probabilities'].get) == \
                   max(right['probabilities'], key=right['probabilities'].get)


def assert_batch_answers_equal(a, b):
    assert 'results' in a and 'results' in b
    assert len(a['results']) == len(b['results'])
    for left, right in zip(a['results'], b['results']):
        assert_answers_equal(left, right)


class ServerInstance:
    """Manages a running statim serve process and asserts clean exit / sanitizers."""
    def __init__(self, binary, model, log_path, max_concurrent=8, workers=2, threads=4, extra_args=None):
        self.binary = binary
        self.model = model
        self.log_path = Path(log_path)
        self.max_concurrent = max_concurrent
        self.workers = workers
        self.threads = threads
        self.extra_args = extra_args or []
        self.port = reserve_port()
        self.proc = None
        self.log_file = None

    def start(self):
        cmd = [
            self.binary, 'serve',
            '-m', f'multilingual={self.model}',
            '--device', 'cpu',
            '--threads', str(self.threads),
            '--workers', str(self.workers),
            '--port', str(self.port),
            '--max-concurrent', str(self.max_concurrent),
            '--no-access-log'
        ] + self.extra_args

        env = dict(os.environ, STATIM_DEVICE='cpu', CUDA_VISIBLE_DEVICES='')
        env.pop('STATIM_API_KEY', None)
        env.pop('STATIM_GPU_FAST', None)

        self.log_file = self.log_path.open('w+')
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log_file, stderr=self.log_file)

        # Wait for /health 200
        for _ in range(300):
            if self.proc.poll() is not None:
                raise AssertionError(f'Server died prematurely:\n{self.log_path.read_text()}')
            try:
                conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=1)
                conn.request('GET', '/health')
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                if resp.status == 200:
                    health = json.loads(body)
                    assert health.get('status') == 'ok'
                    return
            except (OSError, http.client.HTTPException):
                time.sleep(0.05)
        raise AssertionError(f'Server did not become healthy:\n{self.log_path.read_text()}')

    def assert_clean_exit(self, expected_code=0):
        assert self.proc is not None
        code = self.proc.poll()
        assert code is not None, 'Server process is still running'
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
        log_text = self.log_path.read_text()
        assert code == expected_code, f'Expected exit status {expected_code}, got {code}. Log:\n{log_text}'
        assert 'ERROR: AddressSanitizer' not in log_text, f'AddressSanitizer error:\n{log_text}'
        assert 'ERROR: LeakSanitizer' not in log_text, f'LeakSanitizer error:\n{log_text}'
        assert 'runtime error:' not in log_text, f'UBSan error:\n{log_text}'

    def stop(self, sig=signal.SIGTERM):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=shutdown_bound())
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                raise AssertionError(f'Server did not stop within {shutdown_bound():.0f}s on signal {sig}')
        self.assert_clean_exit(expected_code=0)


def test_overload(binary, model, tmp_dir, golden_payload_slow):
    """P1 #24: Overload beyond --max-concurrent returns 503 Retry-After: 1, drains to 200."""
    print('Testing overload (P1 #24)...', flush=True)
    log_path = Path(tmp_dir) / 'overload.log'
    # Start with --max-concurrent 2
    server = ServerInstance(binary, model, log_path, max_concurrent=2, workers=1, threads=2)
    server.start()
    try:
        body_bytes = json.dumps(golden_payload_slow).encode()

        # Measure a single unloaded run to verify slow inference and obtain baseline
        t0 = time.monotonic()
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=600)
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        baseline_bytes = resp.read()
        conn.close()
        elapsed = time.monotonic() - t0
        assert resp.status == 200, f'Baseline request failed: {resp.status}'
        global SLOW_SECONDS
        SLOW_SECONDS = elapsed
        baseline_ans = json.loads(baseline_bytes)
        print(f'  Single unloaded slow batch took {elapsed:.2f}s (inference verified slow enough)')
        assert elapsed >= 0.5, f'Inference was unexpectedly fast: {elapsed:.2f}s'

        def send_request(idx):
            c = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
            c.request('POST', '/v1/systemone/batch', body_bytes,
                      {'Content-Type': 'application/json', 'X-Request-Id': f'overload-{idx}'})
            r = c.getresponse()
            raw_body = r.read()
            headers = dict(r.getheaders())
            c.close()
            return r.status, headers, raw_body

        # Send at least 8 concurrent slow requests
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(send_request, i) for i in range(8)]

            # Saturated /ready check while slots are busy
            time.sleep(0.15)
            rconn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
            rconn.request('GET', '/ready')
            rresp = rconn.getresponse()
            rbody = rresp.read()
            rconn.close()
            assert rresp.status == 503, f'Expected /ready 503 while saturated, got {rresp.status}'
            ready_json = json.loads(rbody)
            assert ready_json == {'ready': False}, f'Unexpected /ready body: {ready_json}'

            results = [f.result() for f in futures]

        statuses = [res[0] for res in results]
        print(f'  8 concurrent requests completed with statuses: {statuses}')

        # Expect at least one 503 with Retry-After: 1 and exact detail JSON.
        # No other status codes, no connection resets.
        assert 503 in statuses, f'Expected at least one 503 under overload, got: {statuses}'
        for status, headers, body in results:
            if status == 503:
                assert headers.get('Retry-After') == '1', f"Expected Retry-After: 1, got {headers.get('Retry-After')}"
                body_json = json.loads(body)
                assert body_json == {'detail': 'server busy, try again later'}, f'Unexpected 503 detail: {body_json}'
            elif status == 200:
                ans = json.loads(body)
                assert_batch_answers_equal(ans, baseline_ans)
            else:
                raise AssertionError(f'Unexpected status code {status}: {body}')

        # After the load drains, the same request answers 200. /ready and /health answer 200.
        rconn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
        rconn.request('GET', '/ready')
        rresp = rconn.getresponse()
        assert rresp.status == 200, f'Expected /ready 200 after drain, got {rresp.status}'
        assert json.loads(rresp.read()) == {'ready': True}
        rconn.close()

        hconn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
        hconn.request('GET', '/health')
        hresp = hconn.getresponse()
        assert hresp.status == 200, f'Expected /health 200, got {hresp.status}'
        hconn.close()

        # Follow-up request answers 200 and matches baseline
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        post_resp = conn.getresponse()
        post_body = post_resp.read()
        conn.close()
        assert post_resp.status == 200, f'Post-drain request failed: {post_resp.status}'
        assert_batch_answers_equal(json.loads(post_body), baseline_ans)
        print('  Overload test passed cleanly.')
    finally:
        server.stop()


def test_client_drop(binary, model, tmp_dir, golden_payload_slow):
    """P1 #27/#54: client drop cancels compute and leaves the server healthy."""
    print('Testing client drop (P1 #27)...', flush=True)
    log_path = Path(tmp_dir) / 'client_drop.log'
    server = ServerInstance(binary, model, log_path, max_concurrent=4, workers=1, threads=2)
    server.start()
    try:
        body_bytes = json.dumps(golden_payload_slow).encode()

        # Baseline clean run
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        baseline_resp = conn.getresponse()
        baseline_ans = json.loads(baseline_resp.read())
        conn.close()
        assert baseline_resp.status == 200

        quick_payload = {
            'state': 'ok',
            'questions': {'q': {'type': 'choice', 'instructions': 'choose', 'criteria': ['yes', 'no']}}
        }
        quick_bytes = json.dumps(quick_payload).encode()
        qt0 = time.monotonic()
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
        conn.request('POST', '/v1/systemone', quick_bytes, {'Content-Type': 'application/json'})
        quick_resp = conn.getresponse()
        quick_baseline = json.loads(quick_resp.read())
        conn.close()
        quick_seconds = time.monotonic() - qt0
        assert quick_resp.status == 200

        req_headers = (
            f'POST /v1/systemone/batch HTTP/1.1\r\n'
            f'Host: 127.0.0.1:{server.port}\r\n'
            f'Content-Type: application/json\r\n'
            f'Content-Length: {len(body_bytes)}\r\n'
            f'Connection: close\r\n\r\n'
        ).encode()

        # Case A: Close socket mid-request BEFORE response
        print('  Case A: Dropping connection before response...', flush=True)
        s1 = socket.create_connection(('127.0.0.1', server.port), timeout=5)
        s1.sendall(req_headers + body_bytes)
        time.sleep(0.5)  # the slow forward pass is running, not merely queued
        s1.close()

        # With one worker this is blocked for nearly SLOW_SECONDS unless the dead
        # forward pass cooperatively stops. Scale the bound from both unloaded runs.
        follow_t0 = time.monotonic()
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
        conn.request('POST', '/v1/systemone', quick_bytes, {'Content-Type': 'application/json'})
        follow_resp = conn.getresponse()
        follow_answer = json.loads(follow_resp.read())
        conn.close()
        follow_seconds = time.monotonic() - follow_t0
        cancel_bound = max(1.0, 4 * quick_seconds, 0.6 * SLOW_SECONDS)
        assert follow_resp.status == 200
        assert_answers_equal(follow_answer, quick_baseline)
        assert follow_seconds < cancel_bound, \
            f'follow-up took {follow_seconds:.2f}s; cancellation bound is {cancel_bound:.2f}s (slow={SLOW_SECONDS:.2f}s)'
        for _ in range(100):
            if '"event":"inference_cancelled"' in log_path.read_text():
                break
            time.sleep(0.01)
        assert '"event":"inference_cancelled"' in log_path.read_text(), 'missing inference_cancelled log event'

        # Server must still be healthy and answer follow-up request normally
        hconn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
        hconn.request('GET', '/health')
        assert hconn.getresponse().status == 200
        hconn.close()

        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        r = conn.getresponse()
        ans1 = json.loads(r.read())
        conn.close()
        assert r.status == 200
        assert_batch_answers_equal(ans1, baseline_ans)

        # Case B: Close socket mid-request WHILE response is being read
        print('  Case B: Dropping connection while reading response...', flush=True)
        s2 = socket.create_connection(('127.0.0.1', server.port), timeout=client_timeout())
        s2.sendall(req_headers + body_bytes)
        # Read partial bytes of HTTP response header
        partial = s2.recv(25)
        assert len(partial) > 0, 'Did not receive any response bytes before drop'
        s2.close()
        time.sleep(0.5)

        # Server must still be healthy and answer follow-up request normally
        hconn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
        hconn.request('GET', '/health')
        assert hconn.getresponse().status == 200
        hconn.close()

        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        r = conn.getresponse()
        ans2 = json.loads(r.read())
        conn.close()
        assert r.status == 200
        assert_batch_answers_equal(ans2, baseline_ans)
        print('  Client drop tests passed cleanly.')
    finally:
        server.stop()


def test_microbatch_client_drop(binary, model, tmp_dir, golden_payload_slow):
    """One departed item must not cancel a shared forward pass with a live item."""
    print('Testing client drop from a shared micro-batch...', flush=True)
    log_path = Path(tmp_dir) / 'microbatch_client_drop.log'
    server = ServerInstance(binary, model, log_path, max_concurrent=4, workers=1, threads=2,
                            extra_args=['--batch-window-ms', '250', '--max-batch', '8'])
    server.start()
    try:
        payload = {'state': golden_payload_slow['states'],
                   'questions': golden_payload_slow['questions']}
        body = json.dumps(payload).encode()

        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
        conn.request('POST', '/v1/systemone', body, {'Content-Type': 'application/json'})
        baseline_resp = conn.getresponse()
        baseline = json.loads(baseline_resp.read())
        conn.close()
        assert baseline_resp.status == 200

        headers = (
            f'POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n'
            f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
            f'X-Request-Id: microbatch-drop\r\nConnection: close\r\n\r\n'
        ).encode()
        dropped = socket.create_connection(('127.0.0.1', server.port), timeout=5)
        dropped.sendall(headers + body)

        def live_request():
            c = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
            c.request('POST', '/v1/systemone', body,
                      {'Content-Type': 'application/json', 'X-Request-Id': 'microbatch-live'})
            r = c.getresponse()
            answer = json.loads(r.read())
            c.close()
            return r.status, answer

        with ThreadPoolExecutor(max_workers=1) as pool:
            live = pool.submit(live_request)
            # Wait until the shared batch owns the only engine, then drop one peer.
            busy_seen = False
            for _ in range(200):
                metrics = http.client.HTTPConnection('127.0.0.1', server.port, timeout=2)
                metrics.request('GET', '/metrics')
                metrics_resp = metrics.getresponse()
                metrics_text = metrics_resp.read().decode()
                metrics.close()
                if 'statim_workers_busy{model="multilingual"} 1' in metrics_text:
                    busy_seen = True
                    break
                time.sleep(0.005)
            assert busy_seen, 'shared micro-batch never started inference'
            dropped.close()
            status, answer = live.result()
        assert status == 200
        assert_answers_equal(answer, baseline)
        # Baseline contributed size 1 and the shared forward pass size 2.
        batch_sum = next(float(line.split()[1]) for line in metrics_text.splitlines()
                         if line.startswith('statim_batch_size_sum '))
        assert batch_sum >= 3, f'requests did not share a batch: statim_batch_size_sum={batch_sum}'
        print('  Live item completed correctly after its batch peer disconnected.')
    finally:
        server.stop()


def run_quick_inferences(port, count=200, workers=4):
    """Executes quick inferences to populate engine compute and check for leaks."""
    quick_payload = {
        'state': 'ok',
        'questions': {
            'q1': {'type': 'choice', 'instructions': 'choose', 'criteria': ['yes', 'no']}
        }
    }
    data = json.dumps(quick_payload).encode()

    def worker_loop(n):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
        for _ in range(n):
            conn.request('POST', '/v1/systemone', data, {'Content-Type': 'application/json'})
            r = conn.getresponse()
            r.read()
            assert r.status == 200
        conn.close()

    per_worker = count // workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker_loop, [per_worker] * workers))


def run_sigterm_drain_case(binary, model, tmp_dir, golden_payload_slow, sig, run_label, measured_times, quick=20):
    """Executes a single signal drain case: quick inferences -> 4 slow requests -> signal -> drain."""
    log_path = Path(tmp_dir) / f'drain_{run_label}.log'
    server = ServerInstance(binary, model, log_path, max_concurrent=8, workers=2, threads=4,
                            extra_args=['--inference-timeout', str(drain_inference_timeout())])
    server.start()
    try:
        # Quick inferences first, so the leak check at exit (P1 #25) covers real work
        t_quick_start = time.monotonic()
        run_quick_inferences(server.port, count=quick, workers=4)
        t_quick = time.monotonic() - t_quick_start

        # Start 4 slow requests
        body_bytes = json.dumps(golden_payload_slow).encode()

        def send_slow(idx):
            try:
                conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=client_timeout())
                conn.request('POST', '/v1/systemone/batch', body_bytes,
                             {'Content-Type': 'application/json', 'X-Request-Id': f'{run_label}-{idx}'})
                r = conn.getresponse()
                raw_body = r.read()
                conn.close()
                return r.status, raw_body
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                # Clean connection close is acceptable on shutdown
                return 0, b''

        executor = ThreadPoolExecutor(max_workers=4)
        futures = [executor.submit(send_slow, i) for i in range(4)]

        # Let requests start executing
        time.sleep(0.3)

        # Send signal while slow requests run
        t0 = time.monotonic()
        server.proc.send_signal(sig)

        # Immediately assert that a new connection after the signal is refused
        refused = False
        for _ in range(20):
            try:
                s = socket.create_connection(('127.0.0.1', server.port), timeout=0.5)
                s.close()
                time.sleep(0.05)
            except (ConnectionRefusedError, OSError):
                refused = True
                break
        assert refused, 'New connection was NOT refused after signal!'

        # Process must exit with status 0 within bound
        try:
            server.proc.wait(timeout=shutdown_bound())
        except subprocess.TimeoutExpired:
            server.proc.kill()
            server.proc.wait()
            raise AssertionError(f'{run_label}: server hung after {sig}, exceeded bound {shutdown_bound():.0f}s')

        drain_duration = time.monotonic() - t0
        measured_times.append(drain_duration)
        assert drain_duration <= shutdown_bound(), \
            f'{run_label}: shutdown took {drain_duration:.2f}s, exceeding bound {shutdown_bound():.0f}s'

        # Requests admitted before the signal are drained: each one completes with a full 200 answer.
        # (DEPLOY.md documents this; a dropped connection here would contradict it.)
        for idx, fut in enumerate(futures):
            status, raw_body = fut.result()
            assert status == 200, f'{run_label} req {idx}: in-flight request not drained (status {status}): {raw_body[:200]}'
            body_json = json.loads(raw_body)
            assert len(body_json.get('results', [])) == len(golden_payload_slow['states']), \
                f'{run_label} req {idx}: incomplete results in {str(body_json)[:200]}'

        # Assert exit status 0 and stderr contains shutdown event and no sanitizer errors
        server.assert_clean_exit(expected_code=0)
        log_text = server.log_path.read_text()
        assert '\"event\":\"shutdown\"' in log_text, f'{run_label}: missing shutdown event in log:\n{log_text}'
        print(f'  {run_label} ({sig.name}): {quick} quick inferences in {t_quick:.2f}s; drained in {drain_duration:.2f}s, exit 0')
    finally:
        if server.proc and server.proc.poll() is None:
            server.proc.kill()
            server.proc.wait()


def test_signals(binary, model, tmp_dir, golden_payload_slow):
    """P1 #26: SIGTERM/SIGINT drain: exits within bound, in-flight requests complete/close cleanly,
    new connections refused, shutdown event logged, repeated 5 times for SIGTERM + 1 SIGINT."""
    print('Testing signals and graceful drain (P1 #26)...', flush=True)
    measured_times = []

    # SIGINT once, after a few hundred inferences (P1 #25: leaks at exit after real work); the
    # SIGTERM repeats look for races in the drain and keep the warm-up short, so the suite fits a
    # sanitizer build on a small CI runner
    run_sigterm_drain_case(binary, model, tmp_dir, golden_payload_slow, signal.SIGINT, 'sigint-1', measured_times, quick=200)

    # Repeat SIGTERM 5 times to catch races
    for i in range(1, 6):
        run_sigterm_drain_case(binary, model, tmp_dir, golden_payload_slow, signal.SIGTERM, f'sigterm-{i}', measured_times)

    print(f'  Shutdown drain times over 5 SIGTERM runs: '
          f'min={min(measured_times[1:]):.2f}s, max={max(measured_times[1:]):.2f}s, '
          f'avg={sum(measured_times[1:])/5:.2f}s (bound: {shutdown_bound():.0f}s)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build-rel/statim', help='Path to statim binary')
    parser.add_argument('--model', default='models/laya-multilingual-f32.gguf', help='Path to GGUF model')
    parser.add_argument('--inputs', default='tests/data/golden_inputs.json', help='Path to golden inputs')
    args = parser.parse_args()

    # Skip 77 if binary or model is missing
    if not os.path.isfile(args.binary):
        print(f'SKIP: binary not found at {args.binary}')
        return 77
    if not os.path.isfile(args.model):
        print(f'SKIP: model not found at {args.model}')
        return 77
    if not os.path.isfile(args.inputs):
        print(f'SKIP: inputs not found at {args.inputs}')
        return 77

    # Skip 77 if sandbox cannot bind localhost
    try:
        p = reserve_port()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', p))
    except PermissionError:
        print('SKIP: sandbox cannot bind localhost')
        return 77

    # Load golden inputs for slow requests (8 states, 3 questions)
    with open(args.inputs) as f:
        fixture = json.load(f)
    states = [s for s in fixture['states'] if s != ''][:8]
    questions = {k: fixture['questions'][k] for k in list(fixture['questions'])[:3]}
    golden_payload_slow = {'states': states, 'questions': questions}

    with tempfile.TemporaryDirectory(prefix='statim-lifecycle-') as tmp_dir:
        test_overload(args.binary, args.model, tmp_dir, golden_payload_slow)
        test_client_drop(args.binary, args.model, tmp_dir, golden_payload_slow)
        test_microbatch_client_drop(args.binary, args.model, tmp_dir, golden_payload_slow)
        test_signals(args.binary, args.model, tmp_dir, golden_payload_slow)

    print('server_lifecycle: ALL TESTS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
