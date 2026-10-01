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
SHUTDOWN_BOUND_SECONDS = 45.0


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
                self.proc.wait(timeout=SHUTDOWN_BOUND_SECONDS)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                raise AssertionError(f'Server did not stop within {SHUTDOWN_BOUND_SECONDS}s on signal {sig}')
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
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=30)
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        baseline_bytes = resp.read()
        conn.close()
        elapsed = time.monotonic() - t0
        assert resp.status == 200, f'Baseline request failed: {resp.status}'
        baseline_ans = json.loads(baseline_bytes)
        print(f'  Single unloaded slow batch took {elapsed:.2f}s (inference verified slow enough)')
        assert elapsed >= 0.5, f'Inference was unexpectedly fast: {elapsed:.2f}s'

        def send_request(idx):
            c = http.client.HTTPConnection('127.0.0.1', server.port, timeout=60)
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
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=30)
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
    """P1 #27: Client drop mid-request (before response and during read) leaves server healthy."""
    print('Testing client drop (P1 #27)...', flush=True)
    log_path = Path(tmp_dir) / 'client_drop.log'
    server = ServerInstance(binary, model, log_path, max_concurrent=4, workers=1, threads=2)
    server.start()
    try:
        body_bytes = json.dumps(golden_payload_slow).encode()

        # Baseline clean run
        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=30)
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        baseline_resp = conn.getresponse()
        baseline_ans = json.loads(baseline_resp.read())
        conn.close()
        assert baseline_resp.status == 200

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
        time.sleep(0.1)  # allow server to read request and begin processing
        s1.close()
        time.sleep(0.5)

        # Server must still be healthy and answer follow-up request normally
        hconn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
        hconn.request('GET', '/health')
        assert hconn.getresponse().status == 200
        hconn.close()

        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=30)
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        r = conn.getresponse()
        ans1 = json.loads(r.read())
        conn.close()
        assert r.status == 200
        assert_batch_answers_equal(ans1, baseline_ans)

        # Case B: Close socket mid-request WHILE response is being read
        print('  Case B: Dropping connection while reading response...', flush=True)
        s2 = socket.create_connection(('127.0.0.1', server.port), timeout=30)
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

        conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=30)
        conn.request('POST', '/v1/systemone/batch', body_bytes, {'Content-Type': 'application/json'})
        r = conn.getresponse()
        ans2 = json.loads(r.read())
        conn.close()
        assert r.status == 200
        assert_batch_answers_equal(ans2, baseline_ans)
        print('  Client drop tests passed cleanly.')
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


def run_sigterm_drain_case(binary, model, tmp_dir, golden_payload_slow, sig, run_label, measured_times):
    """Executes a single signal drain case: quick inferences -> 4 slow requests -> signal -> drain."""
    log_path = Path(tmp_dir) / f'drain_{run_label}.log'
    server = ServerInstance(binary, model, log_path, max_concurrent=8, workers=2, threads=4)
    server.start()
    try:
        # Run 200 quick inferences first (P1 #25)
        t_quick_start = time.monotonic()
        run_quick_inferences(server.port, count=200, workers=4)
        t_quick = time.monotonic() - t_quick_start

        # Start 4 slow requests
        body_bytes = json.dumps(golden_payload_slow).encode()

        def send_slow(idx):
            try:
                conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=60)
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
            server.proc.wait(timeout=SHUTDOWN_BOUND_SECONDS)
        except subprocess.TimeoutExpired:
            server.proc.kill()
            server.proc.wait()
            raise AssertionError(f'{run_label}: server hung after {sig}, exceeded bound {SHUTDOWN_BOUND_SECONDS}s')

        drain_duration = time.monotonic() - t0
        measured_times.append(drain_duration)
        assert drain_duration <= SHUTDOWN_BOUND_SECONDS, \
            f'{run_label}: shutdown took {drain_duration:.2f}s, exceeding bound {SHUTDOWN_BOUND_SECONDS}s'

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
        print(f'  {run_label} ({sig.name}): 200 quick inferences in {t_quick:.2f}s; drained in {drain_duration:.2f}s, exit 0')
    finally:
        if server.proc and server.proc.poll() is None:
            server.proc.kill()
            server.proc.wait()


def test_signals(binary, model, tmp_dir, golden_payload_slow):
    """P1 #26: SIGTERM/SIGINT drain: exits within bound, in-flight requests complete/close cleanly,
    new connections refused, shutdown event logged, repeated 5 times for SIGTERM + 1 SIGINT."""
    print('Testing signals and graceful drain (P1 #26)...', flush=True)
    measured_times = []

    # Test SIGINT once
    run_sigterm_drain_case(binary, model, tmp_dir, golden_payload_slow, signal.SIGINT, 'sigint-1', measured_times)

    # Repeat SIGTERM 5 times to catch races
    for i in range(1, 6):
        run_sigterm_drain_case(binary, model, tmp_dir, golden_payload_slow, signal.SIGTERM, f'sigterm-{i}', measured_times)

    print(f'  Shutdown drain times over 5 SIGTERM runs: '
          f'min={min(measured_times[1:]):.2f}s, max={max(measured_times[1:]):.2f}s, '
          f'avg={sum(measured_times[1:])/5:.2f}s (bound: {SHUTDOWN_BOUND_SECONDS}s)')


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
        test_signals(args.binary, args.model, tmp_dir, golden_payload_slow)

    print('server_lifecycle: ALL TESTS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
