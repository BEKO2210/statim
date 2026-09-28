#!/usr/bin/env python3
"""CPU-only HTTP regression suite. Exit 77 when the environment forbids bind.
Run manually: python3 tests/security/test_http.py --binary build/statim
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', default='build/statim')
    parser.add_argument('--model', default='models/laya-multilingual-f32.gguf')
    args = parser.parse_args()
    env = dict(os.environ, STATIM_DEVICE='cpu')
    env.pop('STATIM_API_KEY', None)
    env.pop('STATIM_GPU_FAST', None)
    base = [args.binary, 'serve', '-m', 'multilingual=' + args.model, '--device', 'cpu', '--threads', '2']
    checks = 0

    def fail_start(extra=(), keys=None):
        nonlocal checks
        e = env.copy()
        if keys is not None:
            e['STATIM_API_KEY'] = keys
        p = subprocess.run(base + list(extra), env=e, capture_output=True, text=True, timeout=30)
        assert p.returncode != 0 and 'listening' not in p.stderr, p.stderr
        assert 'error:' in p.stderr, p.stderr
        checks += 1

    with tempfile.TemporaryDirectory(prefix='statim-security-') as tmp:
        fail_start(['--api-key-file', str(Path(tmp) / 'missing')])
        for contents in ('', '# comments only\n  # comment\r\n'):
            keyfile = Path(tmp) / 'keys'
            keyfile.write_text(contents)
            fail_start(['--api-key-file', str(keyfile)])
            fail_start(['--api-key-file', str(keyfile)], keys='otherwise-valid')
        for value in ('', ' , , ', 'invalid key'):
            fail_start(keys=value)
        for flag in ('--max-len', '--head-max-len', '--ensemble'):
            fail_start([flag, '4294967328'])
        fail_start(['--max-len', '32', '--head-max-len', '8192'])
        print(f'security startup: {checks} checks passed', flush=True)
        try:
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', 8094))
                except OSError as exc:
                    if exc.errno in (1, 13):
                        raise
                    probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
        except PermissionError:
            print('SKIP: sandbox cannot bind localhost; run tests/security/test_http.py manually. '
                  'C++ security tests exercise socket-free HTTP processing.', flush=True)
            return 77

        def request(path, payload=None, auth=True, rid=None):
            nonlocal checks
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
            headers = {'Content-Type': 'application/json'}
            if auth:
                headers['Authorization'] = 'Bearer secret'
            if rid:
                headers['X-Request-Id'] = rid
            if isinstance(payload, dict):
                payload = json.dumps(payload)
            conn.request('POST' if payload is not None else 'GET', path, payload, headers)
            res = conn.getresponse()
            status, body, headers = res.status, res.read(), dict(res.getheaders())
            conn.close()
            checks += 1
            if status >= 400:
                assert set(json.loads(body)) == {'detail'}, body
            return status, body, headers

        def raw(headers, status):
            nonlocal checks
            with socket.create_connection(('127.0.0.1', port), timeout=3) as conn:
                conn.sendall(('POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n' + headers + '\r\n\r\n').encode())
                data = conn.recv(8192)
                assert data.startswith(f'HTTP/1.1 {status}'.encode()), data
            checks += 1

        logfile = Path(tmp) / 'server.log'
        with logfile.open('w+') as log:
            proc = subprocess.Popen(base + ['--port', str(port), '--max-concurrent', '2', '--http-queue', '4',
                                           '--request-timeout', '2'],
                                    env=dict(env, STATIM_API_KEY='secret'), stdout=log, stderr=log)
            try:
                for _ in range(300):
                    if proc.poll() is not None:
                        raise AssertionError(logfile.read_text())
                    try:
                        if request('/health', auth=False)[0] == 200:
                            break
                    except (OSError, http.client.HTTPException):
                        time.sleep(.1)
                else:
                    raise AssertionError('server did not become healthy')
                status, body, _ = request('/health', auth=False)
                assert status == 200 and set(json.loads(body)) == {'status', 'version'}
                assert request('/ready', auth=False)[0] == 200
                for path in ('/metrics', '/v1/models'):
                    assert request(path, auth=False)[0] == 401
                    assert request(path)[0] == 200
                q = {'x': {'type': 'choice', 'instructions': 'Choose', 'criteria': ['yes', 'no']}}
                req = {'state': 'hello', 'questions': q}
                assert request('/v1/systemone', req, auth=False)[0] == 401
                status, body, _ = request('/v1/systemone/batch', {'states': ['one', 'two'], 'questions': {}})
                batch = json.loads(body)
                assert status == 200 and set(batch) == {'results'} and len(batch['results']) == 2
                assert all(set(result) == {'model', 'answers', 'usage', 'routing'} for result in batch['results'])
                deep = '{"state":' + '[' * 30000 + '0' + ']' * 30000 + ',"questions":{}}'
                assert request('/v1/systemone', deep)[0] == 413
                wide = {'state': {f'k{i}': 0 for i in range(30000)}, 'questions': {}}
                assert request('/v1/systemone', wide)[0] == 413
                for value in (4294967328, -4294967264, 18446744073709551615):
                    for flag in ('max_len', 'head_max_len', 'ensemble'):
                        assert request('/v1/systemone', dict(req, **{flag: value}))[0] == 422
                assert request('/v1/systemone', dict(req, max_len=32, head_max_len=8192))[0] == 422
                enormous = {'states': ['hello'] * 256, 'questions': {str(i): q['x'] for i in range(64)}}
                assert request('/v1/systemone/batch', enormous)[0] == 413
                def rss_kib():
                    with open('/proc/%d/status' % proc.pid) as f:
                        return int(next(l for l in f if l.startswith('VmRSS:')).split()[1])
                # Unknown fields are ignored (laya.serve does too) but must never be cached. The allocator
                # keeps some freed memory per HTTP thread, so RSS rises at first and then flattens; a cache
                # holding the ignored bytes would instead grow by at least the payload (64 MiB below).
                def send_extra(tag):
                    extra = {'x': dict(q['x'], ignored=tag + 'x' * 1048576)}
                    assert request('/v1/systemone', dict(req, questions=extra, calibrate=True))[0] == 200
                for i in range(32):
                    send_extra('w%d' % i)
                before = rss_kib()
                for i in range(64):
                    send_extra(str(i))
                grown = rss_kib() - before
                print('rss growth over 64 x 1 MiB ignored fields after warm-up: %d KiB' % grown, flush=True)
                assert grown < 32 * 1024, 'memory grows with ignored fields (%d KiB for 64 MiB sent)' % grown
                # selective prediction: no threshold -> response unchanged; threshold -> escalate flag per answer
                status, body_ok, _ = request('/v1/systemone', req)
                assert status == 200 and all('escalate' not in a for a in json.loads(body_ok)['answers'].values())
                status, body_hi, _ = request('/v1/systemone', dict(req, min_confidence=0.999999))
                assert status == 200 and all(a['escalate'] is (a['answer_confidence'] < 0.999999) for a in json.loads(body_hi)['answers'].values())
                assert request('/v1/systemone', dict(req, min_confidence=1.5))[0] == 422
                assert request('/v1/systemone', dict(req, min_confidence='high'))[0] == 422
                for rid in ('audit%0D%0Aforged-log-line', 'audit", "forged":true', 'x' * 129):
                    status, _, headers = request('/v1/systemone', {'state': 'ok', 'questions': {}}, rid=rid)
                    assert status == 200 and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', headers['X-Request-Id'])
                raw('Content-Length: 100', 401)  # no body sent: reject immediately
                raw('Authorization: Bearer secret\r\nContent-Length: 999999999', 413)
                for framing in ('Content-Length: 100\r\nTransfer-Encoding: chunked',
                                'Content-Length: 0\r\nTransfer-Encoding: chunked',
                                'Content-Length: 2\r\nContent-Length: 2',
                                'Content-Length: 2\r\nContent-Length: 3', 'Content-Length: 2, 2', 'Content-Length: -1'):
                    raw('Authorization: Bearer secret\r\n' + framing, 400)
                # A partial authenticated upload consumes one admission slot.
                slow = []
                try:
                    for _ in range(2):
                        conn = socket.create_connection(('127.0.0.1', port), timeout=3)
                        conn.sendall(b'POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Length: 100\r\n\r\n{')
                        slow.append(conn)
                    time.sleep(.1)
                    raw('Authorization: Bearer secret\r\nContent-Length: 100', 503)
                    assert request('/health', auth=False)[0] == 200
                    # Drip bytes faster than the idle timeout; absolute timeout must still expire.
                    start = time.monotonic()
                    while time.monotonic() - start < 2.5:
                        for conn in slow:
                            try:
                                conn.sendall(b' ')
                            except OSError:
                                pass
                        time.sleep(.1)
                    for conn in slow:
                        data = conn.recv(8192)
                        assert not data or data.startswith(b'HTTP/1.1 400'), data
                    checks += 1
                finally:
                    for conn in slow:
                        conn.close()
                # Headers have the same absolute deadline; dripping bytes cannot hold a worker.
                with socket.create_connection(('127.0.0.1', port), timeout=3) as conn:
                    conn.sendall(b'GET /health HTTP/1.1\r\nHost: localhost\r\nX-Slow: ')
                    start = time.monotonic()
                    while time.monotonic() - start < 2.5:
                        try:
                            conn.sendall(b'x')
                        except OSError:
                            break
                        time.sleep(.1)
                    data = conn.recv(8192)
                    assert not data or data.startswith(b'HTTP/1.1 400'), data
                    checks += 1
                # Calibration hits must return identical decision contents.
                first = json.loads(request('/v1/systemone', dict(req, calibrate=True))[1])
                second = json.loads(request('/v1/systemone', dict(req, calibrate=True))[1])
                assert first == second and 'answers' in first
                assert request('/health', auth=False)[0] == 200
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for line in logfile.read_text().splitlines():
            if line.startswith('{'):
                record = json.loads(line)
                assert 'forged' not in record
                if record.get('event') == 'request':
                    assert re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', record['request_id'])
        # No source configured is the explicit local auth-off default.
        with logfile.open('w+') as log:
            proc = subprocess.Popen(base + ['--port', str(port)], env=env, stdout=log, stderr=log)
            try:
                for _ in range(300):
                    if proc.poll() is not None:
                        raise AssertionError(logfile.read_text())
                    try:
                        if request('/metrics', auth=False)[0] == 200:
                            break
                    except (OSError, http.client.HTTPException):
                        time.sleep(.1)
                else:
                    raise AssertionError('auth-off server did not start')
            finally:
                proc.terminate()
                proc.wait(timeout=15)
        records = [json.loads(line) for line in logfile.read_text().splitlines() if line.startswith('{')]
        assert any(r.get('event') == 'listening' and r.get('auth') is False and r.get('auth_status') == 'off' for r in records)
        # A path from the command line is arbitrary bytes. model_loaded must not throw.
        tiny = Path(__file__).resolve().parents[2] / 'fuzz' / 'data' / 'tiny-metaspace.gguf'
        assert tiny.is_file(), tiny
        link = os.path.join(os.fsencode(tmp), b'm-\xff.gguf')
        os.symlink(os.fsencode(tiny), link)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        badlog = Path(tmp) / 'bad-path.log'
        cmd = [os.fsencode(args.binary), b'serve', b'-m', b'tiny=' + link, b'--device', b'cpu',
               b'--threads', b'2', b'--port', str(port).encode()]
        with badlog.open('w+b') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise AssertionError(badlog.read_bytes())
                    try:
                        if request('/health', auth=False)[0] == 200:
                            break
                    except (OSError, http.client.HTTPException):
                        time.sleep(.05)
                else:
                    raise AssertionError(b'server did not become healthy:\n' + badlog.read_bytes())
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        text = badlog.read_bytes().decode('utf-8')
        loaded = [json.loads(line) for line in text.splitlines() if line.startswith('{') and '"model_loaded"' in line]
        assert loaded and loaded[0]['event'] == 'model_loaded' and '\ufffd' in loaded[0]['path'], text
        print('invalid-utf8 model path: health 200', flush=True)
        print(f'security HTTP: {checks} checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
