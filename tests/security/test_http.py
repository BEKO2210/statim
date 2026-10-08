#!/usr/bin/env python3
"""CPU-only HTTP regression suite. Exit 77 when the environment forbids bind.
Run manually: python3 tests/security/test_http.py --binary build/statim
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

KEY = 'statim-test-key-0123456789abcdef'


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
        def test_non_loopback_without_key_fails_closed():
            nonlocal checks
            missing_model = str(Path(tmp) / 'must-not-be-loaded.gguf')
            started = time.monotonic()
            p = subprocess.run([args.binary, 'serve', '-m', missing_model, '--host', '0.0.0.0'],
                               env=env, capture_output=True, text=True, timeout=5)
            elapsed = time.monotonic() - started
            assert p.returncode == 2, (p.returncode, p.stderr)
            assert 'host "0.0.0.0" is not loopback' in p.stderr, p.stderr
            assert '--allow-unauthenticated' in p.stderr, p.stderr
            assert 'model_loaded' not in p.stderr and missing_model not in p.stderr, p.stderr
            assert elapsed < 5, elapsed
            checks += 1

        test_non_loopback_without_key_fails_closed()
        def test_short_keys_fail_before_model_load():
            nonlocal checks
            missing_model = str(Path(tmp) / 'short-key-must-not-load.gguf')
            keyfile = Path(tmp) / 'short-keys'
            keyfile.write_text('# first line\n' + 'f' * 31 + '\n', encoding='ascii')
            cases = [([args.binary, 'serve', '-m', missing_model], dict(env, STATIM_API_KEY='e' * 31),
                      'STATIM_API_KEY'),
                     ([args.binary, 'serve', '-m', missing_model, '--api-key-file', str(keyfile)], env,
                      str(keyfile) + ':2')]
            for command, case_env, source in cases:
                p = subprocess.run(command, env=case_env, capture_output=True, text=True, timeout=5)
                assert p.returncode == 2, (p.returncode, p.stderr)
                assert source in p.stderr and 'length 31' in p.stderr and 'minimum is 32' in p.stderr, p.stderr
                assert 'openssl rand -hex 32' in p.stderr, p.stderr
                assert 'model_loaded' not in p.stderr and missing_model not in p.stderr, p.stderr
                checks += 1

        test_short_keys_fail_before_model_load()
        def test_bad_scope_fails_before_model_load():
            nonlocal checks
            missing_model = str(Path(tmp) / 'bad-scope-must-not-load.gguf')
            keyfile = Path(tmp) / 'bad-scope-keys'
            keyfile.write_text('s' * 32 + ' unknown\n', encoding='ascii')
            p = subprocess.run([args.binary, 'serve', '-m', missing_model, '--api-key-file', str(keyfile)],
                               env=env, capture_output=True, text=True, timeout=5)
            assert p.returncode == 2, (p.returncode, p.stderr)
            assert str(keyfile) + ':1' in p.stderr and 'unknown API key scope' in p.stderr, p.stderr
            assert 's' * 32 not in p.stderr, p.stderr
            assert 'model_loaded' not in p.stderr and missing_model not in p.stderr, p.stderr
            checks += 1

        test_bad_scope_fails_before_model_load()
        def test_invalid_frame_ancestors_fails_before_model_load():
            nonlocal checks
            missing_model = str(Path(tmp) / 'frame-ancestors-must-not-load.gguf')
            p = subprocess.run([args.binary, 'serve', '-m', missing_model,
                                '--frame-ancestors', 'https://a.example/path'],
                               env=env, capture_output=True, text=True, timeout=5)
            assert p.returncode == 2, (p.returncode, p.stderr)
            assert 'invalid --frame-ancestors origin' in p.stderr, p.stderr
            assert 'model_loaded' not in p.stderr and missing_model not in p.stderr, p.stderr
            checks += 1

        test_invalid_frame_ancestors_fails_before_model_load()
        def test_invalid_cors_origins_fail_before_model_load():
            nonlocal checks
            missing_model = str(Path(tmp) / 'cors-origin-must-not-load.gguf')
            cases = [
                ('https://a.example/path', 'invalid --cors-origin origin'),
                (' '.join('https://%d.example' % i for i in range(9)),
                 '--cors-origin accepts at most 8 origins'),
            ]
            for value, message in cases:
                p = subprocess.run([args.binary, 'serve', '-m', missing_model, '--cors-origin', value],
                                   env=env, capture_output=True, text=True, timeout=5)
                assert p.returncode == 2, (p.returncode, p.stderr)
                assert message in p.stderr and '--frame-ancestors' not in p.stderr, p.stderr
                assert 'model_loaded' not in p.stderr and missing_model not in p.stderr, p.stderr
                checks += 1

        test_invalid_cors_origins_fail_before_model_load()
        fail_start(['--api-key-file', str(Path(tmp) / 'missing')])
        for contents in ('', '# comments only\n  # comment\r\n'):
            keyfile = Path(tmp) / 'keys'
            keyfile.write_text(contents)
            fail_start(['--api-key-file', str(keyfile)])
            fail_start(['--api-key-file', str(keyfile)], keys=KEY)
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

        def reserve_port(host='127.0.0.1', family=socket.AF_INET):
            with socket.socket(family) as probe:
                probe.bind((host, 0))
                return probe.getsockname()[1]

        def wait_for_health(proc, logfile, connect_host, server_port):
            for _ in range(300):
                if proc.poll() is not None:
                    raise AssertionError(logfile.read_text())
                try:
                    conn = http.client.HTTPConnection(connect_host, server_port, timeout=1)
                    conn.request('GET', '/health')
                    response = conn.getresponse()
                    body = response.read()
                    conn.close()
                    if response.status == 200:
                        assert set(json.loads(body)) == {'status', 'version'}
                        return
                except (OSError, http.client.HTTPException):
                    time.sleep(.1)
            raise AssertionError('server did not become healthy:\n' + logfile.read_text())

        def test_allowed_frame_ancestor():
            nonlocal checks
            server_port = reserve_port()
            logfile = Path(tmp) / 'frame-ancestor-server.log'
            command = base + ['--port', str(server_port), '--frame-ancestors', 'https://huggingface.co']
            with logfile.open('w+') as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
                try:
                    wait_for_health(proc, logfile, '127.0.0.1', server_port)
                    conn = http.client.HTTPConnection('127.0.0.1', server_port, timeout=3)
                    conn.request('GET', '/')
                    response = conn.getresponse()
                    response.read()
                    headers = dict(response.getheaders())
                    conn.close()
                    assert response.status == 200, response.status
                    assert 'X-Frame-Options' not in headers, headers
                    csp = headers.get('Content-Security-Policy', '')
                    assert 'frame-ancestors https://huggingface.co' in csp, csp
                    checks += 1
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

        test_allowed_frame_ancestor()

        def test_cors():
            nonlocal checks
            allowed_origin = 'https://beko2210.github.io'
            server_port = reserve_port()
            logfile = Path(tmp) / 'cors-server.log'
            command = base + ['--port', str(server_port), '--cors-origin',
                              allowed_origin + ', ' + allowed_origin]

            def call(method, path, origin, body=None, auth=True, extra=None):
                conn = http.client.HTTPConnection('127.0.0.1', server_port, timeout=30)
                headers = {'Origin': origin}
                if auth:
                    headers['Authorization'] = 'Bearer ' + KEY
                if body is not None:
                    headers['Content-Type'] = 'application/json'
                if extra:
                    headers.update(extra)
                conn.request(method, path, body=body, headers=headers)
                response = conn.getresponse()
                result = response.status, response.read(), dict(response.getheaders())
                conn.close()
                return result

            with logfile.open('w+') as log:
                proc = subprocess.Popen(command, env=dict(env, STATIM_API_KEY=KEY), stdout=log, stderr=log)
                try:
                    wait_for_health(proc, logfile, '127.0.0.1', server_port)
                    payload = json.dumps({'state': 'hello', 'questions': {}})
                    status, _, allowed = call('POST', '/v1/systemone', allowed_origin, payload)
                    assert status == 200, status
                    assert allowed['Access-Control-Allow-Origin'] == allowed_origin
                    assert allowed['Vary'] == 'Origin'
                    assert allowed['Access-Control-Expose-Headers'] == 'X-Request-Id'
                    assert 'Access-Control-Allow-Credentials' not in allowed

                    status, body, error = call('POST', '/v1/systemone', allowed_origin, '{')
                    assert status == 400 and set(json.loads(body)) == {'detail'}, (status, body)
                    assert error['Access-Control-Allow-Origin'] == allowed_origin
                    assert error['Access-Control-Expose-Headers'] == 'X-Request-Id'

                    status, body, preflight = call(
                        'OPTIONS', '/v1/systemone', allowed_origin, auth=False,
                        extra={'Access-Control-Request-Method': 'POST'})
                    assert status == 204 and body == b'', (status, body)
                    assert preflight['Access-Control-Allow-Origin'] == allowed_origin
                    assert preflight['Vary'] == 'Origin'
                    assert preflight['Access-Control-Allow-Methods'] == 'GET, POST, OPTIONS'
                    assert preflight['Access-Control-Allow-Headers'] == 'Content-Type, Authorization'
                    assert preflight['Access-Control-Max-Age'] == '600'
                    assert 'Access-Control-Allow-Credentials' not in preflight

                    status, _, other = call('POST', '/v1/systemone', 'https://other.example', payload)
                    assert status == 200 and not any(k.lower().startswith('access-control-') for k in other), other
                    assert other.get('Vary') == 'Origin'  # the allowlist makes every response depend on Origin
                    print('CORS allowed POST: ' + repr({k: allowed[k] for k in allowed if k.startswith('Access-Control-') or k == 'Vary'}), flush=True)
                    print('CORS allowed preflight: ' + repr({k: preflight[k] for k in preflight if k.startswith('Access-Control-') or k == 'Vary'}), flush=True)
                    print('CORS other origin: {}', flush=True)
                    checks += 4
                finally:
                    proc.terminate()
                    proc.wait(timeout=15)

        test_cors()

        def run_startup_server(name, host, connect_host, keys=None, allow=False, family=socket.AF_INET):
            nonlocal checks
            server_port = reserve_port(connect_host, family)
            logfile = Path(tmp) / (name + '.log')
            command = base + ['--host', host, '--port', str(server_port)]
            if allow:
                command.append('--allow-unauthenticated')
            server_env = env if keys is None else dict(env, STATIM_API_KEY=keys)
            with logfile.open('w+') as log:
                proc = subprocess.Popen(command, env=server_env, stdout=log, stderr=log)
                try:
                    wait_for_health(proc, logfile, connect_host, server_port)
                    conn = http.client.HTTPConnection(connect_host, server_port, timeout=3)
                    conn.request('GET', '/metrics')
                    anonymous = conn.getresponse()
                    anonymous.read()
                    conn.close()
                    if keys is None:
                        assert anonymous.status == 200, anonymous.status
                    else:
                        assert anonymous.status == 401, anonymous.status
                        conn = http.client.HTTPConnection(connect_host, server_port, timeout=3)
                        conn.request('GET', '/metrics', headers={'Authorization': 'Bearer ' + keys})
                        authenticated = conn.getresponse()
                        authenticated.read()
                        conn.close()
                        assert authenticated.status == 200, authenticated.status
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            checks += 1
            return logfile.read_text()

        def test_non_loopback_allow_unauthenticated():
            log = run_startup_server('non-loopback-allow', '0.0.0.0', '127.0.0.1', allow=True)
            records = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
            assert any(record.get('event') == 'auth_off_on_network' for record in records), log

        def test_non_loopback_env_key():
            run_startup_server('non-loopback-key', '0.0.0.0', '127.0.0.1', keys=KEY)

        def test_loopback_ipv4_without_key():
            run_startup_server('loopback-ipv4', '127.0.0.1', '127.0.0.1')

        def test_loopback_ipv6_without_key():
            try:
                reserve_port('::1', socket.AF_INET6)
            except OSError as exc:
                print('SKIP: test_loopback_ipv6_without_key: IPv6 loopback unavailable: %s' % exc,
                      flush=True)
                return
            run_startup_server('loopback-ipv6', '::1', '::1', family=socket.AF_INET6)

        test_non_loopback_allow_unauthenticated()
        test_non_loopback_env_key()
        test_loopback_ipv4_without_key()
        test_loopback_ipv6_without_key()

        def test_scoped_keys():
            nonlocal checks
            metrics_key, inference_key, all_key = 'm' * 32, 'i' * 32, 'a' * 32
            keyfile = Path(tmp) / 'scoped-keys'
            keyfile.write_text(f'{metrics_key} metrics\n{inference_key} inference\n{all_key}\n', encoding='ascii')
            scoped_port = reserve_port()
            scoped_log = Path(tmp) / 'scoped-server.log'
            command = base + ['--port', str(scoped_port), '--api-key-file', str(keyfile)]

            def scoped_request(path, key, payload=None):
                conn = http.client.HTTPConnection('127.0.0.1', scoped_port, timeout=30)
                headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
                body = json.dumps(payload) if payload is not None else None
                conn.request('POST' if payload is not None else 'GET', path, body, headers)
                response = conn.getresponse()
                result = response.status, response.read()
                conn.close()
                return result

            with scoped_log.open('w+') as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
                try:
                    wait_for_health(proc, scoped_log, '127.0.0.1', scoped_port)
                    empty_decision = {'state': 'hello', 'questions': {}}
                    assert scoped_request('/metrics', metrics_key)[0] == 200
                    status, body = scoped_request('/v1/systemone', metrics_key, empty_decision)
                    assert status == 403 and json.loads(body) == {'detail': "API key lacks the 'inference' scope"}
                    status, body = scoped_request('/v1/models', metrics_key)
                    assert status == 403 and json.loads(body) == {'detail': "API key lacks the 'inference' scope"}
                    assert scoped_request('/v1/systemone', inference_key, empty_decision)[0] == 200
                    status, body = scoped_request('/metrics', inference_key)
                    assert status == 403 and json.loads(body) == {'detail': "API key lacks the 'metrics' scope"}
                    assert scoped_request('/metrics', all_key)[0] == 200
                    assert scoped_request('/v1/systemone', all_key, empty_decision)[0] == 200
                    assert scoped_request('/metrics', 'z' * 32)[0] == 401
                    checks += 8
                finally:
                    proc.terminate()
                    proc.wait(timeout=15)
            log_text = scoped_log.read_text()
            for key in (metrics_key, inference_key, all_key):
                assert key not in log_text, log_text
            records = [json.loads(line) for line in log_text.splitlines() if line.startswith('{')]
            listening = next(record for record in records if record.get('event') == 'listening')
            assert listening['keys'] == 3
            expected = [
                {'id': hashlib.sha256(metrics_key.encode()).hexdigest()[:8], 'scopes': ['metrics']},
                {'id': hashlib.sha256(inference_key.encode()).hexdigest()[:8], 'scopes': ['inference']},
                {'id': hashlib.sha256(all_key.encode()).hexdigest()[:8], 'scopes': ['inference', 'metrics']},
            ]
            assert listening['key_ids'] == expected, listening
            request_records = [record for record in records if record.get('event') == 'request']
            assert {record.get('key_id') for record in request_records} >= {expected[1]['id'], expected[2]['id']}, request_records

        test_scoped_keys()
        port = reserve_port()

        def request(path, payload=None, auth=True, rid=None):
            nonlocal checks
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
            headers = {'Content-Type': 'application/json'}
            if auth:
                headers['Authorization'] = 'Bearer ' + KEY
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
            proc = subprocess.Popen(base + ['--port', str(port), '--workers', '2', '--max-concurrent', '2', '--http-queue', '4',
                                           '--request-timeout', '2'],
                                    env=dict(env, STATIM_API_KEY=KEY), stdout=log, stderr=log)
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
                status, body, hdr = request('/health', auth=False)
                assert status == 200 and set(json.loads(body)) == {'status', 'version'}
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                conn.request('OPTIONS', '/v1/systemone', headers={
                    'Origin': 'https://beko2210.github.io',
                    'Access-Control-Request-Method': 'POST',
                })
                unset_options = conn.getresponse()
                unset_body = unset_options.read()
                unset_headers = dict(unset_options.getheaders())
                conn.close()
                assert unset_options.status == 401 and json.loads(unset_body) == {
                    'detail': 'invalid or missing bearer token'
                }, (unset_options.status, unset_body)
                assert not any(k.lower().startswith('access-control-') for k in unset_headers), unset_headers
                assert 'Vary' not in unset_headers
                print('CORS flag unset OPTIONS: {}', flush=True)
                checks += 1
                # security headers on every response, also on errors (READINESS P1 #14)
                assert hdr.get('X-Content-Type-Options') == 'nosniff' and hdr.get('Referrer-Policy') == 'no-referrer', hdr
                _, _, hdr = request('/metrics', auth=False)
                assert hdr.get('X-Content-Type-Options') == 'nosniff', hdr
                status, page, hdr = request('/', auth=False)
                csp = hdr.get('Content-Security-Policy', '')
                assert status == 200 and hdr.get('X-Frame-Options') == 'DENY', hdr
                script_src = csp.split('script-src', 1)[1].split(';', 1)[0]
                assert "'sha256-" in script_src and 'unsafe-inline' not in script_src, csp
                assert "frame-ancestors 'none'" in csp and "default-src 'none'" in csp, csp
                assert b'localStorage.getItem("statim_key")' not in page and b'sessionStorage' in page
                assert request('/ready', auth=False)[0] == 200
                for path in ('/metrics', '/v1/models'):
                    assert request(path, auth=False)[0] == 401
                    assert request(path)[0] == 200
                _, models_body, _ = request('/v1/models')
                listed_model = json.loads(models_body)['data'][0]
                fingerprint = listed_model['fingerprint']
                checkpoint_sha256 = listed_model['checkpoint_sha256']
                assert re.fullmatch(r'[0-9a-f]{64}', fingerprint), fingerprint
                assert checkpoint_sha256 is None or re.fullmatch(r'[0-9a-f]{64}', checkpoint_sha256), checkpoint_sha256
                _, metrics_body, _ = request('/metrics')
                metrics_lines = metrics_body.decode().splitlines()
                assert 'statim_max_concurrent 2' in metrics_lines, metrics_body
                assert 'statim_workers{model="multilingual"} 2' in metrics_lines, metrics_body
                model_info = next(line for line in metrics_lines
                                  if line.startswith('statim_model_info{model="multilingual",'))
                assert 'fingerprint="%s"' % fingerprint in model_info, model_info
                q = {'x': {'type': 'choice', 'instructions': 'Choose', 'criteria': ['yes', 'no']}}
                req = {'state': 'hello', 'questions': q}
                assert request('/v1/systemone', req, auth=False)[0] == 401
                yes_no_q = {'answer': {'type': 'yes_no', 'instructions': 'Is this a greeting?',
                                       'labels': {'false': 'no', 'true': 'yes'}}}
                noul_q = {'answer': dict(yes_no_q['answer'], type='noul')}
                yes_no_status, yes_no_body, _ = request('/v1/systemone', {'state': 'hello', 'questions': yes_no_q})
                noul_status, noul_body, _ = request('/v1/systemone', {'state': 'hello', 'questions': noul_q})
                yes_no_answer = json.loads(yes_no_body)['answers']['answer']
                noul_answer = json.loads(noul_body)['answers']['answer']
                assert yes_no_status == noul_status == 200
                assert yes_no_answer['type'] == 'noul'
                assert yes_no_answer['noul'] == noul_answer['noul']
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
                raw('Authorization: Bearer ' + KEY + '\r\nContent-Length: 999999999', 413)
                for framing in ('Content-Length: 100\r\nTransfer-Encoding: chunked',
                                'Content-Length: 0\r\nTransfer-Encoding: chunked',
                                'Content-Length: 2\r\nContent-Length: 2',
                                'Content-Length: 2\r\nContent-Length: 3', 'Content-Length: 2, 2', 'Content-Length: -1'):
                    raw('Authorization: Bearer ' + KEY + '\r\n' + framing, 400)
                # A partial authenticated upload consumes one admission slot.
                slow = []
                try:
                    for _ in range(2):
                        conn = socket.create_connection(('127.0.0.1', port), timeout=3)
                        conn.sendall(('POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer ' + KEY +
                                      '\r\nContent-Length: 100\r\n\r\n{').encode())
                        slow.append(conn)
                    time.sleep(.1)
                    raw('Authorization: Bearer ' + KEY + '\r\nContent-Length: 100', 503)
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
        assert loaded[0]['gemm'] in ('packed_sgemm', 'ggml'), loaded[0]
        assert isinstance(loaded[0]['cpu_features'], list), loaded[0]
        assert set(loaded[0]['cpu_features']) <= {'avx2', 'fma', 'f16c', 'avx512f'}, loaded[0]
        print('invalid-utf8 model path: health 200', flush=True)
        print(f'security HTTP: {checks} checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
