#!/usr/bin/env python3
"""CPU live-server parity test for opt-in single-request micro-batching."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time


def reserve_port():
    probe = socket.socket()
    try:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


def request(port, payload, request_id):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
    conn.request('POST', '/v1/systemone', json.dumps(payload),
                 {'Content-Type': 'application/json', 'X-Request-Id': request_id})
    res = conn.getresponse()
    body, headers = res.read(), dict(res.getheaders())
    conn.close()
    assert res.status == 200, (res.status, body)
    assert headers['X-Request-Id'] == request_id
    assert float(headers['X-Inference-Time-Ms']) >= 0
    return json.loads(body)


def run(binary, model, payloads, batched, tmp):
    port = reserve_port()
    log_path = Path(tmp) / ('batch.log' if batched else 'plain.log')
    command = [binary, 'serve', '-m', 'multilingual=' + model, '--device', 'cpu', '--threads', '2',
               '--port', str(port), '--max-concurrent', '16', '--no-access-log']
    if batched:
        command += ['--batch-window-ms', '20', '--max-batch', '16']
    env = dict(os.environ, STATIM_DEVICE='cpu')
    env.pop('STATIM_API_KEY', None)
    env.pop('STATIM_GPU_FAST', None)
    with log_path.open('w+') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            for _ in range(300):
                if proc.poll() is not None:
                    raise AssertionError(log_path.read_text())
                try:
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                    conn.request('GET', '/health')
                    ready = conn.getresponse()
                    ready.read()
                    conn.close()
                    if ready.status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    time.sleep(.05)
            else:
                raise AssertionError('server did not become healthy')
            # Warm weights and keep this request out of the measured concurrent wave.
            request(port, payloads[0], 'warmup')
            with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
                futures = [pool.submit(request, port, payload, 'parity-%d' % i)
                           for i, payload in enumerate(payloads)]
                results = [future.result() for future in futures]
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            conn.request('GET', '/metrics')
            metrics_res = conn.getresponse()
            metrics = metrics_res.read().decode()
            conn.close()
            assert metrics_res.status == 200
            if batched:
                values = {line.split()[0]: float(line.split()[1]) for line in metrics.splitlines()
                          if line and not line.startswith('#') and len(line.split()) == 2}
                assert values['statim_batch_size_count'] >= 2
                assert values['statim_batch_size_sum'] > values['statim_batch_size_count']
                assert values['statim_batch_wait_ms_count'] >= len(payloads)
            return results
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', required=True)
    args = parser.parse_args()
    try:
        port = reserve_port()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    except PermissionError:
        print('SKIP: sandbox cannot bind localhost')
        return 77

    fixture = json.load(open(args.inputs))
    states = [state for state in fixture['states'] if state != ''][:6]
    qids = list(fixture['questions'])[:3]
    questions = {qid: fixture['questions'][qid] for qid in qids}
    payloads = [{'state': state, 'questions': questions} for state in states]
    incompatible = json.loads(json.dumps(questions))
    incompatible[qids[0]]['instructions'] += ' Return the closest option.'
    payloads.insert(2, {'state': states[-1], 'questions': incompatible})

    with tempfile.TemporaryDirectory(prefix='statim-microbatch-') as tmp:
        plain = run(args.binary, args.model, payloads, False, tmp)
        packed = run(args.binary, args.model, payloads, True, tmp)
    for left, right in zip(plain, packed):
        assert_answers_equal(left, right)
    print('server micro-batch: %d concurrent requests match within 1e-4' % len(payloads))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
