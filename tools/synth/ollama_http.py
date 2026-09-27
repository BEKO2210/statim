"""Local Ollama chat client. Generation options always include num_gpu."""
import json
import time
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    pass


class OllamaHTTP:
    def __init__(self, host, model, num_gpu, keep_alive="30m", timeout=600):
        self.host = host.rstrip("/")
        self.model = model
        self.num_gpu = int(num_gpu)
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.cpu_snapshot = None

    def _post(self, path, body, timeout=None):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"})
        delay = 2.0
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last = OllamaError("HTTP %s %s" % (exc.code, detail))
                if exc.code < 500 and exc.code != 429:
                    raise last
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as exc:
                last = OllamaError(str(exc))
            if attempt < 3:
                time.sleep(delay)
                delay *= 2
        raise last

    def show(self):
        return self._post("/api/show", {"model": self.model}, timeout=60)

    def ps(self):
        req = urllib.request.Request(self.host + "/api/ps")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def assert_cpu(self):
        """Abort if this model was loaded onto the GPU. num_gpu 0 must stay on CPU."""
        if self.num_gpu != 0:
            return
        try:
            payload = self.ps()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError("cpu check failed: %s" % exc)
        for row in payload.get("models") or []:
            name = row.get("model") or row.get("name") or ""
            if name != self.model:
                continue
            vram = int(row.get("size_vram") or 0)
            self.cpu_snapshot = {
                "model": name, "size_vram": vram,
                "size": row.get("size"), "context_length": row.get("context_length")}
            if vram > 0:
                raise SystemExit(
                    "refusing to continue: %s loaded with size_vram=%s (num_gpu=%s)"
                    % (name, vram, self.num_gpu))
            return
        # Not listed: the model is unloaded, which is not evidence of GPU use.

    def chat(self, messages, schema, temperature, num_predict, seed=None):
        options = {
            "temperature": temperature,
            "num_gpu": self.num_gpu,
            "num_predict": int(num_predict),
            "num_ctx": 4096,
        }
        if seed is not None:
            options["seed"] = int(seed) % (2 ** 31 - 1)
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        data = self._post("/api/chat", body)
        self.assert_cpu()
        msg = data.get("message") or {}
        return {
            "content": msg.get("content") or "",
            "thinking": msg.get("thinking") or "",
            "done_reason": data.get("done_reason"),
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "total_s": (data.get("total_duration") or 0) / 1e9,
        }
