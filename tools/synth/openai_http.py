"""Chat client for an OpenAI-compatible server (vLLM), with the interface of ollama_http.OllamaHTTP.

The server runs ONE Hugging Face model; provenance is that repo id and the exact revision the
caller pinned (vLLM does not report it, so the caller passes the sha it downloaded).
"""
import json
import time
import urllib.error
import urllib.request

from ollama_http import OllamaError


class OpenAIHTTP:
    def __init__(self, host, model, revision, timeout=600):
        self.host = host.rstrip("/")
        self.model = model
        self.revision = revision
        self.timeout = timeout

    def _post(self, path, body, timeout=None):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"})
        delay, last = 2.0, None
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

    def _get(self, path):
        with urllib.request.urlopen(self.host + path, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def show(self):
        """Provenance of the served model: the repo id must be the one this client was built for."""
        served = [row.get("id") for row in (self._get("/v1/models").get("data") or [])]
        if self.model not in served:
            raise OllamaError("server serves %s, not %s" % (served, self.model))
        return {"hf_id": self.model, "revision": self.revision, "backend": "vllm-openai",
                "served_models": served}

    def chat(self, messages, schema, temperature, num_predict, seed=None):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(num_predict),
            # vLLM structured output: the reply must match the schema
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "item", "schema": schema}},
            # Qwen3 templates: answer directly, no thinking block (ignored by other templates)
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed is not None:
            body["seed"] = int(seed) % (2 ** 31 - 1)
        started = time.monotonic()
        data = self._post("/v1/chat/completions", body)
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return {
            "content": (choice.get("message") or {}).get("content") or "",
            "thinking": "",
            "done_reason": choice.get("finish_reason"),
            "eval_count": usage.get("completion_tokens"),
            "prompt_eval_count": usage.get("prompt_tokens"),
            "total_s": time.monotonic() - started,
        }
