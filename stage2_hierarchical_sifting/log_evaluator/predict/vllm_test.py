# Thin OpenAI-compatible client wrapper used to query a locally served
# Log Evaluator model (e.g. served with `vllm serve <merge_model> --port 8080`).

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


def create_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAI:
    base_url = base_url or os.getenv("VLLM_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = api_key or os.getenv("VLLM_OPENAI_API_KEY", "")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_once(
    prompt: str,
    model: str,
    *,
    client: Optional[OpenAI] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_create_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = client or create_client(base_url=base_url, api_key=api_key)
    kwargs = dict(extra_create_kwargs or {})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    content = None
    if resp.choices and resp.choices[0].message is not None:
        content = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    return {
        "content": content,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
        "raw": resp,
    }


def chat_parallel(
    prompts: Iterable[str],
    model: str,
    *,
    max_workers: int = 8,
    client: Optional[OpenAI] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_create_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    prompts_list = list(prompts)
    client = client or create_client(base_url=base_url, api_key=api_key)
    extra_create_kwargs = dict(extra_create_kwargs or {})

    def _one(p: str) -> Dict[str, Any]:
        return chat_once(
            p,
            model,
            client=client,
            extra_create_kwargs=extra_create_kwargs,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, prompts_list))


if __name__ == "__main__":
    _model = os.getenv(
        "VLLM_OPENAI_MODEL",
        "/path/to/log_evaluator/merge_model",
    )
    _prompt = "Hello, what's deepinfra?"
    r = chat_once(_prompt, _model)
    print(r["content"])
    print(r["prompt_tokens"], r["completion_tokens"])
