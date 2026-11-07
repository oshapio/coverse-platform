"""Ask vision-language models questions through any OpenAI-compatible chat API.

One client covers every backend that speaks the OpenAI chat-completions protocol:
a vLLM server (`vllm serve`), a LiteLLM proxy, OpenRouter, OpenAI.
Each model is one entry in a YAML file (models.yaml):

    qwen3-vl-8b:
      base_url: http://localhost:8000/v1
      model_id: qwen
    claude-sonnet-5:
      base_url: https://openrouter.ai/api/v1
      model_id: anthropic/claude-sonnet-5
      api_key_env: OPENROUTER_API_KEY
      image_first: true
      params: {max_tokens: 4096}

`generate` sends many requests at once (`concurrency` in flight) and appends each answer to a
JSONL file as soon as it arrives, so a stopped run picks up where it left off.

Each request is one user message with the image(s) and the prompt.
"""
import asyncio
import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field

import yaml


@dataclass
class Model:
    name: str
    base_url: str
    model_id: str
    api_key_env: str | None = None   # env var holding the key; None for a local vLLM server
    image_first: bool = False        # put the image(s) before the prompt in the message
    params: dict = field(default_factory=dict)  # passed as-is to chat.completions.create


def load_models(path):
    """Read the models YAML into {name: Model}. `model_id` defaults to the name;
    ${VAR} in base_url is filled from the environment."""
    with open(path) as f:
        entries = yaml.safe_load(f)
    models = {}
    for name, entry in entries.items():
        entry["base_url"] = os.path.expandvars(entry["base_url"])
        models[name] = Model(name=name, model_id=entry.pop("model_id", name), **entry)
    return models


def image_part(path):
    """One image (a file path, or the image file's bytes) as a base64 data URL content part."""
    if isinstance(path, bytes):
        mime = "image/jpeg" if path[:3] == b"\xff\xd8\xff" else "image/png"
        data = base64.b64encode(path).decode()
    else:
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": "data:{};base64,{}".format(mime, data)}}


def build_messages(model, prompt, image_paths):
    images = [image_part(p) for p in image_paths]
    text = {"type": "text", "text": prompt}
    content = images + [text] if model.image_first else [text] + images
    return [{"role": "user", "content": content}]


def read_done(out_path):
    """Ids already answered in `out_path` (rows that ended in an error are asked again)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                if row["error"] is None:
                    done.add(row["id"])
    return done


async def _ask(client, model, request, semaphore, extra):
    async with semaphore:
        start = time.time()
        row = {"id": request["id"], "model": model.name, **extra,
               "response": None, "reasoning": None, "finish_reason": None, "usage": None, "error": None}
        try:
            reply = await client.chat.completions.create(
                model=model.model_id,
                messages=build_messages(model, request["prompt"], request.get("images", [])),
                **model.params,
            )
            message = reply.choices[0].message
            row["response"] = message.content
            # Thinking text, when the endpoint returns it: OpenRouter "reasoning", vLLM/LiteLLM "reasoning_content".
            extra_fields = message.model_extra or {}
            row["reasoning"] = extra_fields.get("reasoning") or extra_fields.get("reasoning_content")
            row["finish_reason"] = reply.choices[0].finish_reason  # "length" = cut off at max_tokens
            row["usage"] = reply.usage.model_dump() if reply.usage else None
        except Exception as e:  # keep going; the row is retried on the next run
            row["error"] = "{}: {}".format(type(e).__name__, e)
        row["seconds"] = round(time.time() - start, 2)
        return row


async def _generate(model, requests, out_path, concurrency, max_retries, timeout, extra):
    from openai import AsyncOpenAI

    api_key = os.environ[model.api_key_env] if model.api_key_env else "EMPTY"
    client = AsyncOpenAI(base_url=model.base_url, api_key=api_key, max_retries=max_retries, timeout=timeout)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(_ask(client, model, r, semaphore, extra)) for r in requests]
    rows = []
    with open(out_path, "a") as f:
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            row = await task
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            if i % 50 == 0 or i == len(tasks):
                errors = sum(r["error"] is not None for r in rows)
                print("{}/{} answered, {} errors".format(i, len(tasks), errors), flush=True)
    await client.close()
    return rows


def generate(model, requests, out_path, concurrency=32, max_retries=5, timeout=600, extra=None):
    """Ask `model` every request not yet answered in `out_path`; return the new rows.

    requests: [{"id": str, "prompt": str, "images": [path, ...]}, ...]
    Each written row: {"id", "model", **extra, "response", "reasoning", "finish_reason", "usage", "error",
    "seconds"}; "reasoning" is the model's thinking text when the endpoint returns it, else None;
    `extra` (e.g. {"mode": "steps"}) is copied into every row.
    The OpenAI client retries 429s, 5xx and timeouts `max_retries` times with backoff.
    """
    done = read_done(out_path)
    todo = [r for r in requests if r["id"] not in done]
    print("{}: {} requests, {} already answered, {} to ask".format(
        model.name, len(requests), len(requests) - len(todo), len(todo)), flush=True)
    if not todo:
        return []
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    return asyncio.run(_generate(model, todo, out_path, concurrency, max_retries, timeout, extra or {}))
