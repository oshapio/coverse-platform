"""Ask one model every question of one set; answers go to a JSONL file.

    python -m coverse_bench.run --set grid2d_d1 --model my-model --mode steps

--mode answer  asks each question as is
--mode steps   asks for every step too: one JSON object with the boxes of each step and the final answer

Rerunning the same command asks only the questions not answered yet. Models are defined in models.yaml.
"""
import argparse
import dataclasses
import json
import os
import time

from coverse_bench.data import DEFAULT_SOURCE, SETS, load_set
from coverse_bench.models import generate, load_models

PROMPT_KEYS = {"answer": "prompt", "steps": "steps_prompt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, choices=SETS)
    parser.add_argument("--model", required=True, help="a name in --models")
    parser.add_argument("--mode", choices=sorted(PROMPT_KEYS), default="answer")
    parser.add_argument("--models", default="models.yaml")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Hugging Face dataset or local dataset folder")
    parser.add_argument("--out", help="JSONL file (default: outputs/<set>/<model>_<mode>.jsonl)")
    parser.add_argument("--concurrency", type=int, default=32, help="requests in flight at once")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    args = parser.parse_args()

    model = load_models(args.models)[args.model]
    rows = load_set(args.set, args.source)[:args.limit]
    requests = [{"id": r["id"], "prompt": r[PROMPT_KEYS[args.mode]], "images": [r["image"]["bytes"]]} for r in rows]
    out = args.out or os.path.join("outputs", args.set, f"{args.model}_{args.mode}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    runs = json.load(open(out + ".meta.json")) if os.path.exists(out + ".meta.json") else []
    runs.append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "set": args.set, "source": args.source,
                 "mode": args.mode, "questions": len(requests), "model": dataclasses.asdict(model)})  # never the key
    json.dump(runs, open(out + ".meta.json", "w"), indent=1)
    generate(model, requests, out, concurrency=args.concurrency, extra={"mode": args.mode})
    print("answers ->", out)


if __name__ == "__main__":
    main()
