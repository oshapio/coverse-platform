# coverse-platform

Code for running vision-language models on the CoVerse benchmark and scoring what they say.

The benchmark has 2,176 questions about images. The images come from four sources: flat 2D grids of characters
(grid2d), rendered 3D characters (chars3d), ProcTHOR house interiors (procthor3d) and real indoor scans (ca1m3d).
Every question is built from a small program, and the questions get harder as the program gets longer. Depth 1 asks
one thing, while depth 3 chains several steps, like "find the chairs, keep the ones closer to the table than the
lamp is, then count them". Each environment comes in three depths, so there are 12 sets in total.

The data is on Hugging Face at
[`Gigglingface/coverse-bench`](https://huggingface.co/datasets/Gigglingface/coverse-bench). The code downloads it
the first time you run it.

## Setup

```bash
git clone git@github.com:oshapio/coverse-platform.git
cd coverse-platform
pip install -e .
```

`uv sync` works too.

## Running a model

Any model behind an OpenAI-compatible API will do: your own vLLM server, OpenRouter, OpenAI and so on. Add it to
`models.yaml`. There are a few examples in there already. API keys are read from environment variables and never
written anywhere.

Then ask it one set and score the answers:

```bash
python -m coverse_bench.run --set grid2d_d1 --model vllm --mode answer
python -m coverse_bench.score --set grid2d_d1 --preds outputs/grid2d_d1/vllm_answer.jsonl
```

Use `--limit 5` to check that everything works before you pay for a full run. If a run dies halfway, just start it
again. It skips the questions that already have an answer and retries the ones that failed.

To do the whole benchmark in both modes:

```bash
for env in grid2d chars3d procthor3d ca1m3d; do
  for d in 1 2 3; do
    for mode in answer steps; do
      python -m coverse_bench.run --set ${env}_d$d --model vllm --mode $mode
      python -m coverse_bench.score --set ${env}_d$d --preds outputs/${env}_d$d/vllm_$mode.jsonl
    done
  done
done
```

## The two modes

In **answer** mode the model sees the image and the question and only has to give the final answer: a letter, a
number, or an ordered list.

In **steps** mode we also show it the program, step by step. It has to draw boxes around the objects each step
picks out and then give the final answer. This shows us where a model goes wrong, not just whether it does. Boxes
use a 0 to 1000 grid over the image.

## How scoring works

The final answer is right if the letter, the count or the whole list matches. A list counts only when every item is
in the right order. For box steps we pair the model's boxes with the true ones and compute a match score from their
overlap (IoU). A step counts as right when the model gave the right number of boxes and the match score is above a
threshold. We report several thresholds. "Program accuracy" means the answer and every single box step were right.
That's strict, and on depth 3 almost no model manages it.

`score` prints a summary and writes `<preds>.scores.json`, which breaks the results down by depth and by question
type.
