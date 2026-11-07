"""Score a model's answers to one set of the benchmark.

    python -m coverse_bench.score --set grid2d_d1 --preds outputs/grid2d_d1/<model>_<mode>.jsonl

--preds is the file coverse_bench.run writes: one JSON line per question with "id" (the dataset row's id),
"mode" ("answer" or "steps") and "response". The truth comes from the dataset: "answer", and for
steps mode each step's object_ids ("program") and the objects' boxes ("boxes").

What counts as correct:
- the final answer: an option letter (questions with "A. ..." option lines), a count (last integer in
  the reply), or a JSON list (rankings; same items, same order);
- a box step (steps mode): as many boxes as true objects, and a match score above the IoU threshold.
  The match score is 2 * (sum of IoUs of the Hungarian-matched pairs) / (#true + #predicted boxes);
  both empty scores 1. Step accuracy is reported at every threshold in --iou (default 0.2, 0.3, ... 0.9;
  the paper used 0.8). Match scores are stored per step, so other thresholds cost no model calls;
- a whole program (steps mode): the answer and every box step correct, at each threshold.
Replies that errored or were cut off at max_tokens count as wrong; they are also counted separately.

Writes <preds>.scores.json (summary + one entry per question) and prints the summary.
"""
import argparse
import ast
import json
import math
import re
from collections import defaultdict

from scipy.optimize import linear_sum_assignment

from coverse_bench.data import DEFAULT_SOURCE, load_set

IOU_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------- reading the answer in a reply

def parse_literal(text):
    """A dict or list written as JSON or as a Python literal, else None."""
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(text)
        except Exception:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def parse_unquoted_list(text):
    """'[B, C, "A", 2]' -> ['B', 'C', 'A', 2], a list whose items are not all quoted; else None."""
    if not (text.startswith("[") and text.endswith("]")):
        return None
    items = []
    for item in text[1:-1].split(","):
        item = item.strip()
        if not item:
            continue
        if len(item) >= 2 and item[0] == item[-1] and item[0] in "'\"":
            items.append(item[1:-1].strip())
        elif re.fullmatch(r"[-+]?\d+", item):
            items.append(int(item))
        elif re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", item):
            items.append(float(item))
        else:
            items.append(item)
    return items


def bracketed_parts(text, open_char, close_char):
    """Every balanced open_char...close_char span in text, in order of where it starts."""
    for start, char in enumerate(text):
        if char != open_char:
            continue
        depth, in_string, escaped = 0, False, False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    yield text[start:end + 1]
                    break


def parse_json(text):
    """The JSON object or list in a reply: the whole reply, the reply without ``` fences, or the first
    {...} (then [...]) in it that parses. None if there is none."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    unfenced = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in [text, unfenced]:
        value = parse_literal(candidate)
        if value is None:
            value = parse_unquoted_list(candidate)
        if value is not None:
            return value
    for open_char, close_char in ["{}", "[]"]:
        for part in bracketed_parts(unfenced, open_char, close_char):
            value = parse_literal(part)
            if value is None and open_char == "[":
                value = parse_unquoted_list(part)
            if value is not None:
                return value
    return None


ANSWER_CUE = r"(?i)(?:final answer|answer|option|choice|letter|choose|select)\s*(?:is|:|=)?\s*([A-Za-z])\b"
CAPITAL_LETTER = r"\b([A-Z])\b"
ANY_LETTER = r"\b([A-Za-z])\b"


def option_letters(prompt):
    """The letters of the "A. ..." option lines in a prompt, in order."""
    return list(dict.fromkeys(re.findall(r"(?m)^\s*([A-Z])(?:[.):]?)\s+\S", prompt or "")))


def chosen_letter(reply, options=None):
    """The option letter a reply picks: the last one after a cue ("answer: C"), else the last
    standalone capital letter, else the last standalone letter. Letters that are not options are
    skipped, unless that leaves none."""
    reply = str(reply).strip()
    for allowed in ([options, None] if options else [None]):
        for pattern in (ANSWER_CUE, CAPITAL_LETTER, ANY_LETTER):
            found = [letter.upper() for letter in re.findall(pattern, reply)]
            if allowed:
                found = [letter for letter in found if letter in allowed]
            if found:
                return found[-1]
    return None


def last_number(text):
    numbers = re.findall(r"\d+", str(text))
    return numbers[-1] if numbers else None


def normalize(value):
    """Letters 'b.' -> 'B', other text lower-cased, inside lists and dicts too."""
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        return text.rstrip(".").upper() if re.fullmatch(r"[A-Za-z]\.?", text) else text.lower()
    return value


def json_matches(truth, pred):
    """Every key/value of truth is in pred (extra keys ignored); lists: same length, item by item."""
    if isinstance(truth, dict):
        return isinstance(pred, dict) and all(key in pred and json_matches(item, pred[key]) for key, item in truth.items())
    if isinstance(truth, list):
        return isinstance(pred, list) and len(truth) == len(pred) and all(map(json_matches, truth, pred))
    if isinstance(truth, str) and isinstance(pred, str):
        return truth.strip().lower() == pred.strip().lower()
    return truth == pred


def answer_correct(prompt, pred, truth):
    """Is the predicted final answer (`pred`, as the model gave it) the true one (`truth`, as stored)?"""
    truth_json = parse_json(truth)
    if truth_json is not None:                                       # rankings: '["B", "C", "D", "A"]'
        pred_json = parse_json(pred)
        if isinstance(truth_json, list) and isinstance(pred_json, dict):  # {"answer": [...]} -> [...]
            for key in ("order", "answer"):
                if isinstance(pred_json.get(key), list):
                    pred_json = pred_json[key]
                    break
        if isinstance(truth_json, dict) and set(truth_json) == {"order"} and isinstance(pred_json, list):
            pred_json = {"order": pred_json}
        return pred_json is not None and json_matches(normalize(truth_json), normalize(pred_json))
    options = option_letters(prompt)
    if options:                                                      # option letters
        pred_letter = chosen_letter(pred, options)
        return pred_letter is not None and pred_letter == chosen_letter(truth, options)
    pred_number, true_number = last_number(pred), last_number(truth)  # counts
    if pred_number is not None or true_number is not None:
        return pred_number is not None and pred_number == true_number
    return pred is not None and str(pred).strip().lower() == str(truth).strip().lower()


# ---------------------------------------------------------------- boxes

# Box formats a model may use instead of the asked {"x_min", "y_min", "x_max", "y_max"}. Only formats that
# say which number is which: corner names, or a [x_min, y_min, x_max, y_max] list under one of these keys
# (Qwen answers {"bbox_2d": [...], "label": ...}). Not read: Gemini's native "box_2d", which is
# [y_min, x_min, y_max, x_max] (the other order), and any list whose order is not stated.
CORNER_NAMES = [("x_min", "y_min", "x_max", "y_max"), ("xmin", "ymin", "xmax", "ymax"),
                ("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")]
CORNER_LIST_KEYS = ("bbox_2d", "bbox", "box", "bounding_box")
STEP_BOX_KEYS = ("bboxes", "boxes")


def box_corners(box):
    """The 4 values of a dict box in [x_min, y_min, x_max, y_max] order (see CORNER_NAMES,
    CORNER_LIST_KEYS, and {"x", "y", "width", "height"}); None for a dict in no known format."""
    for names in CORNER_NAMES:
        if all(name in box for name in names):
            return [box[name] for name in names]
    for key in CORNER_LIST_KEYS:
        if isinstance(box.get(key), (list, tuple)):
            return box[key]
    if all(name in box for name in ("x", "y", "width", "height")):
        try:
            x, y, width, height = (float(box[name]) for name in ("x", "y", "width", "height"))
        except (TypeError, ValueError):
            return None
        return [x, y, x + width, y + height]
    return None


def read_box(box):
    """[x_min, y_min, x_max, y_max] from a 4-number list or a dict box (see box_corners), if all 4
    are finite numbers >= 0; else None."""
    if isinstance(box, dict):
        box = box_corners(box)
    try:
        box = [float(v) for v in box]
    except Exception:
        return None
    return box if len(box) == 4 and all(math.isfinite(v) and v >= 0 for v in box) else None


def read_boxes(value):
    """The valid boxes in one box or a list of boxes (see read_box); invalid ones are dropped."""
    if value is None or isinstance(value, str) or not hasattr(value, "__len__"):
        return []
    one_box = isinstance(value, dict) or (len(value) == 4 and not hasattr(value[0], "__len__"))
    boxes = [value] if one_box else value
    return [box for box in map(read_box, boxes) if box is not None]


def iou(a, b):
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def match_score(truth, pred):
    """2 * (sum of IoUs of the Hungarian-matched pairs) / (#true + #pred); 1 if both are empty."""
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    ious = [[iou(t, p) for p in pred] for t in truth]
    rows, cols = linear_sum_assignment(ious, maximize=True)
    return 2 * sum(ious[r][c] for r, c in zip(rows, cols)) / (len(truth) + len(pred))


def step_correct(step, threshold):
    """A box step is correct at a threshold: as many boxes as true objects, match score above it."""
    return step["true_boxes"] == step["pred_boxes"] and step["score"] > threshold


# ---------------------------------------------------------------- one question

def reply_steps(reply):
    """{"S0": ..., "S1": ...} from a parsed steps-mode reply (under "steps", or at the top)."""
    if not isinstance(reply, dict):
        return {}
    if isinstance(reply.get("steps"), dict):
        return reply["steps"]
    return {str(key): value for key, value in reply.items() if str(key).startswith("S")}


def reply_final_answer(reply, steps, final_ref):
    """The final answer in a parsed steps-mode reply: steps[final]["answer"], else the answer the
    reply gives some other way (under the final step's name, under "answer", or the reply itself)."""
    payload = steps.get(final_ref, {})
    answer = payload.get("answer") if isinstance(payload, dict) else payload
    if answer is not None:
        return answer
    if isinstance(reply, list):
        return reply
    if not isinstance(reply, dict):
        return None
    payload = reply.get(final_ref)
    if isinstance(payload, dict) and payload.get("answer") is not None:
        return payload["answer"]
    if payload is not None and not isinstance(payload, dict):
        return payload
    if reply.get("answer") is not None:
        return reply["answer"]
    has_steps = isinstance(reply.get("steps"), dict) or any(str(key).startswith("S") for key in reply)
    return None if has_steps else reply


def ends_in_boxes(program):
    """True for a question whose answer is a group of objects (e.g. "S0 = all the characters in the
    image."): its last step asks for boxes, and there is no answer step."""
    return bool(program) and program["steps"][-1].get("output_kind") == "bboxes"


def score_question(mode, prompt, response, truth, program=None, char_boxes=None):
    """Score one reply. `program` (meta.instruction_program) and `char_boxes` (bboxes_chars) are
    needed for steps mode only. Box steps get their match score and box counts; correctness per
    IoU threshold is decided later (step_correct).
    A question that ends in boxes has no answer to check ("correct" is None); it is right at a
    threshold when all its box steps are. Its answer-mode prompt is the steps prompt, so both modes
    are scored the same way."""
    if mode == "answer" and not ends_in_boxes(program):
        return {"correct": answer_correct(prompt, response, truth)}
    reply = parse_json(response)
    steps = reply_steps(reply)
    box_steps = []
    for step in program["steps"]:
        kind = step.get("output_kind") or ("answer" if str(step.get("filter", "")).startswith("answer_") else "bboxes")
        if kind != "bboxes":
            continue
        true_boxes = read_boxes([char_boxes[i] for i in step["object_ids"] if 0 <= int(i) < len(char_boxes)])
        given = steps.get(step["step_ref"], {})
        if isinstance(given, dict):
            given = next((given[key] for key in STEP_BOX_KEYS if key in given), [])
        pred_boxes = read_boxes(given)
        box_steps.append({"step": step["step_ref"], "score": round(match_score(true_boxes, pred_boxes), 4),
                          "true_boxes": len(true_boxes), "pred_boxes": len(pred_boxes)})
    correct = None if ends_in_boxes(program) else \
        answer_correct(prompt, reply_final_answer(reply, steps, program["final_step_ref"]), truth)
    return {"correct": correct, "json_ok": reply is not None, "steps": box_steps}


# ---------------------------------------------------------------- a whole answers file

def rate(hits, total):
    return round(hits / total, 4) if total else None


def summary_of(results, thresholds):
    """Accuracy (and in steps mode: step accuracy and program accuracy per IoU threshold, mean match
    score, share of valid JSON) for a list of scored questions. Accuracy counts only questions that
    ask for an answer (None when none does)."""
    answered = [r for r in results if r["correct"] is not None]     # questions that ask for an answer
    out = {"questions": len(results), "accuracy": rate(sum(r["correct"] for r in answered), len(answered))}
    if results and "steps" in results[0]:
        steps = [s for r in results for s in r["steps"]]
        out["step_accuracy"] = {str(t): rate(sum(step_correct(s, t) for s in steps), len(steps)) for t in thresholds}
        out["program_accuracy"] = {
            str(t): rate(sum(r["correct"] is not False and all(step_correct(s, t) for s in r["steps"]) for r in results),
                 len(results))
            for t in thresholds}
        out["step_score"] = rate(sum(s["score"] for s in steps), len(steps))
        out["json_ok"] = rate(sum(r["json_ok"] for r in results), len(results))
    return out


def summarize(results, thresholds):
    by_type, by_depth = defaultdict(list), defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)
        by_depth[r["depth"]].append(r)
    summary = summary_of(results, thresholds)
    summary["errors"] = sum(r["error"] for r in results)
    summary["cut_off"] = sum(r["cut_off"] for r in results)
    summary["by_depth"] = {str(k): summary_of(v, thresholds) for k, v in sorted(by_depth.items())}
    summary["by_question_type"] = {k: summary_of(v, thresholds) for k, v in sorted(by_type.items())}
    return summary


def score_file(set_name, preds_path, thresholds=IOU_THRESHOLDS, source=DEFAULT_SOURCE):
    rows = {row["id"]: row for row in load_set(set_name, source, images=False)}
    with open(preds_path) as f:
        preds = [json.loads(line) for line in f]
    latest = {p["id"]: p for p in preds}                       # a rerun's answer replaces an errored one
    results = []
    for qid in sorted(latest, key=int):
        p, row = latest[qid], rows[qid]
        prompt = row["prompt" if p["mode"] == "answer" else "steps_prompt"]
        r = score_question(p["mode"], prompt, p["response"], json.loads(row["answer"]),
                           json.loads(row["program"]), json.loads(row["boxes"]))
        r.update({"id": qid, "question_type": row["question_type"], "depth": row["depth"],
                  "error": p.get("error") is not None, "cut_off": p.get("finish_reason") == "length"})
        results.append(r)
    return {"set": set_name, "preds": preds_path, "mode": preds[0]["mode"] if preds else None,
            "iou_thresholds": thresholds, "unanswered": len(rows) - len(results),
            "summary": summarize(results, thresholds), "questions": results}


def print_summary(scores):
    """One table per grouping (all, by depth, by question type); in steps mode one column per IoU
    threshold for step accuracy (step@t) and program accuracy (prog@t)."""
    def cell(value):
        return "-" if value is None else "{:.3f}".format(value)

    s = scores["summary"]
    print("{} questions ({} in the set not answered), mode {}, errors {}, cut off {}".format(
        s["questions"], scores["unanswered"], scores["mode"], s["errors"], s["cut_off"]))
    steps_mode = "step_accuracy" in s
    columns = ["n", "answer"]
    if steps_mode:
        columns += ["json_ok"] + ["step@{}".format(t) for t in s["step_accuracy"]] + ["prog@{}".format(t) for t in s["program_accuracy"]]
    groups = [("all", {"all": s}), ("by depth", s["by_depth"]), ("by question type", s["by_question_type"])]
    for title, rows in groups:
        print("\n{:<44}".format(title) + "".join("{:>9}".format(c) for c in columns))
        for name, row in rows.items():
            values = [str(row["questions"]), cell(row["accuracy"])]
            if steps_mode:
                values += [cell(row["json_ok"])] + [cell(v) for v in row["step_accuracy"].values()] \
                          + [cell(v) for v in row["program_accuracy"].values()]
            print("{:<44}".format(name[:43]) + "".join("{:>9}".format(v) for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, help="set name, e.g. grid2d_d1")
    parser.add_argument("--preds", required=True, help="JSONL written by coverse_bench.run")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Hugging Face dataset or local dataset folder")
    parser.add_argument("--iou", type=float, nargs="+", default=IOU_THRESHOLDS,
                        help="IoU thresholds for box steps (default: %(default)s; the paper used 0.8)")
    args = parser.parse_args()

    scores = score_file(args.set, args.preds, args.iou, args.source)
    out = args.preds.rsplit(".jsonl", 1)[0] + ".scores.json"
    with open(out, "w") as f:
        json.dump(scores, f, indent=1)
    print_summary(scores)
    print("scores ->", out)


if __name__ == "__main__":
    main()
