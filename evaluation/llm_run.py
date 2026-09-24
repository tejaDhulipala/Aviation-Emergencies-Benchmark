"""Runs an LLM (via OpenRouter) on every scenario in the dataset/ tree, RUNS_PER_SCENARIO
times each, and reports per-run correctness plus aggregate accuracy (overall, per scenario,
and per tag) to the terminal.

Scenarios are discovered by recursively walking dataset/ for scenario.json files, wherever
they live in the class-organization folder tree -- so the folder structure itself is never
interpreted, only used to locate samples.

Each real (non-dry-run) run also writes evaluation/results/<model>_<runs>runs_<timestamp>/,
containing:
  - summary.txt: overall/per-tag/per-example accuracy, each two ways -- averaged (every run
    counts once) and majority vote (each example counts once, by its majority outcome).
    viewport:glide ratio and viewport_width_nm are bucketed and reported as extra tags
    alongside the real ones.
  - responses.txt: every example's tags (plus that same bucketed metadata) and every run's
    full raw model response with its right/wrong verdict.

Usage:
    python3 llm_run.py                      # full run, default model
    python3 llm_run.py --model openai/gpt-4o-mini --runs 1
    python3 llm_run.py --limit 2 --dry-run  # print the prompt for the first 2 scenarios, no API calls
"""

import argparse
import base64
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# Resolved from this file's own location rather than the working directory, so the script
# behaves the same whether it's run as `python3 evaluation/llm_run.py` from the repo root,
# `python3 llm_run.py` from inside evaluation/, or anything else.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
ENV_FILE = REPO_ROOT / ".env"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_NAME = "OPENROUTER-KEY"
RUNS_PER_SCENARIO = 3
DEFAULT_MODEL = "google/gemma-4-31b-it"
REQUEST_TIMEOUT_S = 120

# Bucket widths for treating the continuous viewport:glide ratio / viewport_width_nm
# metadata as extra "tags" in the per-tag accuracy breakdown, alongside the real tags.
GLIDE_RATIO_BUCKET_RESOLUTION = 0.25
VIEWPORT_WIDTH_BUCKET_RESOLUTION_NM = 1.0

OUTCOME_LABELS = {"correct": "CORRECT", "wrong": "WRONG", "unparseable": "UNPARSEABLE", "error": "ERROR"}

SYSTEM_PROMPT = """You are assisting a general aviation pilot flying a Cessna 172 whose engine \
has just failed. You must choose the single best place to attempt an emergency landing. 

You will be shown a top-down satellite image of the area, along with some flight parameters \
that are not visible in the image itself.

How to read the image:
- The aircraft's current position is marked with a red circle and a red arrow. The arrow \
points in the direction the aircraft is currently traveling (its heading).
- Yellow circles, each containing a number, mark candidate landing options. These are \
locations on the ground, not the aircraft's current position. The arrow shows the direction \
the aircraft would be traveling at touchdown if it landed there (e.g. the axis of a runway, \
road, or field). 
- White grid lines with red numeric labels form a ruler in nautical miles (nm), centered on \
the aircraft's current position (labeled +0.0/+0.0). North is up and east is right; a label's \
sign shows whether a line is north/east (+) or south/west (-) of the aircraft. Use these to \
judge distances and directions to each option.

You will also be given, as text (since these are not visible in the image): current altitude \
above ground level, airspeed, and wind speed/direction (the wind direction given is where the wind is blowing FROM, standard \
aviation convention). You may also be given additional situational information (e.g. ATC \
instructions, NOTAMs, or CTAF) -- use your judgement about whether such instructions should still be \
followed during an actual emergency.

Reason step by step about which candidate option is actually reachable and safest, considering \
glide distance, wind, the landing surface, obstacles, and any other relevant factors visible in the image. You may have to do actual calculations in order to reach the right answer. \
Assume the pilot can precisely execute any maneuver needed to reach an option, as long as it's physically achievable given the aircraft's performance.

Format your response as a JSON output in the following format:
{"reasoning": "Explanation for the answer choice given",
"answer": integer answer choice representing one of the yellow circles in the image. 
}
"""

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def load_api_key():
    env_path = ENV_FILE
    if not env_path.exists():
        sys.exit(f"Missing {ENV_FILE} -- expected a line like {OPENROUTER_KEY_NAME}=sk-or-...")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == OPENROUTER_KEY_NAME:
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    sys.exit(f"{OPENROUTER_KEY_NAME} not found (or empty) in {ENV_FILE}")


def _scenario_sort_key(scenario_path):
    """Sorts by the trailing number in the containing folder's name (e.g. 'scenario_10',
    'example_7') so ordering is numeric (0, 1, 2, ..., 10, 11) rather than lexicographic
    (0, 1, 10, 11, ..., 2). Falls back to the plain path string for folders that don't end
    in a number, sorted after all the numbered ones."""
    match = re.search(r"(\d+)$", scenario_path.parent.name)
    if match:
        return (0, int(match.group(1)), str(scenario_path))
    return (1, 0, str(scenario_path))


def find_scenarios(dataset_dir):
    return sorted(Path(dataset_dir).rglob("scenario.json"), key=_scenario_sort_key)


def build_user_text(scenario):
    lines = [
        f"Altitude AGL: {scenario['altitude_agl_ft']} ft",
        f"Airspeed: {scenario['airspeed_kt']} kt",
        f"Wind: from {scenario['wind_direction_deg']} degrees at {scenario['wind_speed_kt']} kt",
    ]
    prompt_additions = (scenario.get("prompt_additions") or "").strip()
    if prompt_additions:
        lines.append(f"Additional information: {prompt_additions}")
    lines.append("\nWhich landing option should the pilot choose?")
    return "\n".join(lines)


def build_messages(scenario, image_path):
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_text(scenario)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]


def call_openrouter(model, messages, api_key):
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages},
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_answer(response_text):
    """Parses the model's {"reasoning": ..., "answer": <int>} JSON response. Tolerates the
    JSON being wrapped in a ```json ... ``` fence, since some models add one despite
    instructions not to. Returns None if no integer "answer" can be extracted."""
    text = response_text.strip()
    candidates = [text]
    fence_match = JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.insert(0, fence_match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        answer = parsed.get("answer") if isinstance(parsed, dict) else None
        if isinstance(answer, bool):
            continue
        if isinstance(answer, int):
            return answer
        if isinstance(answer, str) and answer.strip().lstrip("-").isdigit():
            return int(answer.strip())
    return None


def correct_option_number(scenario):
    gt_index = scenario.get("ground_truth_index")
    if gt_index is None:
        return None
    return scenario["landing_options"][gt_index]["number"]


def slugify_model(model):
    """'google/gemma-4-31b-it' -> 'google-gemma-4-31b-it', safe for a folder name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def example_label(scenario_path, fallback_index):
    """'dataset/example_7/scenario.json' -> 'example_7' (also recognizes the older
    'scenario_7' folder-naming convention); falls back to a sequential label if the
    containing folder isn't named either way."""
    folder_name = scenario_path.parent.name
    if re.fullmatch(r"(example|scenario)_\d+", folder_name):
        return folder_name
    return f"example_{fallback_index}"


def bucket_value(value, resolution):
    return round(value / resolution) * resolution


def metadata_pseudo_tags(scenario):
    """viewport:glide ratio and viewport_width_nm, bucketed and treated as extra tags
    wherever tags are reported (per-tag accuracy breakdown, responses.txt tag listings)."""
    tags = []
    glide_ratio = scenario.get("viewport:glide ratio")
    if glide_ratio is not None:
        tags.append(f"viewport_glide_ratio~{bucket_value(glide_ratio, GLIDE_RATIO_BUCKET_RESOLUTION):.2f}")
    viewport_width = scenario.get("viewport_width_nm")
    if viewport_width is not None:
        tags.append(f"viewport_width_nm~{bucket_value(viewport_width, VIEWPORT_WIDTH_BUCKET_RESOLUTION_NM):.0f}")
    return tags


def majority_is_correct(outcomes):
    """Strict majority of 'correct' outcomes among a scenario's runs; ties count as not
    correct (e.g. 1/2 correct is not a majority)."""
    return outcomes.count("correct") > len(outcomes) / 2


def write_summary_file(path, model, runs_per_scenario, timestamp, examples, tag_results, tag_majority,
                        total_correct, total_wrong, total_unparseable, skipped_count):
    total_runs = total_correct + total_wrong + total_unparseable
    overall_majority_correct = sum(1 for ex in examples if majority_is_correct(ex["outcomes"]))

    lines = [
        "LLM Benchmark Run Summary",
        f"Model: {model}",
        f"Runs per scenario: {runs_per_scenario}",
        f"Timestamp: {timestamp}",
        f"Scenarios graded: {len(examples)} (skipped: {skipped_count})",
        "",
        "=== Overall ===",
    ]
    if total_runs:
        lines.append(f"Averaged (every run counts once):  {total_correct}/{total_runs} correct "
                      f"({100 * total_correct / total_runs:.1f}%)")
    if examples:
        lines.append(f"Majority vote (per example):       {overall_majority_correct}/{len(examples)} correct "
                      f"({100 * overall_majority_correct / len(examples):.1f}%)")
    lines.append("")

    lines.append("=== By tag ===")
    for tag in sorted(tag_results):
        avg_correct, avg_total = tag_results[tag]
        maj_correct, maj_total = tag_majority[tag]
        avg_pct = 100 * avg_correct / avg_total if avg_total else 0.0
        maj_pct = 100 * maj_correct / maj_total if maj_total else 0.0
        lines.append(tag)
        lines.append(f"    averaged: {avg_correct}/{avg_total} ({avg_pct:.1f}%)   "
                      f"majority: {maj_correct}/{maj_total} ({maj_pct:.1f}%)")
    lines.append("")

    lines.append("=== By example ===")
    for ex in examples:
        correct_count = ex["outcomes"].count("correct")
        total_count = len(ex["outcomes"])
        avg_pct = 100 * correct_count / total_count if total_count else 0.0
        verdict = "CORRECT" if majority_is_correct(ex["outcomes"]) else "WRONG"
        lines.append(f"{ex['label']}: averaged {correct_count}/{total_count} ({avg_pct:.1f}%)   majority: {verdict}")

    path.write_text("\n".join(lines) + "\n")


def write_responses_file(path, examples):
    lines = []
    for ex in examples:
        lines.append(f"=== {ex['label']} ===")
        lines.append(f"Path: {ex['path']}")
        lines.append(f"Tags: {', '.join(ex['tags'])}")
        lines.append(f"Correct answer: #{ex['correct_number']}")
        lines.append("")
        for i, run in enumerate(ex["runs"], start=1):
            outcome_label = OUTCOME_LABELS[run["outcome"]]
            if run["chosen_number"] is not None:
                header = f"--- run {i}/{len(ex['runs'])}: {outcome_label} (chose #{run['chosen_number']}) ---"
            else:
                header = f"--- run {i}/{len(ex['runs'])}: {outcome_label} ---"
            lines.append(header)
            lines.append(run["response_text"])
            lines.append("")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--runs", type=int, default=RUNS_PER_SCENARIO, help="Runs per scenario (default: 3)")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help=f"Root to search for scenario.json (default: {DATASET_DIR})")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N scenarios found")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt for each scenario instead of calling the API")
    args = parser.parse_args()

    scenario_paths = find_scenarios(args.dataset_dir)
    if args.limit is not None:
        scenario_paths = scenario_paths[: args.limit]
    if not scenario_paths:
        sys.exit(f"No scenario.json files found under {args.dataset_dir}/")

    api_key = None if args.dry_run else load_api_key()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    total_correct = total_wrong = total_unparseable = 0
    skipped_count = 0
    examples = []  # list of {label, path, tags, correct_number, runs, outcomes}
    tag_results = defaultdict(lambda: [0, 0])  # tag_id -> [correct, total] (averaged, per run)
    tag_majority = defaultdict(lambda: [0, 0])  # tag_id -> [correct, total] (majority vote, per example)

    for index, scenario_path in enumerate(scenario_paths):
        scenario = json.loads(scenario_path.read_text())
        image_path = scenario_path.parent / scenario["image_file"]
        correct_number = correct_option_number(scenario)
        if correct_number is None:
            print(f"[{scenario_path}] SKIPPED: no ground_truth_index set")
            skipped_count += 1
            continue

        messages = build_messages(scenario, image_path)
        tags = (scenario.get("starting_condition_tags", []) + scenario.get("expected_behavior_tags", [])
                + metadata_pseudo_tags(scenario))

        if args.dry_run:
            print(f"=== {scenario_path} ===")
            print("[SYSTEM]\n" + messages[0]["content"])
            print("[USER TEXT]\n" + messages[1]["content"][0]["text"])
            print(f"[IMAGE] {image_path} (attached)")
            print(f"[CORRECT ANSWER] {correct_number}\n")
            continue

        runs = []  # list of {response_text, chosen_number, outcome}
        for run_index in range(1, args.runs + 1):
            label = f"[{scenario_path}] run {run_index}/{args.runs}"
            try:
                response_text = call_openrouter(args.model, messages, api_key)
            except requests.RequestException as e:
                print(f"{label}: REQUEST FAILED ({e})")
                runs.append({"response_text": f"REQUEST FAILED: {e}", "chosen_number": None, "outcome": "error"})
                continue

            chosen_number = parse_answer(response_text)
            if chosen_number is None:
                print(f"{label}: UNPARSEABLE (no integer \"answer\" field found in JSON response)")
                runs.append({"response_text": response_text, "chosen_number": None, "outcome": "unparseable"})
                total_unparseable += 1
                continue

            if chosen_number == correct_number:
                print(f"{label}: chose #{chosen_number}, correct #{correct_number} -> CORRECT")
                outcome = "correct"
                total_correct += 1
            else:
                print(f"{label}: chose #{chosen_number}, correct #{correct_number} -> WRONG")
                outcome = "wrong"
                total_wrong += 1
            runs.append({"response_text": response_text, "chosen_number": chosen_number, "outcome": outcome})

            for tag in tags:
                tag_results[tag][1] += 1
                if outcome == "correct":
                    tag_results[tag][0] += 1

        outcomes = [r["outcome"] for r in runs]
        is_majority_correct = majority_is_correct(outcomes)
        for tag in tags:
            tag_majority[tag][1] += 1
            if is_majority_correct:
                tag_majority[tag][0] += 1

        examples.append({
            "label": example_label(scenario_path, index),
            "path": scenario_path,
            "tags": tags,
            "correct_number": correct_number,
            "runs": runs,
            "outcomes": outcomes,
        })

    if args.dry_run:
        return

    total_runs = total_correct + total_wrong + total_unparseable
    print("\n=== Per-scenario results ===")
    for ex in examples:
        summary = "/".join(o[0].upper() for o in ex["outcomes"])
        print(f"{ex['path']}: {summary}")

    print("\n=== Aggregate results ===")
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(examples)}, runs per scenario: {args.runs}, total runs: {total_runs}")
    if total_runs:
        print(f"Correct:     {total_correct:3d} ({100 * total_correct / total_runs:.1f}%)")
        print(f"Wrong:       {total_wrong:3d} ({100 * total_wrong / total_runs:.1f}%)")
        print(f"Unparseable: {total_unparseable:3d} ({100 * total_unparseable / total_runs:.1f}%)")

    if tag_results:
        print("\n=== Accuracy by tag ===")
        for tag in sorted(tag_results):
            correct, total = tag_results[tag]
            print(f"{tag}: {correct}/{total} ({100 * correct / total:.1f}%)")

    run_dir = RESULTS_DIR / f"{slugify_model(args.model)}_{args.runs}runs_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_summary_file(run_dir / "summary.txt", args.model, args.runs, timestamp, examples,
                        tag_results, tag_majority, total_correct, total_wrong, total_unparseable, skipped_count)
    write_responses_file(run_dir / "responses.txt", examples)
    print(f"\nWrote {run_dir}/summary.txt and {run_dir}/responses.txt")


if __name__ == "__main__":
    main()
