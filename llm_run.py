"""Runs an LLM (via OpenRouter) on every scenario in the dataset/ tree, RUNS_PER_SCENARIO
times each, and reports per-run correctness plus aggregate accuracy (overall, per scenario,
and per tag).

Scenarios are discovered by recursively walking dataset/ for scenario.json files, wherever
they live in the class-organization folder tree -- so the folder structure itself is never
interpreted, only used to locate samples.

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
from collections import defaultdict
from pathlib import Path

import requests

DATASET_DIR = "dataset"
ENV_FILE = ".env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_NAME = "OPENROUTER-KEY"
RUNS_PER_SCENARIO = 3
DEFAULT_MODEL = "google/gemma-4-31b-it"
REQUEST_TIMEOUT_S = 120

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
instructions or NOTAMs) -- use your judgement about whether such instructions should still be \
followed during an actual emergency.

Reason step by step about which candidate option is actually reachable and safest, considering \
glide distance, wind, the landing surface, obstacles, and any other relevant factors visible in the image. \
Assume the pilot can precisely execute any maneuver needed to reach an option, as long as it's physically achievable given the aircraft's performance.

Format your response as a JSON output in the following format:
{"reasoning": "Explanation for the answer choice given",
"answer": integer answer choice representing one of the yellow circles in the image. 
}
"""

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def load_api_key():
    env_path = Path(ENV_FILE)
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


def find_scenarios(dataset_dir):
    return sorted(Path(dataset_dir).rglob("scenario.json"))


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

    total_correct = total_wrong = total_unparseable = 0
    per_scenario_results = []  # (path, [outcome, ...])
    tag_results = defaultdict(lambda: [0, 0])  # tag_id -> [correct, total]

    for scenario_path in scenario_paths:
        scenario = json.loads(scenario_path.read_text())
        image_path = scenario_path.parent / scenario["image_file"]
        correct_number = correct_option_number(scenario)
        if correct_number is None:
            print(f"[{scenario_path}] SKIPPED: no ground_truth_index set")
            continue

        messages = build_messages(scenario, image_path)
        tags = scenario.get("starting_condition_tags", []) + scenario.get("expected_behavior_tags", [])

        if args.dry_run:
            print(f"=== {scenario_path} ===")
            print("[SYSTEM]\n" + messages[0]["content"])
            print("[USER TEXT]\n" + messages[1]["content"][0]["text"])
            print(f"[IMAGE] {image_path} (attached)")
            print(f"[CORRECT ANSWER] {correct_number}\n")
            continue

        outcomes = []
        for run_index in range(1, args.runs + 1):
            label = f"[{scenario_path}] run {run_index}/{args.runs}"
            try:
                response_text = call_openrouter(args.model, messages, api_key)
            except requests.RequestException as e:
                print(f"{label}: REQUEST FAILED ({e})")
                outcomes.append("error")
                continue

            chosen_number = parse_answer(response_text)
            if chosen_number is None:
                print(f"{label}: UNPARSEABLE (no integer \"answer\" field found in JSON response)")
                outcomes.append("unparseable")
                total_unparseable += 1
                continue

            if chosen_number == correct_number:
                print(f"{label}: chose #{chosen_number}, correct #{correct_number} -> CORRECT")
                outcomes.append("correct")
                total_correct += 1
            else:
                print(f"{label}: chose #{chosen_number}, correct #{correct_number} -> WRONG")
                outcomes.append("wrong")
                total_wrong += 1

            for tag in tags:
                tag_results[tag][1] += 1
                if outcomes[-1] == "correct":
                    tag_results[tag][0] += 1

        per_scenario_results.append((scenario_path, outcomes))

    if args.dry_run:
        return

    total_runs = total_correct + total_wrong + total_unparseable
    print("\n=== Per-scenario results ===")
    for path, outcomes in per_scenario_results:
        summary = "/".join(o[0].upper() for o in outcomes)
        print(f"{path}: {summary}")

    print("\n=== Aggregate results ===")
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(per_scenario_results)}, runs per scenario: {args.runs}, total runs: {total_runs}")
    if total_runs:
        print(f"Correct:     {total_correct:3d} ({100 * total_correct / total_runs:.1f}%)")
        print(f"Wrong:       {total_wrong:3d} ({100 * total_wrong / total_runs:.1f}%)")
        print(f"Unparseable: {total_unparseable:3d} ({100 * total_unparseable / total_runs:.1f}%)")

    if tag_results:
        print("\n=== Accuracy by tag ===")
        for tag in sorted(tag_results):
            correct, total = tag_results[tag]
            print(f"{tag}: {correct}/{total} ({100 * correct / total:.1f}%)")


if __name__ == "__main__":
    main()
