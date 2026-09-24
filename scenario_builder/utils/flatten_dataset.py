"""Flattens dataset/'s nested class-organization structure into a single flat directory of
example_<i> folders (example_0 .. example_<n-1>) directly under dataset/, ordered by each
scenario's created_at field. Removes every other entry under dataset/ afterward (the old
category/altitude subfolders, including any that only held a placeholder .gitkeep) so
dataset/ ends up containing nothing but the example_<i> folders.

The folder-based class organization was never read by any code -- llm_run.py and this
notebook's setup cell already discover scenarios by walking dataset/ for scenario.json
wherever it lives, and the real tag data lives in each scenario.json's own
starting_condition_tags/expected_behavior_tags fields -- so flattening loses no information.

Run from the repo root: python3 -m scenario_builder.utils.flatten_dataset
"""

import json
import shutil
from pathlib import Path

DATASET_DIR = "dataset"
EXAMPLE_PREFIX = "example_"


def find_scenarios(dataset_dir):
    return sorted(Path(dataset_dir).rglob("scenario.json"))


def flatten_dataset(dataset_dir=DATASET_DIR):
    """Returns the new scenario.json paths, in example_<i> order. Safe to re-run: entries
    already at their target example_<i> path are left alone, and the two-phase move (via a
    temp name) avoids collisions between not-yet-moved sources and already-assigned targets.
    """
    dataset_root = Path(dataset_dir)
    scenario_paths = find_scenarios(dataset_root)

    entries = []
    for path in scenario_paths:
        scenario = json.loads(path.read_text())
        entries.append((scenario.get("created_at", ""), path.parent))
    entries.sort(key=lambda entry: entry[0])

    temp_dirs = []
    for i, (_, old_dir) in enumerate(entries):
        target_dir = dataset_root / f"{EXAMPLE_PREFIX}{i}"
        if old_dir.resolve() == target_dir.resolve():
            temp_dirs.append(old_dir)
            continue
        temp_dir = dataset_root / f"__flattening_{i}"
        shutil.move(str(old_dir), str(temp_dir))
        temp_dirs.append(temp_dir)

    example_dirs = []
    for i, temp_dir in enumerate(temp_dirs):
        final_dir = dataset_root / f"{EXAMPLE_PREFIX}{i}"
        if temp_dir.resolve() != final_dir.resolve():
            shutil.move(str(temp_dir), str(final_dir))
        example_dirs.append(final_dir)

    _remove_non_example_entries(dataset_root, example_dirs)
    return [d / "scenario.json" for d in example_dirs]


def _remove_non_example_entries(dataset_root, example_dirs):
    """Deletes everything directly under dataset_root except the given example_<i> dirs --
    the old (now-empty, or placeholder-only) category/altitude subfolders."""
    keep = {d.resolve() for d in example_dirs}
    for entry in dataset_root.iterdir():
        if entry.resolve() in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


if __name__ == "__main__":
    new_paths = flatten_dataset()
    for i, path in enumerate(new_paths):
        print(f"[{i}] {path}")
