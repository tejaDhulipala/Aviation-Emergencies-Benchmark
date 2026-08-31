"""Scenario tag taxonomy for the benchmark dataset. Two categories: "Starting Condition Tags"
and "Expected Behavior Tags". Each category is a list of entries, where an entry is either:

- Tag(tag_id, label, description): a standalone, checkable tag.
- TagHeader(label, description, subtags): a grouping-only label (not itself checkable) whose
  subtags are Tags with slash-prefixed ids, e.g. "advanced_physics/turning_incorporation".

tag_id is the stable, machine-readable key stored in exported scenario.json files;
label/description are for the tagging UI only.
"""

from typing import NamedTuple


class Tag(NamedTuple):
    tag_id: str
    label: str
    description: str = ""


class TagHeader(NamedTuple):
    label: str
    description: str
    subtags: list


TAG_CATEGORIES = {
    "Starting Condition Tags": [
        Tag("standard_terrain", "Standard Terrain",
            "Over standard terrain like: airports, highways, open fields, forests, lakes, deserts, and urban areas."),
        TagHeader("Nonstandard Terrain", "Nonstandard terrain like: mountains, beaches, oceans, and canyons", [
            Tag("mountains", "Mountains", "Image includes mountainous terrain")
        ]),
        Tag("agl_under_1000", "< 1000 ft AGL"),
        Tag("agl_1000_3000", "1000-3000 ft AGL"),
        Tag("agl_3000_5000", "3000-5000 ft AGL"),
        Tag("agl_5000_7000", "5000-7000 ft AGL"),
        Tag("agl_7000_plus", "7000+ ft AGL"),
        Tag("engine_failure_on_departure", "Engine Failure on Departure"),
    ],
    "Expected Behavior Tags": [
        Tag("basic_physics", "Basic physics",
            "Check for reachability with basic glide ratio calculations (no advanced physics required). "
            "Scenario can have wind, but the LLM doesn't have to consider wind to choose the right answer "
            "(it would get the same answer with or without consideration of wind)."),
        Tag("basic_surface_heuristic", "Basic surface heuristic",
            "Choose the best surface with a basic heuristic (i.e. airport > road > field etc...)."),
        TagHeader("Advanced physics",
                  "Requires using calculations that require incorporation of turning physics or wind physics.",
                  [
                      Tag("advanced_physics/turning_incorporation", "Turning incorporation",
                          "Requires incorporating more height loss during a turn into aviation calculations."),
                      Tag("advanced_physics/wind_incorporation", "Wind incorporations",
                          "Requires incorporating ideas of worse glide ratio against wind into aviation calculations."),
                      Tag("advanced_physics/landing_distances", "Landing distances",
                          "The surface at the point is good (i.e. a field), but when you take into account "
                          "landing distances it is infeasible."),
                  ]),
        TagHeader("Tailwind consideration", "", [
            Tag("tailwind_consideration/turn_into_wind", "Turn into wind",
                "Requires use of the aviation knowledge that you should land with a headwind."),
            Tag("tailwind_consideration/land_with_tailwind", "Land with tailwind",
                "Lands with a tailwind in a case where it is necessary to do so."),
        ]),
        TagHeader("Advanced Surface Analysis", "", [
            Tag("advanced_surface_analysis/wave_consideration", "Wave consideration",
                "Depending on the wind strength there's a right and wrong way to land on waves."),
            Tag("advanced_surface_analysis/forest_type_consideration", "Forest type consideration",
                "When you have to choose a forested area, choose trees that are low and closely spaced together."),
            Tag("advanced_surface_analysis/obstacles_on_approach", "Detecting obstacles on approach to point",
                "The point the LLM will likely choose to land necessarily has an approach path with some "
                "obstacles (like powerlines)."),
            Tag("advanced_surface_analysis/highway_landings", "Highway landings",
                "Choosing to land with, instead of against, traffic."),
            Tag("advanced_surface_analysis/nighttime_decision_making", "Nighttime detection and decision-making",
                "Detecting surfaces during nighttime. Making the aeronautical decision to go for highways or "
                "airports as a first priority."),
            Tag("advanced_surface_analysis/farm_furrows", "Determining the correct direction to land in field given furrows",
                            "Making the decision to land parallel, as opposed to perpendicular to, furrows."),
        ]),
        Tag("harm_minimization", "Harm minimization",
            "Every option has some risk to civilians; choose the one with least expected damage."),
        Tag("anti_rule_following", "Anti-rule following",
            "Give it some instructions and test that it understands it should break rules and instructions "
            "during emergencies. For example, ATC telling it to do a left 360, or a runway being NOTAMed closed."),
        Tag("hallucination_avoidance", "Hallucination avoidance",
            "Doesn't fall for mismatches between text and image; trusts the image."),
    ],
}


def _leaf_tags(entries):
    for entry in entries:
        if isinstance(entry, TagHeader):
            yield from entry.subtags
        else:
            yield entry


def category_json_key(category_label):
    """'Starting Condition Tags' -> 'starting_condition_tags'."""
    return category_label.lower().replace(" ", "_")


TAG_ID_TO_CATEGORY = {
    tag.tag_id: category
    for category, entries in TAG_CATEGORIES.items()
    for tag in _leaf_tags(entries)
}


def split_tags_by_category(selected_tag_ids):
    """Returns {json_key: sorted [tag_id, ...]} for every category in TAG_CATEGORIES,
    partitioning selected_tag_ids by which category each belongs to. Unrecognized tag ids
    (e.g. from an older taxonomy) are ignored."""
    result = {category_json_key(category): [] for category in TAG_CATEGORIES}
    for tag_id in selected_tag_ids:
        category = TAG_ID_TO_CATEGORY.get(tag_id)
        if category is not None:
            result[category_json_key(category)].append(tag_id)
    for key in result:
        result[key].sort()
    return result
