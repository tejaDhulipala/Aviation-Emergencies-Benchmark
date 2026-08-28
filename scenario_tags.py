"""Scenario tag taxonomy for the benchmark dataset. Each category is a list of
(tag_id, label, description) tuples. tag_id is the stable, machine-readable key stored in
exported scenario.json files; label/description are for the tagging UI only.

"Highway landings" appears under both Starting Condition (two opposite-direction options on
a highway) and Expected Behavior (choosing the with-traffic direction) -- these are related
but distinct concepts, so they get distinct tag_ids (highway_landings_option, highway_landings_traffic_direction).
"""

TAG_CATEGORIES = {
    "Starting Condition": [
        ("enroute_standard", "Enroute standard",
         "Above 3000 AGL, over standard terrain: airports, highways, open fields, forests, lakes, deserts, urban areas."),
        ("airport_in_reach", "Airport",
         "Easy test -- there is an airport within reach."),
        ("reject_safest_surface", "Reject safest surface due to physics constraints",
         "The LLM must reject the safest-seeming surface (runway, highway, open field) because of physics "
         "constraints, and instead choose a worse surface (trees, water, marsh)."),
        ("lower_altitudes", "Lower altitudes",
         "1000-3000 AGL. Good for testing things like whether turns matter."),
        ("engine_failure_on_climb", "Engine failure on climb",
         "200-1000 AGL. Engine failure on departure leg or crosswind after takeoff."),
        ("coastal", "Coastal",
         "Engine failure with beach and/or ocean in the frame. May require considering wave direction and wind strength."),
        ("obstacles_on_approach_starting", "Obstacles on approach",
         "Some of the proposed options have obstacles enroute to the landing spot."),
        ("nighttime_starting", "Nighttime",
         "Scenario takes place at night."),
        ("highway_landings_option", "Highway landings",
         "Has two options on a highway in opposite directions."),
    ],
    "Scenario Modification": [
        ("faulty_option_choices", "Faulty option choices",
         "There is a mismatch between what is on the screen and what is being described."),
        ("instructions", "Instructions",
         "There are NOTAMs or instructions from ATC. Sometimes the LLM should follow them, sometimes disregard them."),
    ],
    "Expected Behavior": [
        ("basic_physics", "Basic physics",
         "Check reachability with basic glide ratio calculations (no advanced physics required). Scenario can have "
         "wind, but the LLM doesn't need to consider it to choose the right answer."),
        ("basic_surface_heuristic", "Basic surface heuristic",
         "Choose the best surface with a basic heuristic (airport > road > field > ...)."),
        ("advanced_physics", "Advanced physics",
         "Requires calculations incorporating turning physics or wind physics."),
        ("turning_incorporation", "Turning incorporation",
         "Requires incorporating extra height loss during a turn into the calculation."),
        ("wind_incorporation", "Wind incorporation",
         "Requires incorporating worse glide ratio against the wind into the calculation."),
        ("turn_into_wind", "Turn into wind",
         "Requires knowing you should land with a headwind."),
        ("land_with_tailwind", "Land with tailwind",
         "Lands with a tailwind in a case where it's actually necessary to do so."),
        ("landing_distances", "Landing distances",
         "The surface at the point is good (e.g. a field), but factoring in landing distance makes it infeasible."),
        ("wave_consideration", "Wave consideration",
         "Depending on wind strength, there's a right and wrong way to land on waves."),
        ("forest_type_consideration", "Forest type consideration",
         "When forced into a forested area, choose trees that are low and closely spaced."),
        ("obstacles_on_approach_behavior", "Detecting obstacles on approach to point",
         "The likely landing choice has an approach path with obstacles (e.g. powerlines)."),
        ("nighttime_decision_making", "Nighttime detection and decision-making",
         "Detecting surfaces at night; prioritizing highways or airports as the first choice."),
        ("highway_landings_traffic_direction", "Highway landings (traffic direction)",
         "Choosing to land with, not against, traffic."),
        ("harm_minimization", "Harm minimization",
         "Every option risks civilians; choose the one with least expected damage."),
        ("anti_rule_following", "Anti-rule following",
         "Instructions are given that the LLM should recognize it needs to break during an emergency "
         "(e.g. ATC telling it to do a left 360, or a runway being NOTAMed closed)."),
        ("hallucination_avoidance", "Hallucination avoidance",
         "Doesn't fall for mismatches between text and image; trusts the image."),
    ],
}
