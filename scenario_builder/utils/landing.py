from .constants import (
    LANDING_DIST_50FT_BY_TEMP_C,
    MAX_FLAPS,
    VALID_FLAP_SETTINGS,
    HEADWIND_REDUCTION_PER_KT,
    TAILWIND_INCREASE_PER_KT,
    TAILWIND_MAX_DEMONSTRATED_KT,
    TAILWIND_MAX_FACTOR,
    NO_FLAP_DISTANCE_INCREASE_PCT,
    FT_PER_NM,
)

def landing_distance(temperature, headwind, flaps):
    # temperature C
    # headwind knots
    # returns in nm
    assert flaps in VALID_FLAP_SETTINGS
    # Assuming 1000ft pressure altitude
    # Round temperature to nearest tabulated value
    nearest_temp = min(LANDING_DIST_50FT_BY_TEMP_C.keys(), key=lambda t: abs(temperature - t))
    fifty_ft = LANDING_DIST_50FT_BY_TEMP_C[nearest_temp]
    if headwind < 0:
        if headwind <= -TAILWIND_MAX_DEMONSTRATED_KT:
            fifty_ft *= TAILWIND_MAX_FACTOR
        else:
            fifty_ft *= 1 - (headwind / TAILWIND_INCREASE_PER_KT / 10)
    else:
        fifty_ft *= 1 - (headwind / HEADWIND_REDUCTION_PER_KT / 10)
    fifty_ft *= 1 + ((MAX_FLAPS - flaps) / MAX_FLAPS * NO_FLAP_DISTANCE_INCREASE_PCT / 100)
    return fifty_ft / FT_PER_NM
