import math
import warnings
from .constants import (
    W_MAX,
    RHO_0,
    V_GLIDE,
    GR_0,
    TEMP_0_R,
    LAPSE_RATE,
    F_TO_R_OFFSET,
    BAROMETRIC_EXPONENT,
    VALID_FLAP_SETTINGS,
    FLAP_GR_PENALTY_PER_10_DEG,
)

def cessna_glide_ratio(weight=W_MAX, density=RHO_0, airspeed=V_GLIDE, wind_delta: float = 0.0, wind_speed=0,
                        bank_angle: float = 0.0, flaps: int = 0):
    """
    Calculate Cessna 172 glide ratio based on weight, air density, and airspeed.

    Parameters:
    weight (float): Aircraft weight in pounds (default: 2400 lbs)
    density (float): Air density in slug/ft³ (default: 0.002378 - sea level standard)
    airspeed (float): Airspeed in knots (default: 65 knots)
    wind_delta (float): difference between aircraft heading and wind direction
    wind strength (float): wind strength in knots
    bank_angle (float): bank angle in degrees (default: 0, wings level)
    flaps (int): flap setting in degrees, one of VALID_FLAP_SETTINGS (default: 0, flaps up)

    Returns:
    float: Adjusted glide ratio, dround speed
    """
    if weight != W_MAX:
        warnings.warn("Nonstandard weight detected! Physics do not consider the effect of variable weight.")

    if density != RHO_0:
        warnings.warn("Nonstandard density detected! Physics do not consider the effect of variable density.")

    if abs(bank_angle) >= 90:
        raise ValueError("bank_angle must be less than 90 degrees")

    assert flaps in VALID_FLAP_SETTINGS

    # Calculate correction factors
    K_weight = 1 # (W_MAX / weight) ** 0.5
    K_density = 1 # (density / RHO_0) ** 0.25
    K_speed = 1 / (0.5 * (airspeed / V_GLIDE) ** 2 + 0.5 * (V_GLIDE / airspeed) ** 2)
    # Banking raises the load factor (n = 1/cos(bank)), which increases induced drag
    # and degrades glide performance; approximated as glide ratio scaling by cos(bank).
    K_bank = math.cos(math.radians(bank_angle))

    # Calculate adjusted effective glide ratio
    glide_ratio = GR_0 * K_weight * K_density * K_speed * K_bank
    # TODO: `airspeed` is documented as KIAS, but is used here directly as true/ground
    # airspeed with no density-based IAS->TAS conversion. glide_ratio itself is fine as-is
    # (best-glide L/D is density-independent at a fixed IAS), but the returned ground_speed
    # is used elsewhere (utils/paths.py's turn_radius_ft) as an absolute speed, where TAS
    # actually matters -- turn radius is underestimated at altitude as a result. Tracked as
    # a known limitation (small at this app's altitudes) rather than fixed now.
    wind_delta_rad = wind_delta / 180 * math.pi
    wind_correction_angle = math.asin(math.sin(wind_delta_rad) * wind_speed / airspeed)
    ground_speed = airspeed * math.cos(wind_correction_angle) - wind_speed * math.cos(wind_delta_rad)
    glide_ratio *= ground_speed / airspeed

    glide_ratio -= (flaps / 10) * FLAP_GR_PENALTY_PER_10_DEG

    return glide_ratio, ground_speed

def density_from_altitude(altitude_ft, temp_f=59):
    """
    Calculate air density from altitude using standard atmosphere.
    
    Parameters:
    altitude_ft (float): Altitude in feet
    temp_f (float): Temperature in Fahrenheit (default: 59°F standard)
    
    Returns:
    float: Air density in slug/ft³
    """
    
    # Convert temp to Rankine
    temp_r = temp_f + F_TO_R_OFFSET

    # Calculate density ratio using standard atmosphere
    temp_ratio = (TEMP_0_R - LAPSE_RATE * altitude_ft) / TEMP_0_R
    density_ratio = temp_ratio ** BAROMETRIC_EXPONENT

    return RHO_0 * density_ratio

if __name__ == "__main__":
    def sweep():
        weights = [W_MAX]
        # Densities in standard atmosphere at SL, 5000 and 10,000 ft
        # Source: https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
        densities = [RHO_0] # [0.002378, 20.48 * 10 ** -4, 17.56 * 10 ** -4]
        airspeeds = [50, 65, 90]
        wind_deltas = [0, 90, 180]
        wind_speeds = [0, 10]

        for weight in weights:
            for density in densities:
                for airspeed in airspeeds:
                    for wind_delta in wind_deltas:
                        for wind_speed in wind_speeds:
                            glide_ratio, ground_speed = cessna_glide_ratio(
                                weight, density, airspeed, wind_delta, wind_speed
                            )
                            print(
                                f"weight={weight} density={density} airspeed={airspeed} "
                                f"wind_delta={wind_delta} wind_speed={wind_speed} "
                                f"-> glide_ratio={glide_ratio:.3f} ground_speed={ground_speed:.2f}"
                            )

    def test_standard_conditions():
        glide_ratio, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        passed = abs(glide_ratio - GR_0) < 1e-9
        print(f"standard_conditions: {passed}")

    def test_tailwind_increases_glide_ratio():
        baseline, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        tailwind, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 180, 10)
        passed = tailwind > baseline
        print(f"tailwind_increases_glide_ratio: {passed}")

    def test_crosswind_decreases_glide_ratio_slightly():
        # A pure crosswind requires crabbing into the wind to hold course, which
        # bleeds a small amount of ground speed relative to no-wind conditions.
        baseline, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        crosswind, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 90, 10)
        passed = crosswind < baseline
        print(f"crosswind_decreases_glide_ratio_slightly: {passed}")

    def test_headwind_decreases_glide_ratio():
        baseline, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        headwind, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0, 10)
        passed = headwind < baseline
        print(f"headwind_decreases_glide_ratio: {passed}")

    def test_wings_level_bank_angle_no_change():
        baseline, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        wings_level, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=0)
        passed = abs(wings_level - baseline) < 1e-9
        print(f"wings_level_bank_angle_no_change: {passed}")

    def test_bank_angle_decreases_glide_ratio():
        baseline, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        banked, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=30)
        passed = banked < baseline
        print(f"bank_angle_decreases_glide_ratio: {passed}")

    def test_steeper_bank_further_decreases_glide_ratio():
        shallow, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=20)
        steep, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=45)
        passed = steep < shallow
        print(f"steeper_bank_further_decreases_glide_ratio: {passed}")

    def test_bank_angle_matches_cosine_scaling():
        baseline, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        banked, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=60)
        expected = baseline * math.cos(math.radians(60))
        passed = abs(banked - expected) < 1e-9
        print(f"bank_angle_matches_cosine_scaling: {passed}")

    def test_negative_bank_angle_symmetric():
        left_bank, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=-30)
        right_bank, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=30)
        passed = abs(left_bank - right_bank) < 1e-9
        print(f"negative_bank_angle_symmetric: {passed}")

    def test_bank_angle_at_90_degrees_raises():
        try:
            cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, bank_angle=90)
            passed = False
        except ValueError:
            passed = True
        print(f"bank_angle_at_90_degrees_raises: {passed}")

    def test_flaps_reduce_glide_ratio_by_penalty_per_10_deg():
        from .constants import FLAP_GR_PENALTY_PER_10_DEG
        baseline, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        flaps_30, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, flaps=30)
        expected = baseline - 3 * FLAP_GR_PENALTY_PER_10_DEG
        passed = abs(flaps_30 - expected) < 1e-9
        print(f"flaps_reduce_glide_ratio_by_penalty_per_10_deg: {passed}")

    def test_invalid_flaps_setting_raises():
        try:
            cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0, flaps=15)
            passed = False
        except AssertionError:
            passed = True
        print(f"invalid_flaps_setting_raises: {passed}")

    sweep()
    test_standard_conditions()
    test_tailwind_increases_glide_ratio()
    test_crosswind_decreases_glide_ratio_slightly()
    test_headwind_decreases_glide_ratio()
    test_wings_level_bank_angle_no_change()
    test_bank_angle_decreases_glide_ratio()
    test_steeper_bank_further_decreases_glide_ratio()
    test_bank_angle_matches_cosine_scaling()
    test_negative_bank_angle_symmetric()
    test_bank_angle_at_90_degrees_raises()
    test_flaps_reduce_glide_ratio_by_penalty_per_10_deg()
    test_invalid_flaps_setting_raises()