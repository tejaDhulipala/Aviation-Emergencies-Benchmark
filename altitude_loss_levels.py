import math
import warnings

from utils.basic_math import desired_heading
from utils.glide_ratio_and_density import cessna_glide_ratio
from utils.dubins import dubins_altitude_loss, dubins_maneuver_best_path, DUBINS_BANK_ANGLES_DEG
from utils.constants import GR_0, FT_PER_NM, OBSTACLE_CLEARANCE_FT, W_MAX, RHO_0, V_GLIDE
from plane import Plane, EnvironmentVariables, Instruction

_DEFAULT_TEMPERATURE_C = 15  # app-side placeholder, no physical effect on today's glide-ratio math


def altitude_loss_barebones(x_cur, y_cur, x_goal, y_goal, glide_ratio=GR_0) -> float:
    """Level 1: straight-line distance, fixed glide ratio, no wind."""
    distance_nm = math.dist((x_cur, y_cur), (x_goal, y_goal))
    return distance_nm * FT_PER_NM / glide_ratio


def altitude_loss_wind_aware(x_cur, y_cur, x_goal, y_goal, weight=W_MAX, density=RHO_0,
                              airspeed=V_GLIDE, wind_direction=0.0, wind_speed=0.0,
                              bank_angle=0.0, flaps=0) -> float:
    """Level 2: straight-line distance, glide ratio adjusted for wind, assuming the plane
    instantaneously snaps to (and holds) the direct bearing to the target for the whole glide."""
    distance_nm = math.dist((x_cur, y_cur), (x_goal, y_goal))
    heading = desired_heading(x_cur, y_cur, x_goal, y_goal)
    wind_delta = heading - wind_direction
    glide_ratio, _ = cessna_glide_ratio(weight, density, airspeed, wind_delta, wind_speed,
                                         bank_angle, flaps)
    return distance_nm * FT_PER_NM / glide_ratio


def _build_plane_for_full_physics(x_cur, y_cur, heading_cur, altitude_agl_ft, x_goal, y_goal,
                                   weight, airspeed, wind_direction, wind_speed, bank_angle, flaps):
    env_vars = EnvironmentVariables(wind_strength=wind_speed, wind_direction=wind_direction,
                                     temperature=_DEFAULT_TEMPERATURE_C)
    inst = Instruction(goal_x=x_goal, goal_y=y_goal, airspeed=airspeed, bank_angle=bank_angle, flaps=flaps)
    return Plane(pos_x=x_cur, pos_y=y_cur, alt=altitude_agl_ft, airspeed=airspeed, weight=weight,
                 heading=heading_cur, env_vars=env_vars, inst=inst)


def altitude_loss_full_physics(x_cur, y_cur, heading_cur, x_goal, y_goal, altitude_agl_ft, heading_goal,
                                weight=W_MAX, airspeed=V_GLIDE, wind_direction=0.0, wind_speed=0.0,
                                flaps=0, obstacle_clearance_ft=OBSTACLE_CLEARANCE_FT,
                                bank_angles_deg=DUBINS_BANK_ANGLES_DEG) -> tuple:
    """Level 3: Dubins-path (turn-straight-turn) full physics (utils.dubins.dubins_altitude_loss).

    Unlike the old single-turn model, this requires a target heading (heading_goal) --
    the pose the plane must arrive at the target on, e.g. a landing option's approach
    direction -- since a Dubins path is defined between two full poses, not just points.

    Tries every bank angle in bank_angles_deg (default 20/30/45 degrees) and every CSC
    (turn-straight-turn) combination at each, keeping the lowest-loss geometrically valid
    path; integrates altitude loss over both turn arcs the same way the old model
    integrated over its single turn arc.

    Returns (loss, reachable). `loss` is -1 whenever `reachable` is False -- either no
    bank angle in bank_angles_deg has a geometrically valid path (the turn is impossible),
    or the cheapest valid path costs more altitude than is available -- `reachable` is the
    authoritative signal, exactly as plane.py's Plane.follow_instruction treats the
    (loss, landing_info) pair from utils.paths.altitude_loss (never comparing the loss
    value itself).
    """
    with warnings.catch_warnings():
        # altitude_agl_ft != 0 means density != RHO_0, which triggers cessna_glide_ratio's
        # "Nonstandard density detected" warning (both during Plane construction and inside
        # utils.dubins's own internal calls) even though density has no effect on the math
        # today (K_density is hardcoded to 1). Harmless; just noisy.
        warnings.simplefilter("ignore", UserWarning)
        plane = _build_plane_for_full_physics(x_cur, y_cur, heading_cur, altitude_agl_ft, x_goal, y_goal,
                                               weight, airspeed, wind_direction, wind_speed,
                                               bank_angles_deg[0], flaps)
        loss, reachable = dubins_altitude_loss(plane, (x_goal, y_goal), heading_goal, obstacle_clearance_ft,
                                                flaps=flaps, bank_angles_deg=bank_angles_deg)
    return loss, reachable


def altitude_loss_full_physics_path(x_cur, y_cur, heading_cur, x_goal, y_goal, altitude_agl_ft, heading_goal,
                                     weight=W_MAX, airspeed=V_GLIDE, wind_direction=0.0, wind_speed=0.0,
                                     flaps=0, bank_angles_deg=DUBINS_BANK_ANGLES_DEG):
    """Like altitude_loss_full_physics, but returns the winning Dubins path's geometry
    (for visualization, e.g. scenario_gui.py's altitude-loss probe) instead of just the
    loss. Ignores altitude availability entirely -- unlike altitude_loss_full_physics,
    there's no reachable/-1 collapsing here, since a geometrically valid path is still
    worth drawing even if it costs more altitude than is available.

    Returns (path, bank_angle_deg) for the cheapest geometrically valid path across
    bank_angles_deg -- see utils.dubins.dubins_path_points to turn `path` into plottable
    (x, y) points -- or None if no bank angle has a geometrically valid path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plane = _build_plane_for_full_physics(x_cur, y_cur, heading_cur, altitude_agl_ft, x_goal, y_goal,
                                               weight, airspeed, wind_direction, wind_speed,
                                               bank_angles_deg[0], flaps)
        best = dubins_maneuver_best_path(plane, (x_goal, y_goal), heading_goal, flaps=flaps,
                                          bank_angles_deg=bank_angles_deg)
    if best is None:
        return None
    path, bank_angle_deg, _loss = best
    return path, bank_angle_deg


def compute_altitude_loss(level, x_cur, y_cur, heading_cur, x_goal, y_goal, altitude_agl_ft,
                           weight=W_MAX, airspeed=V_GLIDE, wind_direction=0.0, wind_speed=0.0,
                           bank_angle=30.0, flaps=0, heading_goal=None,
                           obstacle_clearance_ft=OBSTACLE_CLEARANCE_FT,
                           bank_angles_deg=DUBINS_BANK_ANGLES_DEG) -> tuple:
    """Shared interface over the three complexity levels. level in {1, 2, 3}.

    Levels 1/2 derive their own heading (they don't take heading_cur as an input to the
    physics, only level 3 does) and compute reachable = loss <= (altitude_agl_ft - obstacle_clearance_ft)
    themselves; level 3 delegates both loss and reachable to altitude_loss_full_physics.

    Level 3 requires heading_goal (the pose the plane must arrive at the target on -- a
    Dubins path is defined between two full poses, not just points) and picks its own bank
    angle from bank_angles_deg rather than using the shared `bank_angle` param, which levels
    1/2 also ignore already.
    """
    if level == 1:
        loss = altitude_loss_barebones(x_cur, y_cur, x_goal, y_goal)
        return loss, loss <= (altitude_agl_ft - obstacle_clearance_ft)
    elif level == 2:
        loss = altitude_loss_wind_aware(x_cur, y_cur, x_goal, y_goal, weight=weight, airspeed=airspeed,
                                         wind_direction=wind_direction, wind_speed=wind_speed,
                                         bank_angle=0.0, flaps=flaps)
        return loss, loss <= (altitude_agl_ft - obstacle_clearance_ft)
    elif level == 3:
        if heading_goal is None:
            raise ValueError("level 3 (Dubins full physics) requires heading_goal")
        return altitude_loss_full_physics(x_cur, y_cur, heading_cur, x_goal, y_goal, altitude_agl_ft,
                                           heading_goal, weight=weight, airspeed=airspeed,
                                           wind_direction=wind_direction, wind_speed=wind_speed, flaps=flaps,
                                           obstacle_clearance_ft=obstacle_clearance_ft,
                                           bank_angles_deg=bank_angles_deg)
    else:
        raise ValueError(f"level must be 1, 2, or 3, got {level}")


if __name__ == "__main__":
    def test_barebones_matches_manual_calc():
        loss = altitude_loss_barebones(0, 0, 3, 4)
        expected = 5 * FT_PER_NM / GR_0
        passed = abs(loss - expected) < 1e-9
        print(f"barebones_matches_manual_calc: {passed}")

    def test_wind_aware_headwind_increases_loss():
        no_wind = altitude_loss_wind_aware(0, 0, 0, 10, wind_speed=0)
        headwind = altitude_loss_wind_aware(0, 0, 0, 10, wind_direction=0, wind_speed=10)
        passed = headwind > no_wind
        print(f"wind_aware_headwind_increases_loss: {passed}")

    def test_wind_aware_tailwind_decreases_loss():
        no_wind = altitude_loss_wind_aware(0, 0, 0, 10, wind_speed=0)
        tailwind = altitude_loss_wind_aware(0, 0, 0, 10, wind_direction=180, wind_speed=10)
        passed = tailwind < no_wind
        print(f"wind_aware_tailwind_decreases_loss: {passed}")

    def test_wind_aware_matches_cessna_glide_ratio_directly():
        x_cur, y_cur, x_goal, y_goal = 0, 0, 5, 5
        wind_direction, wind_speed = 45, 12
        heading = desired_heading(x_cur, y_cur, x_goal, y_goal)
        expected_gr, _ = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, heading - wind_direction, wind_speed, 0.0, 0)
        expected = math.dist((x_cur, y_cur), (x_goal, y_goal)) * FT_PER_NM / expected_gr
        actual = altitude_loss_wind_aware(x_cur, y_cur, x_goal, y_goal, wind_direction=wind_direction, wind_speed=wind_speed)
        passed = abs(actual - expected) < 1e-9
        print(f"wind_aware_matches_cessna_glide_ratio_directly: {passed}")

    def test_full_physics_matches_dubins_altitude_loss_directly():
        x_cur, y_cur, heading_cur, x_goal, y_goal, alt, heading_goal = 0, 0, 90, 5, 0, 3000, 180
        loss, reachable = altitude_loss_full_physics(x_cur, y_cur, heading_cur, x_goal, y_goal, alt, heading_goal)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plane = _build_plane_for_full_physics(x_cur, y_cur, heading_cur, alt, x_goal, y_goal,
                                                   W_MAX, V_GLIDE, 0.0, 0.0, DUBINS_BANK_ANGLES_DEG[0], 0)
            expected_loss, expected_reachable = dubins_altitude_loss(plane, (x_goal, y_goal), heading_goal,
                                                                       OBSTACLE_CLEARANCE_FT, flaps=0)
        passed = loss == expected_loss and reachable == expected_reachable
        print(f"full_physics_matches_dubins_altitude_loss_directly: {passed}")

    def test_full_physics_requires_two_arc_turn_for_reversal():
        # Arriving with a heading opposite the outbound bearing forces a genuine
        # two-arc Dubins maneuver (no single turn onto the bearing suffices), which is
        # exactly the case the old single-turn "full physics" model couldn't represent.
        loss, reachable = altitude_loss_full_physics(0, 0, 0, 0, 5, 6000, heading_goal=180)
        passed = reachable and loss > 0
        print(f"full_physics_requires_two_arc_turn_for_reversal: {passed} (loss={loss})")

    def test_full_physics_impossible_geometry_returns_minus_one():
        loss, reachable = altitude_loss_full_physics(0, 0, 0, 0, 0, 3000, heading_goal=0)
        passed = loss == -1 and not reachable
        print(f"full_physics_impossible_geometry_returns_minus_one: {passed}")

    def test_reachable_threshold_level1():
        loss = altitude_loss_barebones(0, 0, 1, 0)
        just_enough = loss + OBSTACLE_CLEARANCE_FT
        just_short = loss + OBSTACLE_CLEARANCE_FT - 1
        _, reachable_enough = compute_altitude_loss(1, 0, 0, 0, 1, 0, just_enough)
        _, reachable_short = compute_altitude_loss(1, 0, 0, 0, 1, 0, just_short)
        passed = reachable_enough and not reachable_short
        print(f"reachable_threshold_level1: {passed}")

    def test_reachable_threshold_level2():
        loss = altitude_loss_wind_aware(0, 0, 1, 0)
        just_enough = loss + OBSTACLE_CLEARANCE_FT
        just_short = loss + OBSTACLE_CLEARANCE_FT - 1
        _, reachable_enough = compute_altitude_loss(2, 0, 0, 0, 1, 0, just_enough)
        _, reachable_short = compute_altitude_loss(2, 0, 0, 0, 1, 0, just_short)
        passed = reachable_enough and not reachable_short
        print(f"reachable_threshold_level2: {passed}")

    def test_level3_requires_heading_goal():
        try:
            compute_altitude_loss(3, 0, 0, 90, 1, 0, 5000)
            passed = False
        except ValueError:
            passed = True
        print(f"level3_requires_heading_goal: {passed}")

    def test_reachable_threshold_level3():
        _, reachable_high = compute_altitude_loss(3, 0, 0, 90, 1, 0, 5000, heading_goal=180)
        _, reachable_low = compute_altitude_loss(3, 0, 0, 90, 1, 0, 55, heading_goal=180)
        passed = reachable_high and not reachable_low
        print(f"reachable_threshold_level3: {passed}")

    def test_dispatch_matches_direct_calls():
        args = dict(x_cur=0, y_cur=0, heading_cur=45, x_goal=4, y_goal=3, altitude_agl_ft=2000,
                    wind_direction=90, wind_speed=8)
        loss1, r1 = compute_altitude_loss(1, **args)
        loss2, r2 = compute_altitude_loss(2, **args)
        loss3, r3 = compute_altitude_loss(3, heading_goal=180, **args)
        direct1 = altitude_loss_barebones(0, 0, 4, 3)
        direct2 = altitude_loss_wind_aware(0, 0, 4, 3, wind_direction=90, wind_speed=8, bank_angle=0.0)
        direct3, r3_expected = altitude_loss_full_physics(0, 0, 45, 4, 3, 2000, 180, wind_direction=90, wind_speed=8)
        passed = (loss1 == direct1 and loss2 == direct2 and loss3 == direct3 and r3 == r3_expected)
        print(f"dispatch_matches_direct_calls: {passed}")

    def vibes_check():
        print("--- vibes check: level 1 vs 2 vs 3, target requiring a turn ---")
        for wind_direction, wind_speed in [(0, 0), (270, 10), (90, 10)]:
            l1, r1 = compute_altitude_loss(1, 0, 0, 0, 3, 3, 4000, wind_direction=wind_direction, wind_speed=wind_speed)
            l2, r2 = compute_altitude_loss(2, 0, 0, 0, 3, 3, 4000, wind_direction=wind_direction, wind_speed=wind_speed)
            l3, r3 = compute_altitude_loss(3, 0, 0, 0, 3, 3, 4000, wind_direction=wind_direction, wind_speed=wind_speed,
                                            heading_goal=90)
            print(f"wind={wind_direction}@{wind_speed}: level1={l1:.1f}({r1}) level2={l2:.1f}({r2}) level3={l3:.1f}({r3})")

    test_barebones_matches_manual_calc()
    test_wind_aware_headwind_increases_loss()
    test_wind_aware_tailwind_decreases_loss()
    test_wind_aware_matches_cessna_glide_ratio_directly()
    test_full_physics_matches_dubins_altitude_loss_directly()
    test_full_physics_requires_two_arc_turn_for_reversal()
    test_full_physics_impossible_geometry_returns_minus_one()
    test_level3_requires_heading_goal()
    test_reachable_threshold_level1()
    test_reachable_threshold_level2()
    test_reachable_threshold_level3()
    test_dispatch_matches_direct_calls()
    vibes_check()
