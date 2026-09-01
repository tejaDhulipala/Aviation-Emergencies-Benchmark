import math

from .basic_math import desired_heading
from .glide_ratio_and_density import cessna_glide_ratio
from .paths import (
    compass_direction,
    _compass_rotate,
    turn_center,
    turn_radius_ft,
    integrate_turn_altitude_loss,
    TURN_INTEGRATION_STEPS,
)
from .constants import FT_PER_NM

# Bank angles tried for each Dubins altitude-loss estimate; the lowest-loss
# geometrically valid path across these wins (see dubins_altitude_loss).
DUBINS_BANK_ANGLES_DEG = (20, 30, 45)

# (turn_sign1, turn_sign2) for the four CSC (turn-straight-turn) path types:
# RSR, RSL, LSR, LSL (turn_sign +1 = right, -1 = left, matching utils.paths.turn_center).
_CSC_TURN_SIGN_COMBOS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _vec_scale(a, k):
    return (a[0] * k, a[1] * k)


def _tangent_solution(center1, center2, radius_nm, turn_sign1, turn_sign2):
    """Solve for the CSC tangent line between two turn circles of equal radius.

    Returns (T1, T2, straight_heading_deg, straight_length_nm): T1/T2 are the points
    where the aircraft leaves circle1 / joins circle2, straight_heading_deg is the
    heading flown between them, and straight_length_nm is the straight-segment length.

    Returns None if no solution exists: for opposite-turn combinations (RSL/LSR) an
    internal tangent only exists when the circle centers are at least 2*radius_nm
    apart; for same-turn combinations (RSR/LSL) only the degenerate case of
    coincident centers has no solution.
    """
    v = _vec_sub(center2, center1)
    dist = math.hypot(*v)
    if dist < 1e-9:
        return None

    theta_v = desired_heading(0, 0, v[0], v[1])
    # Perpendicular (to the straight segment) offset between the two circles'
    # radius vectors, in terms of turn_sign; see derivation notes in the module
    # docstring-equivalent PR description. Zero for same-turn combos (external
    # tangent, parallel to the center line); +/-2*radius_nm for opposite-turn
    # combos (internal tangent), which is only solvable if the circles are far
    # enough apart.
    k = -radius_nm * (turn_sign2 - turn_sign1)
    if abs(k) > dist:
        return None

    acos_arg = max(-1.0, min(1.0, -k / dist))
    base_deg = math.degrees(math.acos(acos_arg))

    for sign in (1, -1):
        straight_heading = theta_v - 90 - sign * base_deg
        straight_length_nm = dist * math.cos(math.radians(theta_v - straight_heading))
        if straight_length_nm > 1e-9:
            u = compass_direction(straight_heading)
            t1 = _vec_sub(center1, _vec_scale(_compass_rotate(u, 90 * turn_sign1), radius_nm))
            t2 = _vec_add(t1, _vec_scale(u, straight_length_nm))
            return t1, t2, straight_heading % 360, straight_length_nm

    return None


def _csc_path(cur_pos, heading_cur, target_pos, heading_goal, radius_nm, turn_sign1, turn_sign2):
    """Build one CSC (turn-straight-turn) Dubins path variant from (cur_pos, heading_cur)
    to (target_pos, heading_goal) at a fixed turn radius. Returns a dict describing the
    two arcs and the straight segment, or None if this turn-sign combination has no
    geometric solution (see _tangent_solution)."""
    center1 = turn_center(cur_pos, heading_cur, radius_nm, turn_sign1)
    center2 = turn_center(target_pos, heading_goal, radius_nm, turn_sign2)

    solution = _tangent_solution(center1, center2, radius_nm, turn_sign1, turn_sign2)
    if solution is None:
        return None
    t1, t2, straight_heading, straight_length_nm = solution

    theta1 = ((straight_heading - heading_cur) * turn_sign1) % 360
    theta2 = ((heading_goal - straight_heading) * turn_sign2) % 360

    return {
        "radius_nm": radius_nm,
        "turn_sign1": turn_sign1,
        "theta1": theta1,
        "t1": t1,
        "straight_heading": straight_heading,
        "straight_length_nm": straight_length_nm,
        "t2": t2,
        "turn_sign2": turn_sign2,
        "theta2": theta2,
    }


def _csc_path_altitude_loss(path, heading_cur, glide_ratio_fn, bank_angle_deg, n_integration_steps):
    """Altitude loss (ft) for one CSC path, integrating over both turn arcs the same
    way utils.paths.integrate_turn_altitude_loss integrates the single turn arc in the
    non-Dubins full-physics model, plus a level (0-bank) straight-line glide in between."""

    def turn_glide_ratio_fn(heading_deg):
        return glide_ratio_fn(heading_deg, bank_angle_deg)

    loss_arc1 = integrate_turn_altitude_loss(turn_glide_ratio_fn, heading_cur, path["turn_sign1"],
                                              path["radius_nm"], path["theta1"], n_integration_steps)

    glide_ratio_straight, _ = glide_ratio_fn(path["straight_heading"], 0.0)
    loss_straight = path["straight_length_nm"] * FT_PER_NM / glide_ratio_straight

    loss_arc2 = integrate_turn_altitude_loss(turn_glide_ratio_fn, path["straight_heading"], path["turn_sign2"],
                                              path["radius_nm"], path["theta2"], n_integration_steps)

    return loss_arc1 + loss_straight + loss_arc2


def _best_csc_path_at_bank_angle(cur_pos, heading_cur, target_pos, heading_goal, ground_speed_kt,
                                  bank_angle_deg, glide_ratio_fn, n_integration_steps):
    """Lowest-loss CSC path (and its loss, in ft) among the four turn-sign combinations at
    a given bank angle, or (None, None) if none of the four is geometrically valid at this
    bank angle's turn radius."""
    radius_nm = turn_radius_ft(ground_speed_kt, bank_angle_deg) / FT_PER_NM
    best_path = None
    best_loss = None
    for turn_sign1, turn_sign2 in _CSC_TURN_SIGN_COMBOS:
        path = _csc_path(cur_pos, heading_cur, target_pos, heading_goal, radius_nm, turn_sign1, turn_sign2)
        if path is None:
            continue
        loss = _csc_path_altitude_loss(path, heading_cur, glide_ratio_fn, bank_angle_deg, n_integration_steps)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_path = path
    return best_path, best_loss


def dubins_maneuver_best_path(plane, target_pos, heading_goal, flaps=0,
                               bank_angles_deg=DUBINS_BANK_ANGLES_DEG,
                               n_integration_steps=TURN_INTEGRATION_STEPS):
    """Cheapest Dubins (turn-straight-turn) path for `plane` to glide from its current
    pose to (target_pos, heading_goal), trying every bank angle in bank_angles_deg and
    every CSC turn-sign combination at each.

    Returns (path, bank_angle_deg, loss_ft) for the winner, or None if no (bank angle,
    turn-sign combination) pair has a geometrically valid path -- i.e. the turn is
    impossible at every tried bank angle. `path` is the dict built by _csc_path
    (radius_nm, turn_sign1/2, theta1/2, t1/t2, straight_heading/length) -- see
    dubins_path_points to turn it into plottable (x, y) points.

    `plane` just needs pos_x/pos_y/heading/ground_speed/weight/density/airspeed and
    environment_variables.wind_direction/wind_strength -- duck-typed, same as
    utils.paths._maneuver_altitude_loss. Pure function: does not mutate `plane`.
    """
    cur_pos = (plane.pos_x, plane.pos_y)
    wind_direction = plane.environment_variables.wind_direction
    wind_strength = plane.environment_variables.wind_strength

    def glide_ratio_fn(heading_deg, bank_angle_deg):
        wind_delta = heading_deg - wind_direction
        return cessna_glide_ratio(plane.weight, plane.density, plane.airspeed, wind_delta,
                                   wind_strength, bank_angle=bank_angle_deg, flaps=flaps)

    best = None  # (path, bank_angle_deg, loss)
    for bank_angle_deg in bank_angles_deg:
        path, loss = _best_csc_path_at_bank_angle(cur_pos, plane.heading, target_pos, heading_goal,
                                                    plane.ground_speed, bank_angle_deg, glide_ratio_fn,
                                                    n_integration_steps)
        if path is not None and (best is None or loss < best[2]):
            best = (path, bank_angle_deg, loss)
    return best


def dubins_maneuver_altitude_loss(plane, target_pos, heading_goal, flaps=0,
                                   bank_angles_deg=DUBINS_BANK_ANGLES_DEG,
                                   n_integration_steps=TURN_INTEGRATION_STEPS):
    """Lowest Dubins (turn-straight-turn) altitude loss (ft) for `plane` to glide from
    its current pose to (target_pos, heading_goal); see dubins_maneuver_best_path.

    Returns None if no (bank angle, turn-sign combination) pair has a geometrically
    valid path -- i.e. the turn is impossible at every tried bank angle.
    """
    best = dubins_maneuver_best_path(plane, target_pos, heading_goal, flaps=flaps,
                                      bank_angles_deg=bank_angles_deg,
                                      n_integration_steps=n_integration_steps)
    return None if best is None else best[2]


def dubins_path_points(cur_pos, heading_cur, target_pos, heading_goal, path, points_per_arc=24):
    """Sample (x, y) nm points tracing a full CSC path (arc1 -> straight -> arc2) as
    built by _csc_path / returned by dubins_maneuver_best_path, for plotting."""
    from .paths import position_on_circle

    center1 = turn_center(cur_pos, heading_cur, path["radius_nm"], path["turn_sign1"])
    center2 = turn_center(target_pos, heading_goal, path["radius_nm"], path["turn_sign2"])

    points = [position_on_circle(center1, cur_pos, path["theta1"] * i / points_per_arc, path["turn_sign1"])
              for i in range(points_per_arc + 1)]
    points.append(path["t2"])
    points.extend(position_on_circle(center2, path["t2"], path["theta2"] * i / points_per_arc, path["turn_sign2"])
                  for i in range(1, points_per_arc + 1))
    return points


def dubins_altitude_loss(plane, target_pos, heading_goal, obstacle_clearance_ft, flaps=0,
                          bank_angles_deg=DUBINS_BANK_ANGLES_DEG,
                          n_integration_steps=TURN_INTEGRATION_STEPS):
    """Dubins-path analogue of utils.paths.altitude_loss.

    Returns (loss, reachable):
      - (-1, False) if no bank angle in bank_angles_deg has a geometrically valid
        Dubins path to (target_pos, heading_goal) -- the turn is impossible, so there's
        no loss figure to report at all.
      - (loss_ft, False) if a geometrically valid path exists but its cost exceeds what's
        available (plane.alt - obstacle_clearance_ft) -- loss_ft is still the real cost of
        the cheapest valid path, just more than the plane has to spend.
      - (loss_ft, True) otherwise, loss_ft being the cheapest valid path found.

    `reachable` is always the authoritative feasibility signal; `loss_ft` is only ever the
    -1 placeholder when no path exists to measure in the first place.

    Pure function: does not mutate `plane`.
    """
    best_loss = dubins_maneuver_altitude_loss(plane, target_pos, heading_goal, flaps=flaps,
                                               bank_angles_deg=bank_angles_deg,
                                               n_integration_steps=n_integration_steps)
    if best_loss is None:
        return -1, False

    available = plane.alt - obstacle_clearance_ft
    return best_loss, best_loss <= available


if __name__ == "__main__":
    from .basic_math import signed_heading_diff

    def test_tangent_solution_external_matches_center_line():
        # Same-turn (external) tangent: straight segment should be parallel to (and as
        # long as) the line connecting the two centers, offset by exactly radius_nm.
        radius_nm = 1.0
        center1, center2 = (0.0, 0.0), (0.0, 10.0)
        for turn_sign in (1, -1):
            solution = _tangent_solution(center1, center2, radius_nm, turn_sign, turn_sign)
            passed = solution is not None
            if passed:
                t1, t2, heading, length = solution
                passed = (abs(length - 10.0) < 1e-9 and
                          abs(math.dist(t1, center1) - radius_nm) < 1e-9 and
                          abs(math.dist(t2, center2) - radius_nm) < 1e-9)
            print(f"tangent_solution_external_matches_center_line (sign={turn_sign}): {passed}")

    def test_tangent_solution_internal_requires_separation():
        radius_nm = 1.0
        too_close = _tangent_solution((0.0, 0.0), (0.0, 1.5), radius_nm, 1, -1)
        far_enough = _tangent_solution((0.0, 0.0), (0.0, 3.0), radius_nm, 1, -1)
        passed = too_close is None and far_enough is not None
        print(f"tangent_solution_internal_requires_separation: {passed}")

    def test_tangent_points_lie_on_their_circles():
        radius_nm = 1.3
        center1, center2 = (0.0, 0.0), (2.0, 5.0)
        passed = True
        for turn_sign1, turn_sign2 in _CSC_TURN_SIGN_COMBOS:
            solution = _tangent_solution(center1, center2, radius_nm, turn_sign1, turn_sign2)
            if solution is None:
                continue
            t1, t2, heading, length = solution
            passed = passed and abs(math.dist(t1, center1) - radius_nm) < 1e-6
            passed = passed and abs(math.dist(t2, center2) - radius_nm) < 1e-6
            # T2 must be reachable from T1 flying `heading` for `length` nm.
            dx, dy = compass_direction(heading)
            reconstructed_t2 = (t1[0] + length * dx, t1[1] + length * dy)
            passed = passed and math.dist(reconstructed_t2, t2) < 1e-6
        print(f"tangent_points_lie_on_their_circles: {passed}")

    def test_csc_path_arcs_hit_correct_headings():
        # Building on the tangent-point checks: the arc sweeps recorded in the path
        # dict should actually rotate heading_cur/straight_heading into straight_heading
        # /heading_goal, and the arc endpoints should land exactly on T1/T2.
        radius_nm = 1.0
        cur_pos, heading_cur = (0.0, 0.0), 10.0
        target_pos, heading_goal = (3.0, 6.0), 200.0
        passed = True
        for turn_sign1, turn_sign2 in _CSC_TURN_SIGN_COMBOS:
            path = _csc_path(cur_pos, heading_cur, target_pos, heading_goal, radius_nm, turn_sign1, turn_sign2)
            if path is None:
                continue
            from .paths import position_on_circle, heading_after_turn
            center1 = turn_center(cur_pos, heading_cur, radius_nm, turn_sign1)
            center2 = turn_center(target_pos, heading_goal, radius_nm, turn_sign2)
            arc1_end = position_on_circle(center1, cur_pos, path["theta1"], turn_sign1)
            arc1_end_heading = heading_after_turn(heading_cur, path["theta1"], turn_sign1)
            arc2_start = path["t2"]
            arc2_end = position_on_circle(center2, arc2_start, path["theta2"], turn_sign2)
            arc2_end_heading = heading_after_turn(path["straight_heading"], path["theta2"], turn_sign2)
            passed = passed and math.dist(arc1_end, path["t1"]) < 1e-6
            passed = passed and abs(signed_heading_diff(arc1_end_heading, path["straight_heading"])) < 1e-6
            passed = passed and math.dist(arc2_end, target_pos) < 1e-6
            passed = passed and abs(signed_heading_diff(arc2_end_heading, heading_goal)) < 1e-6
        print(f"csc_path_arcs_hit_correct_headings: {passed}")

    class _FakeEnvironmentVariables:
        def __init__(self, wind_direction, wind_strength):
            self.wind_direction = wind_direction
            self.wind_strength = wind_strength

    class _FakePlane:
        def __init__(self, pos_x, pos_y, heading, ground_speed, weight, density, airspeed,
                     wind_direction, wind_strength, alt=1_000_000):
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.heading = heading
            self.ground_speed = ground_speed
            self.weight = weight
            self.density = density
            self.airspeed = airspeed
            self.alt = alt
            self.environment_variables = _FakeEnvironmentVariables(wind_direction, wind_strength)

    def _make_fake_plane(heading=0.0, wind_direction=0.0, wind_strength=0.0, alt=1_000_000):
        from .constants import W_MAX, RHO_0, V_GLIDE
        _, ground_speed = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, heading - wind_direction, wind_strength)
        return _FakePlane(pos_x=0, pos_y=0, heading=heading, ground_speed=ground_speed, weight=W_MAX,
                           density=RHO_0, airspeed=V_GLIDE, wind_direction=wind_direction,
                           wind_strength=wind_strength, alt=alt)

    def test_dubins_altitude_loss_reachable_basic():
        plane = _make_fake_plane(heading=0.0)
        loss, reachable = dubins_altitude_loss(plane, (0.0, 10.0), 0.0, obstacle_clearance_ft=0)
        passed = reachable and loss > 0
        print(f"dubins_altitude_loss_reachable_basic: {passed} (loss={loss})")

    def test_dubins_altitude_loss_does_not_mutate_plane():
        plane = _make_fake_plane(heading=0.0)
        before = (plane.pos_x, plane.pos_y, plane.heading)
        dubins_altitude_loss(plane, (3.0, 4.0), 90.0, obstacle_clearance_ft=0)
        after = (plane.pos_x, plane.pos_y, plane.heading)
        passed = before == after
        print(f"dubins_altitude_loss_does_not_mutate_plane: {passed}")

    def test_dubins_altitude_loss_unreachable_when_short_on_altitude():
        # loss should still be the real (over-budget) cost, not the -1 placeholder --
        # that's reserved for when no geometrically valid path exists at all.
        plane = _make_fake_plane(heading=0.0, alt=10)
        loss, reachable = dubins_altitude_loss(plane, (0.0, 10.0), 0.0, obstacle_clearance_ft=0)
        passed = loss > 0 and not reachable
        print(f"dubins_altitude_loss_unreachable_when_short_on_altitude: {passed} (loss={loss})")

    def test_dubins_altitude_loss_impossible_geometry_returns_minus_one():
        # Goal pose identical to the start pose: every CSC combination degenerates to a
        # zero-length (or undefined) straight segment, so none of the four turn-sign
        # combinations -- at any of the three tried bank angles -- has a valid path.
        plane = _make_fake_plane(heading=0.0)
        loss, reachable = dubins_altitude_loss(plane, (0.0, 0.0), 0.0, obstacle_clearance_ft=0)
        passed = loss == -1 and not reachable
        print(f"dubins_altitude_loss_impossible_geometry_returns_minus_one: {passed}")

    def test_dubins_altitude_loss_wind_headwind_increases_loss():
        plane_no_wind = _make_fake_plane(heading=0.0)
        plane_headwind = _make_fake_plane(heading=0.0, wind_direction=0.0, wind_strength=10)
        loss_no_wind, _ = dubins_altitude_loss(plane_no_wind, (0.0, 10.0), 0.0, obstacle_clearance_ft=0)
        loss_headwind, _ = dubins_altitude_loss(plane_headwind, (0.0, 10.0), 0.0, obstacle_clearance_ft=0)
        passed = loss_headwind > loss_no_wind
        print(f"dubins_altitude_loss_wind_headwind_increases_loss: {passed}")

    def test_dubins_path_points_endpoints_match_pose():
        cur_pos, heading_cur = (0.0, 0.0), 10.0
        target_pos, heading_goal = (3.0, 6.0), 200.0
        radius_nm = 1.0
        passed = True
        for turn_sign1, turn_sign2 in _CSC_TURN_SIGN_COMBOS:
            path = _csc_path(cur_pos, heading_cur, target_pos, heading_goal, radius_nm, turn_sign1, turn_sign2)
            if path is None:
                continue
            points = dubins_path_points(cur_pos, heading_cur, target_pos, heading_goal, path)
            passed = passed and math.dist(points[0], cur_pos) < 1e-6
            passed = passed and math.dist(points[-1], target_pos) < 1e-6
        print(f"dubins_path_points_endpoints_match_pose: {passed}")

    def test_dubins_maneuver_best_path_matches_altitude_loss():
        plane = _make_fake_plane(heading=0.0)
        target_pos, heading_goal = (3.0, 4.0), 90.0
        best = dubins_maneuver_best_path(plane, target_pos, heading_goal)
        loss = dubins_maneuver_altitude_loss(plane, target_pos, heading_goal)
        passed = best is not None and abs(best[2] - loss) < 1e-9
        print(f"dubins_maneuver_best_path_matches_altitude_loss: {passed}")

    test_tangent_solution_external_matches_center_line()
    test_tangent_solution_internal_requires_separation()
    test_tangent_points_lie_on_their_circles()
    test_csc_path_arcs_hit_correct_headings()
    test_dubins_path_points_endpoints_match_pose()
    test_dubins_maneuver_best_path_matches_altitude_loss()
    test_dubins_altitude_loss_reachable_basic()
    test_dubins_altitude_loss_does_not_mutate_plane()
    test_dubins_altitude_loss_unreachable_when_short_on_altitude()
    test_dubins_altitude_loss_impossible_geometry_returns_minus_one()
    test_dubins_altitude_loss_wind_headwind_increases_loss()
