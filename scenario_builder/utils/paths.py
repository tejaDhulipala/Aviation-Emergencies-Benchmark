import math

from .constants import G_NM_HR_2, FT_PER_NM, MIN_BANK_ANGLE_DEG, MAX_BANK_ANGLE_DEG
from .basic_math import desired_heading, signed_heading_diff
from .glide_ratio_and_density import cessna_glide_ratio

# Tuning parameters for altitude_loss's glide-path approximation (not physical constants,
# so they live here, next to the function that uses them, rather than in constants.py).
CONE_ALPHA_DEG = 15          # heading-error cone within which the small-alpha straight-line assumption applies
BANK_ANGLE_FRACTION = 0.5    # fraction of a heading-error angle assumed as bank for shallow corrections
TURN_INTEGRATION_STEPS = 50  # steps for numerically integrating altitude loss over a turn


def turn_radius_ft(ground_speed_kt, bank_angle_deg):
    """Standard rate-of-turn radius (ft) for a given ground speed (kt) and bank angle (deg)."""
    return ground_speed_kt ** 2 / (G_NM_HR_2 * math.tan(math.radians(bank_angle_deg))) * FT_PER_NM


def compass_direction(heading_deg):
    """Unit vector (dx, dy) for a compass heading (0=north/+y, 90=east/+x)."""
    rad = math.radians(heading_deg)
    return math.sin(rad), math.cos(rad)


def _compass_rotate(vector, angle_deg):
    """Rotate (x, y) by angle_deg in the same rotational sense as increasing compass heading."""
    x, y = vector
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return x * cos_a + y * sin_a, -x * sin_a + y * cos_a


def turn_center(pos, heading_deg, radius_nm, turn_sign):
    """Center of the turn circle: perpendicular to heading, radius_nm away,
    on the right (turn_sign=+1) or left (turn_sign=-1) side."""
    dx, dy = compass_direction(heading_deg + 90 * turn_sign)
    return pos[0] + radius_nm * dx, pos[1] + radius_nm * dy


def position_on_circle(center, start_pos, swept_deg, turn_sign):
    """Position after sweeping swept_deg degrees of turn (turn_sign=+1 right, -1 left)
    around center, starting from start_pos."""
    v0 = (start_pos[0] - center[0], start_pos[1] - center[1])
    vx, vy = _compass_rotate(v0, turn_sign * swept_deg)
    return center[0] + vx, center[1] + vy


def heading_after_turn(start_heading_deg, swept_deg, turn_sign):
    return (start_heading_deg + turn_sign * swept_deg) % 360


def ray_line_intersection(point, heading_deg, line_point, line_heading_deg):
    """Intersection of the ray from `point` (direction heading_deg) with the infinite
    line through `line_point` (direction line_heading_deg). Returns (x, y), or raises
    ValueError if the two directions are parallel."""
    d1x, d1y = compass_direction(heading_deg)
    d2x, d2y = compass_direction(line_heading_deg)
    det = d2x * d1y - d1x * d2y
    if abs(det) < 1e-12:
        raise ValueError("heading_deg and line_heading_deg are parallel; no unique intersection")
    dx = line_point[0] - point[0]
    dy = line_point[1] - point[1]
    t = (d2x * dy - dx * d2y) / det
    return point[0] + t * d1x, point[1] + t * d1y


def integrate_turn_altitude_loss(glide_ratio_fn, start_heading_deg, turn_sign, radius_nm, swept_deg, n_steps):
    """Numerically integrate altitude loss (ft) over a turn, since glide ratio varies
    continuously with heading (wind angle changes as the aircraft turns).
    glide_ratio_fn(heading_deg) -> (glide_ratio, ground_speed), e.g. a partial application
    of cessna_glide_ratio with the aircraft's weight/density/airspeed/wind already bound.
    """
    dtheta = swept_deg / n_steps
    loss = 0.0
    for i in range(n_steps):
        theta_mid = (i + 0.5) * dtheta
        heading_mid = (start_heading_deg + turn_sign * theta_mid) % 360
        glide_ratio, _ = glide_ratio_fn(heading_mid)
        arc_length_nm = radius_nm * math.radians(dtheta)
        loss += arc_length_nm * FT_PER_NM / glide_ratio
    return loss


def _maneuver_altitude_loss(plane, target_pos, flaps=0, alpha_deg=CONE_ALPHA_DEG,
                             bank_fraction=BANK_ANGLE_FRACTION, n_integration_steps=TURN_INTEGRATION_STEPS):
    """Estimate altitude loss (ft) for `plane` to glide from its current position to
    target_pos = (x, y) in nm, assuming it turns onto the target's bearing.

    If the target is already within alpha_deg of the current heading, this is a single
    straight glide with an assumed shallow correction bank. Otherwise, the plane turns
    (at its instruction's actual bank angle) past the direct bearing by exactly alpha_deg,
    flies straight until crossing the line from the current position to the target, then
    makes a final small-alpha corrected glide to the target -- see the design notes for
    the angle-chase that makes the alpha_deg overshoot exact rather than approximate.

    That alpha-overshoot construction assumes the intersection point falls between the
    current position and the target; for a target close enough (relative to the turn
    radius) that doesn't hold, and two fallbacks apply instead:
      - If the target lies inside the turn circle, no coordinated turn at this bank angle
        can ever roll out pointed at it -- raises ValueError.
      - Otherwise, falls back to turning only until the heading matches the direct bearing
        (no alpha overshoot), then gliding straight from wherever that leaves the plane to
        the target, with an assumed bank proportional to *whatever* residual heading error
        remains at that point -- uncapped by alpha_deg, unlike the cone case, since this
        residual is a byproduct of the actual turn geometry rather than a design parameter.

    flaps reduce glide ratio via cessna_glide_ratio's own flaps penalty, applied at every
    glide_ratio_at call below.

    `plane` just needs pos_x/pos_y/heading/ground_speed/weight/density/airspeed,
    environment_variables.wind_direction/wind_strength, and instruction.bank_angle --
    duck-typed, no import of the Plane class needed here. instruction.bank_angle is
    validated here (the root entry point for that value from outside code) rather than
    in the utility functions it's passed on to.

    Pure function: does not mutate `plane`.
    """
    if not (MIN_BANK_ANGLE_DEG <= plane.instruction.bank_angle <= MAX_BANK_ANGLE_DEG):
        raise ValueError(
            f"instruction.bank_angle ({plane.instruction.bank_angle}) must be between "
            f"{MIN_BANK_ANGLE_DEG} and {MAX_BANK_ANGLE_DEG} degrees"
        )

    cur_pos = (plane.pos_x, plane.pos_y)
    b0 = desired_heading(cur_pos[0], cur_pos[1], target_pos[0], target_pos[1])
    delta = signed_heading_diff(plane.heading, b0)
    direct_distance = math.dist(cur_pos, target_pos)

    wind_direction = plane.environment_variables.wind_direction
    wind_strength = plane.environment_variables.wind_strength

    def glide_ratio_at(heading_deg, bank_angle_deg=0.0):
        wind_delta = heading_deg - wind_direction
        return cessna_glide_ratio(plane.weight, plane.density, plane.airspeed, wind_delta,
                                   wind_strength, bank_angle=bank_angle_deg, flaps=flaps)

    if abs(delta) <= alpha_deg:
        assumed_bank = bank_fraction * abs(delta)
        glide_ratio, _ = glide_ratio_at(b0, assumed_bank)
        return direct_distance * FT_PER_NM / glide_ratio

    turn_sign = 1 if delta >= 0 else -1
    radius_nm = turn_radius_ft(plane.ground_speed, plane.instruction.bank_angle) / FT_PER_NM
    center = turn_center(cur_pos, plane.heading, radius_nm, turn_sign)

    def turn_glide_ratio_fn(heading_deg):
        return glide_ratio_at(heading_deg, plane.instruction.bank_angle)

    if math.dist(target_pos, center) < radius_nm:
        raise ValueError(
            "target is inside the turn circle: no coordinated turn at this bank angle "
            "can roll out pointed at it"
        )

    theta_exit = abs(delta) + alpha_deg
    exit_pos = position_on_circle(center, cur_pos, theta_exit, turn_sign)
    h_exit = heading_after_turn(plane.heading, theta_exit, turn_sign)
    intersection = ray_line_intersection(exit_pos, h_exit, cur_pos, b0)

    # Project the intersection onto the cur->target line; for close/tight-angle targets
    # the alpha-overshoot construction can place it past the target, which would mean
    # flying backward to reach it -- fall back to the residual-correction construction instead.
    bearing_unit = compass_direction(b0)
    t_intersection = ((intersection[0] - cur_pos[0]) * bearing_unit[0] +
                       (intersection[1] - cur_pos[1]) * bearing_unit[1])

    if not (0 <= t_intersection <= direct_distance):
        theta_exit_fb = abs(delta)
        exit_pos_fb = position_on_circle(center, cur_pos, theta_exit_fb, turn_sign)
        h_exit_fb = heading_after_turn(plane.heading, theta_exit_fb, turn_sign)  # == b0

        loss_turn_fb = integrate_turn_altitude_loss(turn_glide_ratio_fn, plane.heading, turn_sign,
                                                      radius_nm, theta_exit_fb, n_integration_steps)

        d_final = math.dist(exit_pos_fb, target_pos)
        b1 = desired_heading(exit_pos_fb[0], exit_pos_fb[1], target_pos[0], target_pos[1])
        residual = signed_heading_diff(h_exit_fb, b1)
        assumed_bank_fb = bank_fraction * abs(residual)
        glide_ratio_fb, _ = glide_ratio_at(b1, assumed_bank_fb)
        loss_final = d_final * FT_PER_NM / glide_ratio_fb

        return loss_turn_fb + loss_final

    loss_turn = integrate_turn_altitude_loss(turn_glide_ratio_fn, plane.heading, turn_sign,
                                              radius_nm, theta_exit, n_integration_steps)

    d2 = math.dist(exit_pos, intersection)
    glide_ratio2, _ = glide_ratio_at(h_exit, 0.0)
    loss2 = d2 * FT_PER_NM / glide_ratio2

    d3 = math.dist(intersection, target_pos)
    assumed_bank3 = bank_fraction * alpha_deg
    glide_ratio3, _ = glide_ratio_at(b0, assumed_bank3)
    loss3 = d3 * FT_PER_NM / glide_ratio3

    return loss_turn + loss2 + loss3


def altitude_loss(plane, target_pos, obstacle_clearance_ft, flaps=0, alpha_deg=CONE_ALPHA_DEG,
                   bank_fraction=BANK_ANGLE_FRACTION, n_integration_steps=TURN_INTEGRATION_STEPS):
    """Estimate the altitude cost for `plane` to fly the maneuver from _maneuver_altitude_loss
    and check it against the altitude actually available (plane.alt - obstacle_clearance_ft).

    Returns (loss, landing_info):
      - If the plane has enough altitude to complete the maneuver: (loss_ft, None).
      - If it doesn't: (-1, (heading, pos_x, pos_y)), where heading/pos_x/pos_y describe
        where the plane runs out of usable altitude -- turned onto the target's bearing
        and glided (wings level) as far as the available altitude allows. That's the
        information Plane.initiate_landing needs (together with flaps/wind/temperature,
        which it already has) to compute the landing distance and touchdown point.

    Pure function: does not mutate `plane`.
    """
    loss = _maneuver_altitude_loss(plane, target_pos, flaps=flaps, alpha_deg=alpha_deg,
                                    bank_fraction=bank_fraction, n_integration_steps=n_integration_steps)
    available = plane.alt - obstacle_clearance_ft
    if loss <= available:
        return loss, None

    b0 = desired_heading(plane.pos_x, plane.pos_y, target_pos[0], target_pos[1])
    wind_delta = b0 - plane.environment_variables.wind_direction
    glide_ratio, _ = cessna_glide_ratio(plane.weight, plane.density, plane.airspeed, wind_delta,
                                         plane.environment_variables.wind_strength, flaps=flaps)
    glide_distance_nm = glide_ratio * available / FT_PER_NM
    dx, dy = compass_direction(b0)
    end_x = plane.pos_x + glide_distance_nm * dx
    end_y = plane.pos_y + glide_distance_nm * dy
    return -1, (b0, end_x, end_y)


if __name__ == "__main__":
    def test_position_on_circle_zero_sweep_is_start():
        center = (5, 0)
        start = (0, 0)
        pos = position_on_circle(center, start, 0, 1)
        passed = abs(pos[0] - start[0]) < 1e-9 and abs(pos[1] - start[1]) < 1e-9
        print(f"position_on_circle_zero_sweep_is_start: {passed}")

    def test_position_on_circle_stays_on_radius():
        center = (5, 0)
        start = (0, 0)
        radius = math.dist(center, start)
        passed = True
        for swept in [10, 45, 90, 180, 270]:
            for turn_sign in (1, -1):
                pos = position_on_circle(center, start, swept, turn_sign)
                d = math.dist(center, pos)
                passed = passed and abs(d - radius) < 1e-6
        print(f"position_on_circle_stays_on_radius: {passed}")

    def test_turn_center_matches_expected_side():
        # heading north (0), right turn -> center should be due east (+x)
        pos = (0, 0)
        radius = 2.0
        center_right = turn_center(pos, 0, radius, 1)
        center_left = turn_center(pos, 0, radius, -1)
        passed = (abs(center_right[0] - radius) < 1e-9 and abs(center_right[1]) < 1e-9 and
                  abs(center_left[0] + radius) < 1e-9 and abs(center_left[1]) < 1e-9)
        print(f"turn_center_matches_expected_side: {passed}")

    def test_heading_after_turn_wraps():
        passed = abs(heading_after_turn(350, 20, 1) - 10) < 1e-9
        print(f"heading_after_turn_wraps: {passed}")

    def test_ray_line_intersection_basic():
        # Ray heading east (90) from origin; line heading north (0) through (5, -5)
        point = ray_line_intersection((0, 0), 90, (5, -5), 0)
        passed = abs(point[0] - 5) < 1e-9 and abs(point[1] - 0) < 1e-9
        print(f"ray_line_intersection_basic: {passed}")

    def test_ray_line_intersection_parallel_raises():
        try:
            ray_line_intersection((0, 0), 45, (1, 1), 45)
            passed = False
        except ValueError:
            passed = True
        print(f"ray_line_intersection_parallel_raises: {passed}")

    def test_exit_angle_matches_alpha_construction():
        # Build the case-2 construction directly and check the angle at I equals alpha.
        alpha = 15
        cur = (0, 0)
        target = (10, 30)  # arbitrary target, defines line L and bearing b0
        b0 = desired_heading(cur[0], cur[1], target[0], target[1])
        start_heading = 250  # arbitrary starting heading, well outside the cone
        delta = signed_heading_diff(start_heading, b0)
        turn_sign = 1 if delta >= 0 else -1
        theta_exit = abs(delta) + alpha
        radius = 1.5
        center = turn_center(cur, start_heading, radius, turn_sign)
        T = position_on_circle(center, cur, theta_exit, turn_sign)
        h_exit = heading_after_turn(start_heading, theta_exit, turn_sign)
        I = ray_line_intersection(T, h_exit, cur, b0)
        bearing_I_to_target = desired_heading(I[0], I[1], target[0], target[1])
        angle_at_I = abs(signed_heading_diff(h_exit, bearing_I_to_target))
        passed = abs(angle_at_I - alpha) < 1e-6
        print(f"exit_angle_matches_alpha_construction: {passed} (angle_at_I={angle_at_I:.4f})")

    class _FakeEnvironmentVariables:
        def __init__(self, wind_direction, wind_strength):
            self.wind_direction = wind_direction
            self.wind_strength = wind_strength

    class _FakeInstruction:
        def __init__(self, bank_angle):
            self.bank_angle = bank_angle

    class _FakePlane:
        """Minimal duck-typed stand-in for Plane, so altitude_loss can be exercised here
        without importing plane.py (which itself imports this module)."""
        def __init__(self, pos_x, pos_y, heading, ground_speed, weight, density, airspeed,
                     wind_direction, wind_strength, bank_angle, alt=1_000_000):
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.heading = heading
            self.ground_speed = ground_speed
            self.weight = weight
            self.density = density
            self.airspeed = airspeed
            self.alt = alt
            self.environment_variables = _FakeEnvironmentVariables(wind_direction, wind_strength)
            self.instruction = _FakeInstruction(bank_angle)

    def _make_fake_plane():
        from .constants import W_MAX, RHO_0, V_GLIDE
        _, ground_speed = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, 0.0, 0)
        return _FakePlane(pos_x=0, pos_y=0, heading=0, ground_speed=ground_speed, weight=W_MAX,
                           density=RHO_0, airspeed=V_GLIDE, wind_direction=270, wind_strength=5,
                           bank_angle=30)

    def test_altitude_loss_case1_matches_manual_calc():
        plane = _make_fake_plane()
        target = (0, 5)  # directly ahead of heading 0 (north)
        loss, landing_info = altitude_loss(plane, target, obstacle_clearance_ft=0)
        glide_ratio, _ = cessna_glide_ratio(plane.weight, plane.density, plane.airspeed,
                                             0 - plane.environment_variables.wind_direction,
                                             plane.environment_variables.wind_strength, bank_angle=0)
        expected = 5 * FT_PER_NM / glide_ratio
        passed = landing_info is None and abs(loss - expected) < 1e-6
        print(f"altitude_loss_case1_matches_manual_calc: {passed}")

    def test_altitude_loss_case2_positive_and_monotonic():
        plane = _make_fake_plane()
        loss_90, _ = altitude_loss(plane, (10, 0), obstacle_clearance_ft=0)
        loss_170, _ = altitude_loss(plane, (1, -10), obstacle_clearance_ft=0)
        passed = loss_90 > 0 and loss_170 > 0 and loss_170 > loss_90
        print(f"altitude_loss_case2_positive_and_monotonic: {passed}")

    def test_altitude_loss_does_not_mutate_plane():
        plane = _make_fake_plane()
        before = (plane.pos_x, plane.pos_y, plane.heading)
        altitude_loss(plane, (10, 0), obstacle_clearance_ft=0)
        after = (plane.pos_x, plane.pos_y, plane.heading)
        passed = before == after
        print(f"altitude_loss_does_not_mutate_plane: {passed}")

    def test_altitude_loss_reports_landing_info_when_short_on_altitude():
        plane = _make_fake_plane()
        plane.alt = 100  # far less than the ~675ft needed to reach (0, 5)
        target = (0, 5)
        loss, landing_info = altitude_loss(plane, target, obstacle_clearance_ft=0)
        passed = loss == -1 and landing_info is not None and len(landing_info) == 3
        print(f"altitude_loss_reports_landing_info_when_short_on_altitude: {passed}")

    def _make_plane_facing_west(wind_direction=0, wind_strength=0, bank_angle=30):
        from .constants import W_MAX, RHO_0, V_GLIDE
        heading = 270
        _, ground_speed = cessna_glide_ratio(W_MAX, RHO_0, V_GLIDE, heading - wind_direction, wind_strength)
        return _FakePlane(pos_x=0, pos_y=0, heading=heading, ground_speed=ground_speed, weight=W_MAX,
                           density=RHO_0, airspeed=V_GLIDE, wind_direction=wind_direction,
                           wind_strength=wind_strength, bank_angle=bank_angle)

    def _print_vibes_case(x, y, bank_angle=30, wind_direction=0, wind_strength=0):
        plane = _make_plane_facing_west(wind_direction, wind_strength, bank_angle)
        loss, _ = altitude_loss(plane, (x, y), obstacle_clearance_ft=0)
        radius_ft = turn_radius_ft(plane.ground_speed, plane.instruction.bank_angle)
        radius_nm = radius_ft / FT_PER_NM
        print(f"x={x:>6} y={y:>6}  bank={bank_angle:>3}  wind={wind_direction}@{wind_strength:<3}"
              f"  ->  loss={loss:9.2f} ft   turn_radius={radius_nm:.4f} nm ({radius_ft:.1f} ft)")

    def vibes_check():
        """Not pass/fail -- prints altitude_loss/turn_radius across a spread of cases
        (plane facing west, heading=270) so results can be eyeballed for sanity."""
        print("\n--- vibes check: directly ahead (heading=270=west), no bank needed ---")
        for x, y in [(-0.1, 0), (-1, 0), (-10, 0)]:
            _print_vibes_case(x, y)

        print("\n--- vibes check: bank=20 ---")
        for x, y in [(-1, 0.1), (-1, -0.1), (-0.2, -0.2), (-1, -1)]:
            _print_vibes_case(x, y, bank_angle=20)

        print("\n--- vibes check: bank=45,60 (all combos) ---")
        for bank in [45, 60]:
            for x, y in [(-1, 0.1), (-0.2, -0.2), (-1, -1)]:
                _print_vibes_case(x, y, bank_angle=bank)

        print("\n--- vibes check: bank=20, wind variations ---")
        _print_vibes_case(1, -1, bank_angle=20)
        _print_vibes_case(1, 0, bank_angle=20)
        _print_vibes_case(1, 0, bank_angle=20, wind_direction=270, wind_strength=15)
        _print_vibes_case(1, 0, bank_angle=20, wind_direction=90, wind_strength=15)

    def test_vibes_check_regression():
        """Snapshot of the vibes_check() cases (plane facing west, heading=270) -- catches
        accidental shifts in altitude_loss's output. Tolerance is loose (200ft) since this
        is a regression guard, not a precision check. Three cases below were updated (not
        just left within tolerance) after implementing the overshoot/inside-turn-circle
        fallbacks: (-0.2,-0.2) and both (1,0) cases at bank=20 were confirmed overshoot
        cases (turn-radius-vs-distance too tight for the alpha-overshoot construction),
        so their loss dropped once the backtracking path was replaced by the
        residual-correction fallback -- see altitude_loss's docstring.
        """
        TOLERANCE_FT = 200
        cases = [
            # (x, y, bank_angle, wind_direction, wind_strength, expected_loss_ft)
            (-0.1, 0, 30, 0, 0, 67.51),
            (-1, 0, 30, 0, 0, 675.12),
            (-10, 0, 30, 0, 0, 6751.24),
            (-1, 0.1, 20, 0, 0, 679.34),
            (-1, -0.1, 20, 0, 0, 679.34),
            (-0.2, -0.2, 20, 0, 0, 211.88),   # overshoot fallback (was 271.37)
            (-1, -1, 20, 0, 0, 981.95),
            (-1, 0.1, 45, 0, 0, 679.34),
            (-0.2, -0.2, 45, 0, 0, 214.73),
            (-1, -1, 45, 0, 0, 985.14),
            (-1, 0.1, 60, 0, 0, 679.34),
            (-0.2, -0.2, 60, 0, 0, 220.10),
            (-1, -1, 60, 0, 0, 990.51),
            (1, -1, 20, 0, 0, 1188.90),
            (1, 0, 20, 0, 0, 1104.08),         # overshoot fallback (was 1474.94)
            (1, 0, 20, 270, 15, 799.86),
            (1, 0, 20, 90, 15, 1576.44),        # overshoot fallback (was 3185.27)
        ]
        all_passed = True
        for x, y, bank, wind_dir, wind_str, expected in cases:
            plane = _make_plane_facing_west(wind_dir, wind_str, bank)
            actual, _ = altitude_loss(plane, (x, y), obstacle_clearance_ft=0)
            ok = abs(actual - expected) <= TOLERANCE_FT
            all_passed = all_passed and ok
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] x={x:>5} y={y:>5} bank={bank:>3} wind={wind_dir}@{wind_str:<3}"
                  f"  actual={actual:9.2f}  expected={expected:9.2f}")
        print(f"vibes_check_regression: {all_passed}")

    def test_altitude_loss_raises_when_target_inside_turn_circle():
        plane = _make_plane_facing_west(wind_direction=0, wind_strength=0, bank_angle=20)
        radius_ft = turn_radius_ft(plane.ground_speed, plane.instruction.bank_angle)
        radius_nm = radius_ft / FT_PER_NM
        # Target placed well inside the turn circle for a target 90 degrees off-heading:
        # a coordinated turn can never roll out pointed at a point inside its own circle.
        target = (-radius_nm * 0.1, radius_nm * 0.1)
        try:
            altitude_loss(plane, target, obstacle_clearance_ft=0)
            passed = False
        except ValueError:
            passed = True
        print(f"altitude_loss_raises_when_target_inside_turn_circle: {passed}")

    def test_altitude_loss_residual_fallback_matches_manual_calc():
        # (1, 0) at bank=20, no wind is a confirmed overshoot case (intersection point
        # falls past the target), so this should hit the residual-correction fallback:
        # turn only until heading == bearing (no alpha overshoot), then glide straight
        # from wherever that leaves the plane, banked by a fraction of whatever residual
        # heading error remains there (not capped at alpha_deg).
        plane = _make_plane_facing_west(wind_direction=0, wind_strength=0, bank_angle=20)
        target = (1, 0)
        loss, _ = altitude_loss(plane, target, obstacle_clearance_ft=0)

        cur_pos = (plane.pos_x, plane.pos_y)
        b0 = desired_heading(cur_pos[0], cur_pos[1], target[0], target[1])
        delta = signed_heading_diff(plane.heading, b0)
        turn_sign = 1 if delta >= 0 else -1
        radius_nm = turn_radius_ft(plane.ground_speed, plane.instruction.bank_angle) / FT_PER_NM
        center = turn_center(cur_pos, plane.heading, radius_nm, turn_sign)

        def turn_glide_ratio_fn(heading_deg):
            wind_delta = heading_deg - plane.environment_variables.wind_direction
            return cessna_glide_ratio(plane.weight, plane.density, plane.airspeed, wind_delta,
                                       plane.environment_variables.wind_strength,
                                       bank_angle=plane.instruction.bank_angle)

        theta_exit_fb = abs(delta)
        loss_turn_fb = integrate_turn_altitude_loss(turn_glide_ratio_fn, plane.heading, turn_sign,
                                                      radius_nm, theta_exit_fb, TURN_INTEGRATION_STEPS)
        exit_pos_fb = position_on_circle(center, cur_pos, theta_exit_fb, turn_sign)
        h_exit_fb = heading_after_turn(plane.heading, theta_exit_fb, turn_sign)
        b1 = desired_heading(exit_pos_fb[0], exit_pos_fb[1], target[0], target[1])
        residual = signed_heading_diff(h_exit_fb, b1)
        assumed_bank = BANK_ANGLE_FRACTION * abs(residual)
        wind_delta = b1 - plane.environment_variables.wind_direction
        glide_ratio, _ = cessna_glide_ratio(plane.weight, plane.density, plane.airspeed, wind_delta,
                                             plane.environment_variables.wind_strength, bank_angle=assumed_bank)
        loss_final = math.dist(exit_pos_fb, target) * FT_PER_NM / glide_ratio
        expected = loss_turn_fb + loss_final
        passed = abs(loss - expected) < 1e-6
        print(f"altitude_loss_residual_fallback_matches_manual_calc: {passed}")

    test_position_on_circle_zero_sweep_is_start()
    test_position_on_circle_stays_on_radius()
    test_turn_center_matches_expected_side()
    test_heading_after_turn_wraps()
    test_ray_line_intersection_basic()
    test_ray_line_intersection_parallel_raises()
    test_exit_angle_matches_alpha_construction()
    test_altitude_loss_case1_matches_manual_calc()
    test_altitude_loss_case2_positive_and_monotonic()
    test_altitude_loss_does_not_mutate_plane()
    test_altitude_loss_reports_landing_info_when_short_on_altitude()
    test_vibes_check_regression()
    test_altitude_loss_raises_when_target_inside_turn_circle()
    test_altitude_loss_residual_fallback_matches_manual_calc()
    vibes_check()
