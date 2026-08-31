import math
from dataclasses import dataclass

from utils import cessna_glide_ratio, density_from_altitude, desired_heading, signed_heading_diff, landing_distance, compass_direction
from utils.constants import (
    OBSTACLE_CLEARANCE_FT,
    ENGINE_RPM_DEFAULT,
    VALID_FLAP_SETTINGS,
)
from utils.paths import altitude_loss, turn_radius_ft, CONE_ALPHA_DEG

@dataclass
class EnvironmentVariables:
    wind_strength: int # knots
    wind_direction: int # magnetic degrees
    temperature: int # celsius

@dataclass
class Instruction:
    goal_x: float
    goal_y: float
    airspeed: int
    bank_angle: int
    flaps: int = 0
    assert flaps in VALID_FLAP_SETTINGS



class Plane:
    def __init__(self, pos_x, pos_y, alt, airspeed, weight, heading, env_vars: EnvironmentVariables, inst: Instruction):
        self.pos_x = pos_x # nm
        self.pos_y = pos_y # nm
        self.alt = alt # ft
        self.airspeed = airspeed # kias
        self.heading = heading # degrees
        self.engine_rpm = ENGINE_RPM_DEFAULT # rpm
        self.aircraft_condition = "intact"
        self.environment_variables = env_vars
        self.time_simulated = 0
        self.instruction = inst
        self.instruction_completed = True
        self.density = density_from_altitude(self.alt, self.environment_variables.temperature)
        self.weight = weight
        self.glide_ratio, self.ground_speed = cessna_glide_ratio(self.weight, self.density, self.airspeed,
        self.heading - self.environment_variables.wind_direction, self.environment_variables.wind_strength)
        self.max_glide_ratio, _ = cessna_glide_ratio(weight=self.weight, wind_delta=180,
            wind_speed=self.environment_variables.wind_strength)
        self.landing = None

    def give_instruction(self, instruction: Instruction):
        self.instruction = instruction
        self.instruction_completed = False

    def update_glide_ratio(self):
        self.density = density_from_altitude(self.alt, self.environment_variables.temperature)
        self.glide_ratio, self.ground_speed = cessna_glide_ratio(self.weight, self.density, self.airspeed,
        self.heading - self.environment_variables.wind_direction, self.environment_variables.wind_strength,
        flaps=self.instruction.flaps)
        print(f"Glide Ratio: {self.glide_ratio}")


    def follow_instruction(self):
        # Follow instruction by turning onto the goal's bearing and gliding to it,
        # using altitude_loss's turn+glide-path model (see utils/paths.py) to estimate
        # the altitude cost of the whole maneuver up front.
        if self.instruction_completed:
            return
        self.airspeed = self.instruction.airspeed
        self.update_glide_ratio()  # refresh ground_speed (and glide_ratio) for the new airspeed before estimating loss
        target_pos = (self.instruction.goal_x, self.instruction.goal_y)
        target_heading = desired_heading(self.pos_x, self.pos_y, target_pos[0], target_pos[1])
        if abs(signed_heading_diff(self.heading, target_heading)) <= CONE_ALPHA_DEG:
            print("Target within no-turn envelope at current heading: gliding straight, no turn executed.")
        else:
            print("Target outside no-turn envelope: executing a coordinated turn onto the target bearing.")
        loss, landing_info = altitude_loss(self, target_pos, OBSTACLE_CLEARANCE_FT, flaps=self.instruction.flaps)
        self.instruction_completed = True
        print(f"Turn radius: {turn_radius_ft(self.ground_speed, self.instruction.bank_angle)} ft")
        if landing_info is None:
            print(f"Estimated altitude loss to target: {loss}")
            self.alt -= loss
            self.pos_x, self.pos_y = target_pos
            self.heading = target_heading
            self.update_glide_ratio()
        else:
            # Not enough altitude to reach the target: turn onto the bearing and glide
            # as far as the remaining altitude allows, then land there.
            self.heading, self.pos_x, self.pos_y = landing_info
            self.alt = OBSTACLE_CLEARANCE_FT
            self.aircraft_condition = "landed"
            self.update_glide_ratio()
            self.initiate_landing()
        print(f"({self.pos_x}nm, {self.pos_y}nm, {self.alt}ft, {self.heading} degreees)")
    
    def initiate_landing(self):
        headwind = self.environment_variables.wind_strength * math.cos((self.environment_variables.wind_direction - self.heading) / 180 * math.pi)
        landing_dist = landing_distance(self.environment_variables.temperature, headwind, self.instruction.flaps)
        print(f"Landing distance: {landing_dist}")
        x_start = self.pos_x
        y_start = self.pos_y
        # Calculate end position after traveling landing_dist nm at current heading
        dx, dy = compass_direction(self.heading)
        x_end = x_start + landing_dist * dx
        y_end = y_start + landing_dist * dy
        self.landing = x_start, y_start, x_end, y_end

    def draw(self, surface, color, sx, sy, heading=None, radius=8, scale=None, size_scale=1.0):
        """
        Draw the plane as a circle with an arrow for heading.
        sx, sy: screen coordinates (already scaled)
        heading: if None, use self.heading
        scale: function to convert (x, y) in NM to screen coordinates (sx, sy)
        size_scale: multiplies the arrow's length/head/line-width (independent of radius,
            which only sizes the circle) -- 1.0 reproduces the original fixed
            arrow_length=40/head=15/width=4 exactly, regardless of radius.
        """
        import pygame as pg
        if heading is None:
            heading = self.heading
        pg.draw.circle(surface, color, (sx, sy), radius)
        arrow_length = 40 * size_scale
        head_length = 15 * size_scale
        line_width = max(1, round(4 * size_scale))
        angle_rad = math.radians(heading)
        arrow_dx = arrow_length * math.sin(angle_rad)
        arrow_dy = -arrow_length * math.cos(angle_rad)
        pg.draw.line(surface, color, (sx, sy), (sx + int(arrow_dx), sy + int(arrow_dy)), line_width)
        # Arrowhead
        head_angle1 = angle_rad + math.radians(150)
        head_angle2 = angle_rad - math.radians(150)
        head1 = (sx + int(arrow_dx + head_length * math.sin(head_angle1)), sy + int(arrow_dy - head_length * math.cos(head_angle1)))
        head2 = (sx + int(arrow_dx + head_length * math.sin(head_angle2)), sy + int(arrow_dy - head_length * math.cos(head_angle2)))
        pg.draw.line(surface, color, (sx + int(arrow_dx), sy + int(arrow_dy)), head1, line_width)
        pg.draw.line(surface, color, (sx + int(arrow_dx), sy + int(arrow_dy)), head2, line_width)
        # Draw landing path if landed
        if self.aircraft_condition == "landed" and self.landing is not None and scale is not None:
            x_start, y_start, x_end, y_end = self.landing
            start_px, start_py = scale(x_start, y_start)
            end_px, end_py = scale(x_end, y_end)
            pg.draw.line(surface, (0, 0, 0), (start_px, start_py), (end_px, end_py), 4)