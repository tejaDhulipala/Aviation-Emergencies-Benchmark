import pygame as pg
import sys
import os
import shutil
import ctypes
import math
from plane import Plane, EnvironmentVariables, Instruction
from map import SatelliteMap, offset_latlon
from utils.constants import FT_PER_NM

if sys.platform == "win32":
    # Prevent Windows from bitmap-scaling the whole window on high-DPI
    # displays, which is what makes small text (e.g. the ruler labels)
    # look blurry.
    ctypes.windll.user32.SetProcessDPIAware()

# --- Visualization parameters ---
WIDTH, HEIGHT = 900, 900  # square, matches SatelliteMap's square output
PLANE_COLOR = (255, 0, 0)  # Red
TARGET_COLOR = (255, 100, 100)
AFTER_TURN_COLOR = (0, 255, 100)
ARROW_COLOR = (255, 255, 0)
DOT_RADIUS = 8
MARGIN = 30  # pixels, ruler label spacing only

# --- Geo / imagery parameters ---
START_X, START_Y = 0, 0  # plane's local (pos_x, pos_y) at the origin lat/lon
MIN_SIZE_FT = 200  # floor so size_nm never collapses to ~0 near landing
CUR_PHOTO_DIR = "CurPhoto"

# --- Utility for scaling positions: always centers (center_x, center_y) on screen ---
def make_scale(center_x, center_y, size_nm, width=WIDTH, height=HEIGHT):
    def scale(x, y):
        sx = int(width / 2 + (x - center_x) / size_nm * width)
        sy = int(height / 2 - (y - center_y) / size_nm * height)
        return sx, sy
    return scale

GRID_LINE_COLOR = (255, 255, 255)
GRID_LABEL_COLOR = (255, 0, 0)
N_SIDE_TICKS = 8  # target ticks per side; actual count varies once the step is rounded to a nice value
CORNER_CLEARANCE = 45  # px; skip a label if it would collide with the other axis's labels in the corner

def _nice_step(raw_step):
    """Round raw_step up to the nearest 1-2-5-10 value at the same order of
    magnitude, so gridline spacing is always a clean, consistent increment."""
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for mult in (1, 2, 5, 10):
        candidate = mult * magnitude
        if candidate >= raw_step:
            return candidate

def draw_ruler(screen, scale, size_nm, center_x, center_y, width=WIDTH, height=HEIGHT):
    font = pg.font.SysFont(None, 22, bold=True)
    raw_step = (size_nm / 2) / N_SIDE_TICKS
    step = _nice_step(raw_step)
    decimals = max(0, -math.floor(math.log10(step)))
    n_ticks = math.ceil((size_nm / 2) / step)
    y_label_row = height - MARGIN // 2
    x_label_col = MARGIN // 2

    ticks = []
    grid_surface = pg.Surface((width, height), pg.SRCALPHA)
    for k in range(-n_ticks, n_ticks + 1):
        x_nm = center_x + k * step
        sx, _ = scale(x_nm, center_y)
        pg.draw.line(grid_surface, (*GRID_LINE_COLOR, 90), (sx, 0), (sx, height), 1)

        y_nm = center_y + k * step
        _, sy = scale(center_x, y_nm)
        pg.draw.line(grid_surface, (*GRID_LINE_COLOR, 90), (0, sy), (width, sy), 1)

        ticks.append((sx, x_nm, sy, y_nm))
    screen.blit(grid_surface, (0, 0))

    for sx, x_nm, sy, y_nm in ticks:
        label_x = font.render(f"{x_nm - center_x:+.{decimals}f}", True, GRID_LABEL_COLOR)
        lx = sx - label_x.get_width() // 2
        if lx >= x_label_col + CORNER_CLEARANCE:
            screen.blit(label_x, (lx, y_label_row))

        label_y = font.render(f"{y_nm - center_y:+.{decimals}f}", True, GRID_LABEL_COLOR)
        ly = sy - label_y.get_height() // 2
        if ly <= y_label_row - CORNER_CLEARANCE - label_y.get_height():
            screen.blit(label_y, (x_label_col, ly))

def get_instruction_from_input(plane):
    """Prompts for a goal relative to the plane's current position (nm, east/north positive),
    then converts it to the absolute goal_x/goal_y that Plane/Instruction operate on."""
    s = ""
    while True:
        try:
            s = input("give instruction (rel_goal_x rel_goal_y airspeed bank_angle flaps): ")
            rel_goal_x, rel_goal_y, airspeed, bank_angle, flaps = s.split()
            goal_x = plane.pos_x + float(rel_goal_x)
            goal_y = plane.pos_y + float(rel_goal_y)
            airspeed = int(airspeed)
            bank_angle = int(bank_angle)
            flaps = int(flaps)
            return goal_x, goal_y, airspeed, bank_angle, flaps, True
        except Exception:
            if s.strip().lower() in ["q", "quit"]:
                return None, None, None, None, None, False
            print("Invalid input, try again.")

def fetch_photo(smap, plane, photo_index, origin_lat, origin_lon):
    """Fetch a fresh satellite photo centered on the plane, sized to its glide-reachable radius.
    Pure PIL/requests logic, no pygame dependency, so it's testable without a display."""
    lat, lon = offset_latlon(origin_lat, origin_lon, plane.pos_x - START_X, plane.pos_y - START_Y)
    size_nm = max(plane.alt * plane.max_glide_ratio * 2, MIN_SIZE_FT) / FT_PER_NM
    img = smap.get_image(lat, lon, size_nm, out_size=WIDTH)
    path = os.path.join(CUR_PHOTO_DIR, f"{photo_index:04d}.png")
    img.save(path)
    return path, size_nm, lat, lon

def load_background(path):
    surface = pg.image.load(path).convert_alpha()
    return pg.transform.scale(surface, (WIDTH, HEIGHT))

def render_frame(screen, plane, background, size_nm):
    """Draw one full frame (background + target + plane + grid) onto screen. Returns the scale
    function used, in case a caller needs to convert further nm coordinates to screen pixels."""
    scale = make_scale(plane.pos_x, plane.pos_y, size_nm)
    screen.blit(background, (0, 0))
    ox, oy = scale(plane.pos_x, plane.pos_y)
    if plane.instruction is not None:
        tx, ty = scale(plane.instruction.goal_x, plane.instruction.goal_y)
        pg.draw.circle(screen, TARGET_COLOR, (tx, ty), DOT_RADIUS)
    plane.draw(screen, PLANE_COLOR, ox, oy, heading=plane.heading, radius=DOT_RADIUS, scale=scale)
    draw_ruler(screen, scale, size_nm, plane.pos_x, plane.pos_y)
    return scale

# --- Main visualization ---
def main(origin_lat=28.106733, origin_lon=-80.679769, alt=500, airspeed=80, weight=2400,
         heading=270, env_vars=None):
    # origin_lat/origin_lon default: 28°06'22.75"N 80°41'15.89"W
    if env_vars is None:
        env_vars = EnvironmentVariables(wind_strength=0, wind_direction=0, temperature=15)

    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Plane Turn Visualization")
    clock = pg.time.Clock()

    shutil.rmtree(CUR_PHOTO_DIR, ignore_errors=True)
    os.makedirs(CUR_PHOTO_DIR, exist_ok=True)
    smap = SatelliteMap()
    photo_index = 0

    plane = Plane(START_X, START_Y, alt=alt, airspeed=airspeed, weight=weight, heading=heading, env_vars=env_vars, inst=None)
    print(f"Altitude: {plane.alt}. Weight: {plane.weight}. Heading: {plane.heading}. StartX: {plane.pos_x}. StartY: {plane.pos_y}")
    print(f"Winds {round(env_vars.wind_direction / 10)} @ {env_vars.wind_strength}. Temperature {env_vars.temperature}")

    photo_path, size_nm, _, _ = fetch_photo(smap, plane, photo_index, origin_lat, origin_lon)
    background = load_background(photo_path)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        render_frame(screen, plane, background, size_nm)
        pg.display.flip()
        clock.tick(30)

        # Get new instruction from user
        if not plane.landing and running:
            goal_x, goal_y, airspeed, bank_angle, flaps, running = get_instruction_from_input(plane)
            if not running:
                continue
            new_instruction = Instruction(goal_x=goal_x, goal_y=goal_y, airspeed=airspeed, bank_angle=bank_angle, flaps=flaps)
            plane.give_instruction(new_instruction)
            try:
                plane.follow_instruction()
            except ValueError as e:
                print(f"Instruction not physically possible: {e}")
                continue
            photo_index += 1
            photo_path, size_nm, _, _ = fetch_photo(smap, plane, photo_index, origin_lat, origin_lon)
            background = load_background(photo_path)

    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main(alt=1500)
