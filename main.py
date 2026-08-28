import math

import pygame as pg

# --- Visualization parameters ---
WIDTH, HEIGHT = 900, 900  # square, matches SatelliteMap's square output
MARGIN = 30  # pixels, ruler label spacing only

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
