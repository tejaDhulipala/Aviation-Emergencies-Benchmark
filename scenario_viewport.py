"""Rendering/compositing layer for the scenario builder GUI. No Tkinter dependency, so it's
independently importable/testable. Owns GUI-specific visual constants that don't belong in
utils/constants.py (per that module's own docstring: only real physical constants there)."""

import types

import pygame as pg
from PIL import Image

import main
from plane import Plane

LANDING_OPTION_COLOR = (255, 255, 0)   # yellow
LANDING_OPTION_RADIUS = 10             # px
LANDING_OPTION_NUMBER_FONT_SIZE = 18
SMALL_LANDING_OPTION_RADIUS = 5              # px, half of standard
SMALL_LANDING_OPTION_NUMBER_FONT_SIZE = 10   # roughly half of standard
PLANE_COLOR = (255, 0, 0)              # red, matches main.py's PLANE_COLOR
PLANE_RADIUS = 8
DEFAULT_BANK_ANGLE_DEG = 30            # fixed, non-editable bank angle used for the physics probe


def make_inverse_scale(center_x, center_y, size_nm, width, height):
    """Inverse of main.make_scale: converts screen pixel coordinates back to (x, y) in NM."""
    def inverse_scale(sx, sy):
        x = center_x + (sx - width / 2) / width * size_nm
        y = center_y - (sy - height / 2) / height * size_nm
        return x, y
    return inverse_scale


def pil_to_pygame_surface(pil_image):
    pil_image = pil_image.convert("RGB")
    return pg.image.frombuffer(pil_image.tobytes(), pil_image.size, "RGB")


def surface_to_pil_image(surface):
    data = pg.image.tobytes(surface, "RGB")
    return Image.frombytes("RGB", surface.get_size(), data)


def draw_plane(surface, heading_deg, sx, sy, radius=PLANE_RADIUS, color=PLANE_COLOR):
    """Draws a circle + heading arrow by reusing plane.Plane.draw's vector-drawing logic
    against a minimal stand-in (Plane.draw only reads aircraft_condition/landing, and only
    when aircraft_condition == "landed", which we never set here)."""
    stand_in = types.SimpleNamespace(aircraft_condition="intact", landing=None)
    Plane.draw(stand_in, surface, color, sx, sy, heading=heading_deg, radius=radius, scale=None)


def draw_landing_option(surface, option, sx, sy):
    """option: {"number": int, "rel_x": float, "rel_y": float, "heading": float | None,
    "small": bool (optional, default False)}."""
    if option.get("small", False):
        radius, font_size = SMALL_LANDING_OPTION_RADIUS, SMALL_LANDING_OPTION_NUMBER_FONT_SIZE
    else:
        radius, font_size = LANDING_OPTION_RADIUS, LANDING_OPTION_NUMBER_FONT_SIZE

    if option["heading"] is None:
        pg.draw.circle(surface, LANDING_OPTION_COLOR, (sx, sy), radius)
    else:
        draw_plane(surface, option["heading"], sx, sy, radius=radius, color=LANDING_OPTION_COLOR)

    font = pg.font.SysFont(None, font_size, bold=True)
    label = font.render(str(option["number"]), True, (0, 0, 0))
    surface.blit(label, (sx - label.get_width() // 2, sy - label.get_height() // 2))


def render_full_surface(background_pil, size_nm, resolution, heading_deg, landing_options):
    """Composes background + ruler + centered plane + all landing options into one Surface,
    used for both the live preview and the final export."""
    surface = pil_to_pygame_surface(background_pil)
    scale = main.make_scale(0, 0, size_nm, width=resolution, height=resolution)
    main.draw_ruler(surface, scale, size_nm, 0, 0, width=resolution, height=resolution)
    draw_plane(surface, heading_deg, resolution // 2, resolution // 2, radius=PLANE_RADIUS)
    for option in landing_options:
        sx, sy = scale(option["rel_x"], option["rel_y"])
        draw_landing_option(surface, option, sx, sy)
    return surface
