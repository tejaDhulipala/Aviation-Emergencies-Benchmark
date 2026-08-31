"""Tkinter GUI for authoring emergency-landing benchmark scenarios: position a plane on a
real satellite map, click to place candidate landing points with directions, mark the ground
truth, probe altitude-loss physics at three complexity levels, and export an image+JSON pair.
"""

import json
import os
import random
import time
import tkinter as tk
from tkinter import messagebox, ttk

import pygame as pg
from PIL import ImageTk

from map import SatelliteMap
from scenario_viewport import (
    DEFAULT_BANK_ANGLE_DEG,
    make_inverse_scale,
    render_full_surface,
    surface_to_pil_image,
)
from main import make_scale
from altitude_loss_levels import compute_altitude_loss, altitude_loss_full_physics_path
from utils.basic_math import desired_heading
from utils.constants import W_MAX, V_GLIDE, OBSTACLE_CLEARANCE_FT, FT_PER_NM, GR_0
from utils.dubins import dubins_path_points
from scenario_tags import TAG_CATEGORIES, TagHeader, split_tags_by_category

PROBE_DOT_COLOR = "red"
PROBE_DOT_RADIUS_PX = 3
PROBE_PATH_COLOR = "red"
PROBE_PATH_WIDTH_PX = 1

pg.font.init()  # required for pg.font.SysFont used by main.draw_ruler / landing-option labels.
                # Never call pg.init()/pg.display.set_mode() here -- a real pygame display
                # window competing with Tkinter's own mainloop is the failure mode to avoid.

CANVAS_PREVIEW_MAX_PX = 800
DATASET_DIR = "dataset"

DEFAULT_LAT = "28.106733"
DEFAULT_LON = "-80.679769"


class ScenarioBuilderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Scenario Builder")

        self.smap = SatelliteMap()
        self.landing_options = []       # list of {"number", "rel_x", "rel_y", "heading"}
        self.pending_point_index = None  # index in landing_options awaiting its heading click
        self.ground_truth_index = tk.IntVar(value=-1)

        self._cached_background = None      # PIL.Image, from the last successful map fetch
        self._cached_background_key = None  # (lat, lon, size_nm, resolution) tuple
        self._current_photoimage = None     # keep a live ref so Tk doesn't GC it blank
        self._preview_scale_factor = 1.0    # displayed_px -> full_resolution_px
        self._probe_overlay_ids = []        # canvas item ids for the active altitude-loss probe's dot/path
        self.selected_tags = set()          # tag_ids (from scenario_tags.TAG_CATEGORIES) chosen for this scenario

        self._build_widgets()

    # ------------------------------------------------------------------ widgets

    def _build_widgets(self):
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        setup = ttk.LabelFrame(left, text="Scenario Setup", padding=6)
        setup.grid(row=0, column=0, sticky="ew")

        self.fields = {}
        field_specs = [
            ("latitude", DEFAULT_LAT),
            ("longitude", DEFAULT_LON),
            ("heading_deg", "270"),
            ("viewport_width_nm", "6.5"),
            ("resolution_px", "800"),
            ("altitude_agl_ft", "1500"),
            ("wind_speed_kt", "0"),
            ("wind_direction_deg", "0"),
        ]
        for i, (name, default) in enumerate(field_specs):
            ttk.Label(setup, text=name).grid(row=i, column=0, sticky="w")
            var = tk.StringVar(value=default)
            entry = ttk.Entry(setup, textvariable=var, width=18)
            entry.grid(row=i, column=1, sticky="ew")
            entry.bind("<Return>", lambda e: self.on_render_map())
            self.fields[name] = var

        ttk.Button(setup, text="Render / Refresh Map", command=self.on_render_map).grid(
            row=len(field_specs), column=0, columnspan=2, sticky="ew", pady=(6, 0))

        options_frame = ttk.LabelFrame(left, text="Landing Options", padding=6)
        options_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.options_frame = options_frame
        self.option_row_widgets = []  # list of dicts of per-row StringVars/widgets

        self.text_boxes = {}
        text_box_specs = [
            ("answer_explanation", "Answer Explanation"),
            ("prompt_additions", "Prompt Additions"),
        ]
        row = 2
        for key, title in text_box_specs:
            frame = ttk.LabelFrame(left, text=title, padding=6)
            frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
            text = tk.Text(frame, width=32, height=5, wrap="word")
            text.pack(fill="both", expand=True)
            self.text_boxes[key] = text
            row += 1

        ttk.Button(left, text="Randomize numbers", command=self.on_randomize_numbers).grid(
            row=row, column=0, sticky="ew", pady=(8, 0))
        row += 1
        ttk.Button(left, text="Tags...", command=self.on_open_tags_dialog).grid(
            row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1
        ttk.Button(left, text="Export Scenario", command=self.on_export).grid(
            row=row, column=0, sticky="ew", pady=(4, 0))
        row += 1

        self.status_var = tk.StringVar(value="Enter scenario setup, then click Render / Refresh Map.")
        ttk.Label(left, textvariable=self.status_var, wraplength=260).grid(
            row=row, column=0, sticky="ew", pady=(8, 0))

        canvas_size = min(int(field_specs[4][1]), CANVAS_PREVIEW_MAX_PX)
        self.canvas = tk.Canvas(right, width=canvas_size, height=canvas_size, background="black")
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

    def _rebuild_option_rows(self):
        for row in self.option_row_widgets:
            row["frame"].destroy()
        self.option_row_widgets = []

        for i, option in enumerate(self.landing_options):
            frame = ttk.Frame(self.options_frame)
            frame.grid(row=i, column=0, sticky="ew")

            rel_x_var = tk.StringVar(value=f"{option['rel_x']:.4f}")
            rel_y_var = tk.StringVar(value=f"{option['rel_y']:.4f}")
            heading_var = tk.StringVar(value="" if option["heading"] is None else f"{option['heading']:.1f}")

            ttk.Label(frame, text=f"#{option['number']}").grid(row=0, column=0)
            for col, var in ((1, rel_x_var), (2, rel_y_var), (3, heading_var)):
                entry = ttk.Entry(frame, textvariable=var, width=8)
                entry.grid(row=0, column=col)
                entry.bind("<FocusOut>", lambda e, idx=i: self.on_option_edited(idx))
                entry.bind("<Return>", lambda e, idx=i: self.on_option_edited(idx))
            ttk.Radiobutton(frame, variable=self.ground_truth_index, value=i).grid(row=0, column=4)
            small_var = tk.BooleanVar(value=option.get("small", False))
            ttk.Checkbutton(frame, text="small", variable=small_var,
                            command=lambda idx=i: self.on_toggle_small(idx)).grid(row=0, column=5)
            ttk.Button(frame, text="✕", width=2, command=lambda idx=i: self.on_delete_option(idx)).grid(row=0, column=6)

            self.option_row_widgets.append({
                "frame": frame, "rel_x": rel_x_var, "rel_y": rel_y_var, "heading": heading_var, "small": small_var,
            })

    def on_option_edited(self, index):
        if index >= len(self.landing_options):
            return
        row = self.option_row_widgets[index]
        try:
            rel_x = float(row["rel_x"].get())
            rel_y = float(row["rel_y"].get())
            heading_text = row["heading"].get().strip()
            heading = float(heading_text) if heading_text else None
        except ValueError:
            self.status_var.set("Invalid number in landing option fields; edit ignored.")
            return
        option = self.landing_options[index]
        option["rel_x"], option["rel_y"], option["heading"] = rel_x, rel_y, heading
        self.redraw_canvas()

    def on_toggle_small(self, index):
        if index >= len(self.landing_options):
            return
        small = self.option_row_widgets[index]["small"].get()
        self.landing_options[index]["small"] = small
        self.redraw_canvas()

    def on_delete_option(self, index):
        if index >= len(self.landing_options):
            return
        removed_number = self.landing_options[index]["number"]
        del self.landing_options[index]

        if self.pending_point_index == index:
            self.pending_point_index = None
        elif self.pending_point_index is not None and self.pending_point_index > index:
            self.pending_point_index -= 1

        gt = self.ground_truth_index.get()
        if gt == index:
            self.ground_truth_index.set(-1)
        elif gt > index:
            self.ground_truth_index.set(gt - 1)

        self.status_var.set(f"Deleted option #{removed_number}.")
        self._rebuild_option_rows()
        self.redraw_canvas()

    # ------------------------------------------------------------------ map fetch / redraw

    def _read_float_fields(self):
        return {
            "lat": float(self.fields["latitude"].get()),
            "lon": float(self.fields["longitude"].get()),
            "heading": float(self.fields["heading_deg"].get()),
            "size_nm": float(self.fields["viewport_width_nm"].get()),
            "resolution": int(self.fields["resolution_px"].get()),
            "altitude_agl_ft": float(self.fields["altitude_agl_ft"].get()),
            "wind_speed": float(self.fields["wind_speed_kt"].get()),
            "wind_direction": float(self.fields["wind_direction_deg"].get()),
        }

    def on_render_map(self):
        try:
            values = self._read_float_fields()
        except ValueError:
            messagebox.showerror("Invalid input", "One or more scenario setup fields is not a valid number.")
            return

        key = (self.fields["latitude"].get(), self.fields["longitude"].get(), values["size_nm"], values["resolution"])
        if key != self._cached_background_key:
            self.status_var.set("Fetching satellite imagery...")
            self.root.update_idletasks()
            try:
                image = self.smap.get_image(values["lat"], values["lon"], values["size_nm"], out_size=values["resolution"])
            except ValueError as e:
                messagebox.showerror("Map fetch failed", str(e))
                self.status_var.set("Map fetch failed; adjust viewport width or resolution and retry.")
                return
            except Exception as e:
                messagebox.showerror("Map fetch failed", f"Could not fetch satellite imagery: {e}")
                self.status_var.set("Map fetch failed.")
                return
            self._cached_background = image
            self._cached_background_key = key
            self.status_var.set("Map fetched.")

        display_size = min(values["resolution"], CANVAS_PREVIEW_MAX_PX)
        self._preview_scale_factor = values["resolution"] / display_size
        self.canvas.config(width=display_size, height=display_size)
        self.redraw_canvas()

    def redraw_canvas(self):
        if self._cached_background is None:
            return
        try:
            values = self._read_float_fields()
        except ValueError:
            return
        surface = render_full_surface(self._cached_background, values["size_nm"], values["resolution"],
                                       values["heading"], self.landing_options)
        pil_image = surface_to_pil_image(surface)
        display_size = min(values["resolution"], CANVAS_PREVIEW_MAX_PX)
        if display_size != values["resolution"]:
            pil_image = pil_image.resize((display_size, display_size))
        self._current_photoimage = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._current_photoimage)

    # ------------------------------------------------------------------ canvas interaction

    def _click_to_rel_xy(self, event):
        values = self._read_float_fields()
        full_sx = event.x * self._preview_scale_factor
        full_sy = event.y * self._preview_scale_factor
        inverse_scale = make_inverse_scale(0, 0, values["size_nm"], values["resolution"], values["resolution"])
        return inverse_scale(full_sx, full_sy)

    def on_left_click(self, event):
        if self._cached_background is None:
            return
        try:
            rel_x, rel_y = self._click_to_rel_xy(event)
        except ValueError:
            return

        if self.pending_point_index is None:
            option = {"number": len(self.landing_options) + 1, "rel_x": rel_x, "rel_y": rel_y, "heading": None}
            self.landing_options.append(option)
            self.pending_point_index = len(self.landing_options) - 1
            self.status_var.set(f"Placed option #{option['number']}. Click again to set its landing direction.")
        else:
            option = self.landing_options[self.pending_point_index]
            option["heading"] = desired_heading(option["rel_x"], option["rel_y"], rel_x, rel_y)
            self.status_var.set(f"Set option #{option['number']}'s heading to {option['heading']:.1f} deg.")
            self.pending_point_index = None

        self._rebuild_option_rows()
        self.redraw_canvas()

    def on_right_click(self, event):
        if self._cached_background is None:
            return
        try:
            rel_x, rel_y = self._click_to_rel_xy(event)
            values = self._read_float_fields()
        except ValueError:
            return

        common_args = dict(
            x_cur=0, y_cur=0, heading_cur=values["heading"], x_goal=rel_x, y_goal=rel_y,
            altitude_agl_ft=values["altitude_agl_ft"], weight=W_MAX, airspeed=V_GLIDE,
            wind_direction=values["wind_direction"], wind_speed=values["wind_speed"],
            bank_angle=DEFAULT_BANK_ANGLE_DEG, flaps=0, obstacle_clearance_ft=OBSTACLE_CLEARANCE_FT)

        # Levels 1/2 don't take a target heading, so their results are fixed for this
        # probe point; only level 3 (Dubins) needs the editable heading_goal below.
        static_lines = [f"Probe point: rel_x={rel_x:.3f} nm, rel_y={rel_y:.3f} nm"]
        for level in (1, 2):
            loss, reachable = compute_altitude_loss(level, **common_args)
            level_name = {1: "Barebones", 2: "Wind-aware"}[level]
            static_lines.append(f"Level {level} ({level_name}): {loss:.0f} ft loss, reachable={reachable}")

        # Default to the straight-line bearing to the point (degenerates the arrival
        # turn to ~0 degrees); the user can edit it to see how a required arrival
        # heading (e.g. a landing option's approach direction) changes level 3's loss.
        default_heading_goal = desired_heading(0, 0, rel_x, rel_y)

        dialog = tk.Toplevel(self.root)
        dialog.title("Altitude-loss probe")
        dialog.transient(self.root)

        for i, line in enumerate(static_lines):
            ttk.Label(dialog, text=line).grid(row=i, column=0, columnspan=2, sticky="w", padx=8, pady=(8 if i == 0 else 2, 2))

        heading_row = len(static_lines)
        ttk.Label(dialog, text="Arrival heading (deg):").grid(row=heading_row, column=0, sticky="w", padx=8, pady=(8, 2))
        heading_var = tk.StringVar(value=f"{default_heading_goal:.1f}")
        heading_entry = ttk.Entry(dialog, textvariable=heading_var, width=10)
        heading_entry.grid(row=heading_row, column=1, sticky="w", padx=8, pady=(8, 2))

        level3_var = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=level3_var).grid(
            row=heading_row + 1, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8))

        scale = make_scale(0, 0, values["size_nm"], width=values["resolution"], height=values["resolution"])

        def to_canvas_px(x, y):
            sx, sy = scale(x, y)
            return sx / self._preview_scale_factor, sy / self._preview_scale_factor

        def clear_probe_overlay():
            for item_id in self._probe_overlay_ids:
                self.canvas.delete(item_id)
            self._probe_overlay_ids = []

        def draw_probe_overlay(heading_goal):
            clear_probe_overlay()
            dx, dy = to_canvas_px(rel_x, rel_y)
            r = PROBE_DOT_RADIUS_PX
            self._probe_overlay_ids.append(
                self.canvas.create_oval(dx - r, dy - r, dx + r, dy + r, fill=PROBE_DOT_COLOR, outline=PROBE_DOT_COLOR))

            path_result = altitude_loss_full_physics_path(
                0, 0, values["heading"], rel_x, rel_y, values["altitude_agl_ft"], heading_goal,
                wind_direction=values["wind_direction"], wind_speed=values["wind_speed"])
            if path_result is not None:
                path, _bank_angle_deg = path_result
                nm_points = dubins_path_points((0, 0), values["heading"], (rel_x, rel_y), heading_goal, path)
                flat_px = [coord for x, y in nm_points for coord in to_canvas_px(x, y)]
                self._probe_overlay_ids.append(
                    self.canvas.create_line(*flat_px, fill=PROBE_PATH_COLOR, width=PROBE_PATH_WIDTH_PX))

        def update_level3(*_):
            try:
                heading_goal = float(heading_var.get())
            except ValueError:
                level3_var.set("Level 3 (Full physics): invalid heading")
                return
            loss, reachable = compute_altitude_loss(3, heading_goal=heading_goal, **common_args)
            level3_var.set(f"Level 3 (Full physics): {loss:.0f} ft loss, reachable={reachable}")
            draw_probe_overlay(heading_goal)

        heading_var.trace_add("write", update_level3)
        update_level3()

        def on_close():
            clear_probe_overlay()
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)
        ttk.Button(dialog, text="Close", command=on_close).grid(
            row=heading_row + 2, column=0, columnspan=2, pady=(0, 8))
        heading_entry.focus_set()
        heading_entry.select_range(0, "end")

    def on_randomize_numbers(self):
        if not self.landing_options:
            return
        numbers = [o["number"] for o in self.landing_options]
        random.shuffle(numbers)
        for option, number in zip(self.landing_options, numbers):
            option["number"] = number
        self._rebuild_option_rows()
        self.redraw_canvas()

    def on_open_tags_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Scenario Tags")
        dialog.transient(self.root)
        dialog.geometry("480x600")

        container = ttk.Frame(dialog)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def add_tag_row(parent, tag, indent=0):
            var = tk.BooleanVar(value=tag.tag_id in self.selected_tags)
            tag_vars[tag.tag_id] = var
            ttk.Checkbutton(parent, text=tag.label, variable=var).pack(anchor="w", padx=(indent, 0))
            if tag.description:
                ttk.Label(parent, text=tag.description, foreground="gray", wraplength=420 - indent,
                          justify="left", font=("TkDefaultFont", 8)).pack(anchor="w", padx=(indent + 20, 0), pady=(0, 4))

        tag_vars = {}  # tag_id -> tk.BooleanVar (leaf tags only; headers are not checkable)
        for category, entries in TAG_CATEGORIES.items():
            frame = ttk.LabelFrame(scrollable_frame, text=category, padding=6)
            frame.pack(fill="x", padx=8, pady=6, anchor="n")
            for entry in entries:
                if isinstance(entry, TagHeader):
                    ttk.Label(frame, text=entry.label, font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(4, 0))
                    if entry.description:
                        ttk.Label(frame, text=entry.description, foreground="gray", wraplength=420,
                                  justify="left", font=("TkDefaultFont", 8)).pack(anchor="w", padx=(0, 0), pady=(0, 2))
                    for subtag in entry.subtags:
                        add_tag_row(frame, subtag, indent=20)
                else:
                    add_tag_row(frame, entry)

        def on_done():
            self.selected_tags = {tag_id for tag_id, var in tag_vars.items() if var.get()}
            self.status_var.set(f"{len(self.selected_tags)} tag(s) selected.")
            dialog.destroy()

        button_bar = ttk.Frame(dialog)
        button_bar.pack(fill="x", pady=6)
        ttk.Button(button_bar, text="Done", command=on_done).pack()
        dialog.protocol("WM_DELETE_WINDOW", on_done)

    # ------------------------------------------------------------------ export

    def on_export(self):
        if self._cached_background is None:
            messagebox.showerror("Cannot export", "Render the map before exporting.")
            return
        try:
            values = self._read_float_fields()
        except ValueError:
            messagebox.showerror("Invalid input", "One or more scenario setup fields is not a valid number.")
            return

        os.makedirs(DATASET_DIR, exist_ok=True)
        scenario_dir = os.path.join(DATASET_DIR, time.strftime("scenario_%Y%m%d_%H%M%S"))
        os.makedirs(scenario_dir, exist_ok=True)

        surface = render_full_surface(self._cached_background, values["size_nm"], values["resolution"],
                                       values["heading"], self.landing_options)
        image = surface_to_pil_image(surface)
        image_path = os.path.join(scenario_dir, "viewport.png")
        image.save(image_path)

        ground_truth_index = self.ground_truth_index.get()
        # The viewport is centered on the plane, so center-to-edge is size_nm / 2; compare that
        # to the naive glide radius so a ratio of 1.0 means the frame edge sits exactly at the
        # plane's max naive glide distance.
        naive_glide_distance_nm = (values["altitude_agl_ft"] / FT_PER_NM) * GR_0
        viewport_glide_ratio = (values["size_nm"] / 2) / naive_glide_distance_nm
        scenario = {
            "latitude": self.fields["latitude"].get(),
            "longitude": self.fields["longitude"].get(),
            "heading_deg": values["heading"],
            "viewport_width_nm": values["size_nm"],
            "resolution_px": values["resolution"],
            "altitude_agl_ft": values["altitude_agl_ft"],
            "viewport:glide ratio": viewport_glide_ratio,
            "wind_speed_kt": values["wind_speed"],
            "wind_direction_deg": values["wind_direction"],
            "airspeed_kt": V_GLIDE,
            "weight_lbs": W_MAX,
            "flaps_deg": 0,
            "bank_angle_deg": DEFAULT_BANK_ANGLE_DEG,
            "landing_options": [
                {
                    "number": o["number"], "rel_x_nm": o["rel_x"], "rel_y_nm": o["rel_y"], "heading_deg": o["heading"],
                    **({"small": True} if o.get("small", False) else {}),
                }
                for o in self.landing_options
            ],
            "ground_truth_index": ground_truth_index if ground_truth_index >= 0 else None,
            "answer_explanation": self.text_boxes["answer_explanation"].get("1.0", "end-1c"),
            "prompt_additions": self.text_boxes["prompt_additions"].get("1.0", "end-1c"),
            "image_file": "viewport.png",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        scenario.update(split_tags_by_category(self.selected_tags))
        with open(os.path.join(scenario_dir, "scenario.json"), "w") as f:
            json.dump(scenario, f, indent=2)

        self.status_var.set(f"Exported to {scenario_dir}")
        messagebox.showinfo("Exported", f"Scenario exported to:\n{scenario_dir}")


def main():
    root = tk.Tk()
    app = ScenarioBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
