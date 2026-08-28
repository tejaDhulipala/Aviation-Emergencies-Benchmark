import math
import io
import requests
from PIL import Image
from utils.constants import TILE_SIZE, NM_TO_M, MAX_ZOOM, M_PER_DEG_LAT, WEB_MERCATOR_RES_ZOOM0

TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
HEADERS = {"User-Agent": "emergency-decision-making-sim/1.0"}


def offset_latlon(lat, lon, dx_nm, dy_nm):
    """Offset (lat, lon) by dx_nm east and dy_nm north using a flat-earth approximation."""
    lat_deg_per_m = 1 / M_PER_DEG_LAT
    lon_deg_per_m = 1 / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    dx_m = dx_nm * NM_TO_M
    dy_m = dy_nm * NM_TO_M
    return lat + dy_m * lat_deg_per_m, lon + dx_m * lon_deg_per_m


class SatelliteMap:
    def __init__(self):
        self.session = requests.Session()
        self.tile_cache = {}

    def get_image(self, lat, lon, size_nm, out_size=512):
        zoom = self._pick_zoom(lat, size_nm, out_size)
        lat_deg_per_m = 1 / M_PER_DEG_LAT
        lon_deg_per_m = 1 / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
        half_size_m = (size_nm * NM_TO_M) / 2
        lat_north = lat + half_size_m * lat_deg_per_m
        lat_south = lat - half_size_m * lat_deg_per_m
        lon_west = lon - half_size_m * lon_deg_per_m
        lon_east = lon + half_size_m * lon_deg_per_m

        x_min, y_min = self._deg2num(lat_north, lon_west, zoom)
        x_max, y_max = self._deg2num(lat_south, lon_east, zoom)
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
        if tile_count > 64:
            raise ValueError(f"Refusing to fetch {tile_count} tiles at zoom {zoom} (bbox too large for chosen zoom)")

        canvas = Image.new("RGB", ((x_max - x_min + 1) * TILE_SIZE, (y_max - y_min + 1) * TILE_SIZE))
        for xt in range(x_min, x_max + 1):
            for yt in range(y_min, y_max + 1):
                tile = self._fetch_tile(zoom, xt, yt)
                canvas.paste(tile, ((xt - x_min) * TILE_SIZE, (yt - y_min) * TILE_SIZE))

        canvas_lat_north, canvas_lon_west = self._num2deg(x_min, y_min, zoom)
        canvas_lat_south, canvas_lon_east = self._num2deg(x_max + 1, y_max + 1, zoom)

        px_per_lon = canvas.width / (canvas_lon_east - canvas_lon_west)
        px_per_lat = canvas.height / (canvas_lat_south - canvas_lat_north)  # negative, lat decreases downward

        left = (lon_west - canvas_lon_west) * px_per_lon
        right = (lon_east - canvas_lon_west) * px_per_lon
        top = (lat_north - canvas_lat_north) * px_per_lat
        bottom = (lat_south - canvas_lat_north) * px_per_lat

        cropped = canvas.crop((int(left), int(top), int(right), int(bottom)))
        return cropped.resize((out_size, out_size), Image.LANCZOS)

    def _pick_zoom(self, lat, size_nm, out_size):
        size_m = size_nm * NM_TO_M
        target_res = size_m / out_size  # meters/pixel needed
        for zoom in range(0, MAX_ZOOM + 1):
            res = WEB_MERCATOR_RES_ZOOM0 * math.cos(math.radians(lat)) / (2 ** zoom)
            if res <= target_res:
                return zoom
        return MAX_ZOOM

    def _deg2num(self, lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        xtile = max(0, min(int(n) - 1, xtile))
        ytile = max(0, min(int(n) - 1, ytile))
        return xtile, ytile

    def _num2deg(self, xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def _fetch_tile(self, zoom, xtile, ytile):
        key = (zoom, xtile, ytile)
        if key in self.tile_cache:
            return self.tile_cache[key]
        url = TILE_URL.format(z=zoom, x=xtile, y=ytile)
        resp = self.session.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
        self.tile_cache[key] = tile
        return tile


if __name__ == "__main__":
    smap = SatelliteMap()
    tests = [
        ("wide", 33.9425, -118.4081, 8.0),   # LAX area, wide view
        ("tight", 33.9425, -118.4081, 1.0),  # same center, zoomed in
    ]
    tests = [(f"image-{i}", 33.9425, -118.4081, i) for i in range(1, 9)]
    for name, lat, lon, size in tests:
        img = smap.get_image(lat, lon, size)
        out_path = f"map-api-tests/map_test_{name}.png"
        img.save(out_path)
        print(f"[{name}] center=({lat},{lon}) {size}x{size}nm -> {out_path} size={img.size}")
