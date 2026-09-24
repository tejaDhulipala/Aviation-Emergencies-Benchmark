"""Physical constants and reference values used across the simulation.
Only real-world physical/aircraft/geographic quantities belong here —
app-specific tuning knobs (UI sizes, colors, thresholds) stay in their
own files.
"""

# --- Unit conversions ---
FT_PER_NM = 6076.12               # feet per nautical mile
NM_TO_M = 1852.0                  # meters per nautical mile (international definition)
M_PER_DEG_LAT = 111320.0          # meters per degree of latitude (spherical approximation)
WEB_MERCATOR_RES_ZOOM0 = 156543.03392  # meters/pixel at the equator, Web Mercator zoom 0

# --- Standard atmosphere ---
RHO_0 = 0.002378                  # sea-level standard density (slug/ft^3)
TEMP_0_R = 518.67                 # sea-level standard temperature (deg Rankine)
LAPSE_RATE = 0.00356              # temperature lapse rate (deg Rankine / ft)
F_TO_R_OFFSET = 459.67            # Fahrenheit -> Rankine offset
BAROMETRIC_EXPONENT = 4.256       # standard-atmosphere density exponent

# --- Cessna 172 performance ---
GR_0 = 9.0                        # baseline best-glide ratio
W_MAX = 2400                      # max gross weight (lbs)
V_GLIDE = 65                      # best glide speed @ max gross weight (KIAS)
ENGINE_RPM_DEFAULT = 2300         # default cruise RPM
VALID_FLAP_SETTINGS = [0, 10, 20, 30]
MAX_FLAPS = 30

G_NM_HR_2 = 68626.675878   # Small G in NM/HR^2
OBSTACLE_CLEARANCE_FT = 50        # height of obstacle landing distances are measured over
MIN_BANK_ANGLE_DEG = 5            # shallowest commandable turn bank (below this, turn radius blows up)
MAX_BANK_ANGLE_DEG = 60           # steepest commandable turn bank

MIN_AIRSPEED_KT = 35              # below this the Cessna 172 stalls
MAX_AIRSPEED_KT = 160             # above this exceeds never-exceed speed

# Flap effect on glide ratio
FLAP_GR_PENALTY_PER_10_DEG = 1     # glide ratio lost per 10 degrees of flaps

# --- Cessna 172 landing performance (distance over a 50ft obstacle, by temperature) ---
LANDING_DIST_50FT_BY_TEMP_C = {
    0: 1320,
    10: 1350,
    20: 1385,
    30: 1420,
    40: 1450,
}
HEADWIND_REDUCTION_PER_KT = 9     # distance decreases 10% per this many knots of headwind
TAILWIND_INCREASE_PER_KT = 2      # distance increases 10% per this many knots of tailwind
TAILWIND_MAX_DEMONSTRATED_KT = 10
TAILWIND_MAX_FACTOR = 1.5         # flat multiplier applied beyond max demonstrated tailwind
NO_FLAP_DISTANCE_INCREASE_PCT = 35  # % increase in landing distance with flaps up vs full flaps

# --- Map tile / Web Mercator imagery ---
TILE_SIZE = 256                   # px, standard slippy-map tile size
MAX_ZOOM = 19
