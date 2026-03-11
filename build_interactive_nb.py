import json

cells_text = [
"""# --------------------------------------------------------------
# 1️⃣  Install required packages
# --------------------------------------------------------------
!pip install -q warp-lang matplotlib plotly ipywidgets""",

"""# --------------------------------------------------------------
# 2️⃣  Imports & Warp initialisation
# --------------------------------------------------------------
import math
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# Silence Warp output
wp.config.quiet = True
wp.init()

DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
print(f"Running on: {DEVICE}")""",

"""# --------------------------------------------------------------
# 3️⃣  Data structures (Warp structs)
# --------------------------------------------------------------
@wp.struct
class DiscConfig:
    \"\"\"Geometry & material properties of the disc.\"\"\"
    radius:    float   # outer radius (m)
    thickness: float   # disc thickness (m)
    dr:        float   # radial grid spacing (m)
    dz:        float   # axial grid spacing (m)
    k:         float   # thermal conductivity (W/m·K)
    rho:       float   # density (kg/m³)
    cp:        float   # specific heat capacity (J/kg·K)
    T_g:       float   # glass‑transition temperature (K)
    T_m:       float   # melting temperature (K)
    nx:        int     # radial cells
    nz:        int     # axial cells

@wp.struct
class CoolingParams:
    \"\"\"Process parameters for a single simulation run.\"\"\"
    T_init:          float   # melt temperature (K)
    T_mold:          float   # mould wall temperature (K)
    dt:              float   # time step (s)
    total_time:      float   # total simulation time (s)
    avrami_k0:       float   # Avrami rate pre‑factor (1/s)
    avrami_n:        float   # Avrami exponent
    chi_max:         float   # max crystallinity (0‑1)
    warp_temp_coeff: float   # weight of thermal gradient
    warp_chi_coeff:  float   # weight of crystallinity asymmetry""",

"""# --------------------------------------------------------------
# 4️⃣  Helper functions & kernels
# --------------------------------------------------------------
@wp.func
def idx(i: int, j: int, nz: int) -> int:
    \"\"\"Row‑major flat index for the (nx × nz) grid.\"\"\"
    return i * nz + j

@wp.func
def clamp_i(x: int, lo: int, hi: int) -> int:
    return wp.min(wp.max(x, lo), hi)

@wp.kernel
def init_temperature(T: wp.array(dtype=wp.float32),
                     params: CoolingParams):
    tid = wp.tid()
    T[tid] = wp.float32(params.T_init)

@wp.kernel
def init_scalar(a: wp.array(dtype=wp.float32),
                value: float):
    tid = wp.tid()
    a[tid] = wp.float32(value)

@wp.kernel
def step_temperature(T_in:  wp.array(dtype=wp.float32),
                     T_out: wp.array(dtype=wp.float32),
                     config: DiscConfig,
                     params: CoolingParams):
    \"\"\"Explicit finite‑difference heat diffusion in cylindrical coordinates.\"\"\"
    tid = wp.tid()
    i = tid // config.nz          # radial index
    j = tid %  config.nz          # axial index

    alpha = config.k / (config.rho * config.cp)
    dr = config.dr
    dz = config.dz
    r  = (float(i) + 0.5) * dr

    # Dirichlet BC on top/bottom mould walls
    if j == 0 or j == config.nz - 1:
        T_out[tid] = wp.float32(params.T_mold)
        return

    # Neumann BC (zero‑flux) on axis & outer radius
    im = clamp_i(i - 1, 0, config.nx - 1)
    ip = clamp_i(i + 1, 0, config.nx - 1)
    if i == 0:      im = 1
    if i == config.nx - 1:
        ip = config.nx - 2

    jm = j - 1
    jp = j + 1

    Tc  = T_in[idx(i,  j,  config.nz)]
    Trm = T_in[idx(im, j,  config.nz)]
    Trp = T_in[idx(ip, j,  config.nz)]
    Tzm = T_in[idx(i,  jm, config.nz)]
    Tzp = T_in[idx(i,  jp, config.nz)]

    d2Tdr2      = (Trp - 2.0 * Tc + Trm) / (dr * dr)
    dTdr_over_r = 0.0
    if i > 0:
        dTdr_over_r = (Trp - Trm) / (2.0 * dr * r)
    d2Tdz2 = (Tzp - 2.0 * Tc + Tzm) / (dz * dz)

    lap = d2Tdr2 + dTdr_over_r + d2Tdz2
    Tnew = Tc + params.dt * alpha * lap
    T_out[tid] = wp.float32(Tnew)

@wp.kernel
def update_crystallinity(T:       wp.array(dtype=wp.float32),
                         chi_in:  wp.array(dtype=wp.float32),
                         chi_out: wp.array(dtype=wp.float32),
                         config:  DiscConfig,
                         params:  CoolingParams):
    \"\"\"Simple Avrami‑style crystallisation kinetics.\"\"\"
    tid = wp.tid()
    temp = T[tid]
    chi  = chi_in[tid]

    if config.T_g < temp < config.T_m:
        x = (temp - config.T_g) / (config.T_m - config.T_g)
        x = wp.max(0.0, wp.min(1.0, x))
        rate = params.avrami_k0 * x * (1.0 - chi / params.chi_max)
        chi = chi + params.dt * params.avrami_n * rate
        chi = wp.max(0.0, wp.min(params.chi_max, chi))

    chi_out[tid] = wp.float32(chi)

@wp.kernel
def compute_warp_risk(T:         wp.array(dtype=wp.float32),
                      chi:       wp.array(dtype=wp.float32),
                      warp_risk: wp.array(dtype=wp.float32),
                      config:    DiscConfig,
                      params:    CoolingParams):
    \"\"\"Score each radial position (mid‑plane only).\"\"\"
    tid = wp.tid()
    i   = tid // config.nz
    j   = tid %  config.nz

    if j != config.nz // 2:
        return

    top = T[idx(i, 0,              config.nz)]
    bot = T[idx(i, config.nz - 1,  config.nz)]
    mid = T[idx(i, j,              config.nz)]

    chi_top = chi[idx(i, 1,              config.nz)]
    chi_bot = chi[idx(i, config.nz - 2,  config.nz)]

    dT_thickness = wp.abs(top - bot)
    dT_mid       = wp.abs(mid - 0.5 * (top + bot))
    dchi         = wp.abs(chi_top - chi_bot)

    risk = (params.warp_temp_coeff * (dT_thickness + dT_mid) +
            params.warp_chi_coeff  * dchi)

    warp_risk[i] = wp.float32(risk)""",

"""# --------------------------------------------------------------
# 5️⃣  Simulation class (stores temperature history for animation)
# --------------------------------------------------------------
class ArtifexCoolingSim:
    \"\"\"Manages GPU arrays and runs the cooling simulation.\"\"\"
    def __init__(self, config: DiscConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.n      = config.nx * config.nz

        self.T_a       = wp.zeros(self.n, dtype=wp.float32, device=device)
        self.T_b       = wp.zeros(self.n, dtype=wp.float32, device=device)
        self.chi_a     = wp.zeros(self.n, dtype=wp.float32, device=device)
        self.chi_b     = wp.zeros(self.n, dtype=wp.float32, device=device)
        self.warp_risk = wp.zeros(config.nx, dtype=wp.float32, device=device)

        # For interactive visualisation
        self.history_T = []

    def simulate_cooling(self, params: CoolingParams,
                         record_every: int = 10) -> dict:
        \"\"\"Run the simulation; optionally record temperature every N steps.\"\"\"
        wp.launch(init_temperature, dim=self.n,
                  inputs=[self.T_a, params], device=self.device)
        wp.launch(init_scalar, dim=self.n,
                  inputs=[self.chi_a, 0.0], device=self.device)

        steps = int(params.total_time / params.dt)

        for step in range(steps):
            # Diffusion
            wp.launch(step_temperature, dim=self.n,
                      inputs=[self.T_a, self.T_b, self.config, params],
                      device=self.device)
            self.T_a, self.T_b = self.T_b, self.T_a

            # Crystallinity
            wp.launch(update_crystallinity, dim=self.n,
                      inputs=[self.T_a, self.chi_a, self.chi_b,
                              self.config, params],
                      device=self.device)
            self.chi_a, self.chi_b = self.chi_b, self.chi_a

            # Record temperature field for later visualisation
            if (step % record_every) == 0:
                self.history_T.append(self.T_a.numpy().reshape(
                    (self.config.nx, self.config.nz)))

        # Final warp‑risk
        wp.launch(compute_warp_risk, dim=self.n,
                  inputs=[self.T_a, self.chi_a, self.warp_risk,
                          self.config, params],
                  device=self.device)

        # Pull final fields back to NumPy
        T_np   = self.T_a.numpy().reshape((self.config.nx, self.config.nz))
        chi_np = self.chi_a.numpy().reshape((self.config.nx, self.config.nz))
        risk_np = self.warp_risk.numpy()

        max_delta_t    = float(np.max(np.abs(T_np[:, 0] - T_np[:, -1])))
        avg_chi_groove = float(np.mean(chi_np[:, 1]))
        max_warp_risk  = float(np.max(risk_np))
        is_ok = avg_chi_groove < 0.15 and max_warp_risk < 15.0

        return {
            "max_delta_t":       max_delta_t,
            "avg_chi_groove":    avg_chi_groove,
            "max_warp_risk":     max_warp_risk,
            "is_ok":             is_ok,
            "final_T_field":     T_np,
            "final_chi_field":   chi_np,
            "warp_risk_radial":  risk_np,
        }""",

"""# --------------------------------------------------------------
# 6️⃣  Geometry, material & stable time‑step
# --------------------------------------------------------------
# ---- Geometry -------------------------------------------------
RADIUS_MM    = 60.0   # outer radius (mm)
THICKNESS_MM = 1.2    # disc thickness (mm)
NX           = 60     # radial cells
NZ           = 20     # axial cells (≥4)

radius    = RADIUS_MM    * 1e-3   # m
thickness = THICKNESS_MM * 1e-3   # m
dr        = radius    / NX
dz        = thickness / NZ

# ---- Material -------------------------------------------------
config = DiscConfig()
config.radius    = radius
config.thickness = thickness
config.dr        = dr
config.dz        = dz
config.k         = 0.29      # W/m·K
config.rho       = 1360.0    # kg/m³
config.cp        = 1250.0    # J/kg·K
config.T_g       = 348.0     # K  (~75 °C)
config.T_m       = 533.0     # K  (~260 °C)
config.nx        = NX
config.nz        = NZ

# ---- Stability (Fourier) ---------------------------------------
alpha   = config.k / (config.rho * config.cp)
dt_max  = 0.4 * min(dr, dz) * 2 / alpha   # explicit stability limit
# Use a *more* conservative dt for higher accuracy

print(f"Thermal diffusivity α = {alpha:.2e} m²/s")
print(f"Stability limit dt_max = {dt_max*1e3:.3f} ms")""",

"""# --------------------------------------------------------------
# 7️⃣  Process parameters (feel free to tweak)
# --------------------------------------------------------------
params = CoolingParams()
params.T_init            = 553.0          # K  (melt inlet, ~280 °C)
params.T_mold            = 298.0          # K  (mould wall, ~25 °C)
params.total_time        = 8.0            # s
params.avrami_k0         = 0.005          # 1/s (placeholder)
params.avrami_n          = 2.5            # dimensionless
params.chi_max           = 0.35
params.warp_temp_coeff   = 0.2
params.warp_chi_coeff    = 50.0

params.dt = dt_max * 0.6   # 60 % of the limit
steps = int(params.total_time / params.dt)
print(f"Simulation steps: {steps:,}   (dt = {params.dt*1e3:.3f} ms, total = {params.total_time} s)")""",

"""# --------------------------------------------------------------
# 8️⃣  Run the simulation (record every 5 steps for animation)
# --------------------------------------------------------------
sim = ArtifexCoolingSim(config, device=DEVICE)
results = sim.simulate_cooling(params, record_every=5)

print("\\n=== Results ===")
print(f"  Max top‑bottom ΔT         : {results['max_delta_t']:.2f} K")
print(f"  Mean groove crystallinity : {results['avg_chi_groove']:.4f}")
print(f"  Max warp‑risk score       : {results['max_warp_risk']:.3f}")
print(f"  Quality gate passed       : {results['is_ok']}")""",

"""# --------------------------------------------------------------
# 9️⃣  2‑D Matplotlib visualisations (temperature, crystallinity, risk)
# --------------------------------------------------------------
T_field   = results["final_T_field"]    # (nx, nz)
chi_field = results["final_chi_field"]  # (nx, nz)
risk      = results["warp_risk_radial"] # (nx,)

r_mm = np.linspace(dr/2, radius - dr/2, NX) * 1e3
z_mm = np.linspace(dz/2, thickness - dz/2, NZ) * 1e3
R, Z = np.meshgrid(r_mm, z_mm, indexing="ij")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Temperature field
ax = axes[0]
im = ax.pcolormesh(R, Z, T_field, cmap="inferno", shading="auto")
fig.colorbar(im, ax=ax, label="Temperature (K)")
ax.set_xlabel("Radius (mm)")
ax.set_ylabel("Thickness (mm)")
ax.set_title(f"Final temperature field (t = {params.total_time} s)")

# Crystallinity field
ax = axes[1]
im = ax.pcolormesh(R, Z, chi_field, cmap="viridis", shading="auto",
                   vmin=0, vmax=params.chi_max)
fig.colorbar(im, ax=ax, label="Crystallinity χ")
ax.set_xlabel("Radius (mm)")
ax.set_ylabel("Thickness (mm)")
ax.set_title("Final crystallinity field")

# Warp‑risk radial profile
ax = axes[2]
ax.plot(r_mm, risk, color="crimson", linewidth=2)
ax.axhline(15.0, color="gray", linestyle="--", linewidth=1, label="Quality threshold")
ax.fill_between(r_mm, risk, 15.0,
                where=(risk > 15.0), color="crimson", alpha=0.2, label="At‑risk zone")
ax.set_xlabel("Radius (mm)")
ax.set_ylabel("Warp‑risk score")
ax.set_title("Radial warp‑risk profile")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()""",

"""# --------------------------------------------------------------
# 🔟  2‑D radial profiles (temperature & crystallinity)
# --------------------------------------------------------------
mid = NZ // 2
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Temperature profiles
ax = axes[0]
ax.plot(r_mm, T_field[:, mid] - 273.15, label="Mid‑plane", color="tab:orange")
ax.plot(r_mm, T_field[:, 0]  - 273.15, label="Top surface", color="tab:blue", linestyle="--")
ax.plot(r_mm, T_field[:, -1] - 273.15, label="Bottom surface", color="tab:green", linestyle=":")
ax.set_xlabel("Radius (mm)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Radial temperature profiles")
ax.legend()
ax.grid(alpha=0.3)

# Crystallinity profiles
ax = axes[1]
ax.plot(r_mm, chi_field[:, mid],  label="Mid‑plane",     color="tab:purple")
ax.plot(r_mm, chi_field[:, 1],    label="Near top wall", color="tab:blue",  linestyle="--")
ax.plot(r_mm, chi_field[:, -2],   label="Near bottom wall", color="tab:green", linestyle=":")
ax.axhline(0.15, color="gray", linestyle="--", linewidth=1, label="χ threshold (0.15)")
ax.set_xlabel("Radius (mm)")
ax.set_ylabel("Crystallinity χ")
ax.set_title("Radial crystallinity profiles")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()""",

"""# --------------------------------------------------------------
# 1️⃣1️⃣  3‑D Plotly visualisation of the final temperature field
# --------------------------------------------------------------
R, Z = np.meshgrid(r_mm, z_mm, indexing="ij")
temp_surface = go.Surface(
    x=R, y=Z, z=results["final_T_field"],
    colorscale="Turbo",
    colorbar=dict(title="Temperature (K)"),
    showscale=True
)

fig = go.Figure(data=[temp_surface])
fig.update_layout(
    title="Final Temperature Field (3‑D Surface)",
    scene=dict(
        xaxis_title="Radius (mm)",
        yaxis_title="Thickness (mm)",
        zaxis_title="Temperature (K)"
    ),
    width=800,
    height=600
)
fig.show()""",

"""# --------------------------------------------------------------
# 1️⃣2️⃣  3‑D Plotly visualisation of the final crystallinity field
# --------------------------------------------------------------
chi_surface = go.Surface(
    x=R, y=Z, z=results["final_chi_field"],
    colorscale="Viridis",
    cmin=0, cmax=params.chi_max,
    colorbar=dict(title="Crystallinity χ"),
    showscale=True
)

fig = go.Figure(data=[chi_surface])
fig.update_layout(
    title="Final Crystallinity Field (3‑D Surface)",
    scene=dict(
        xaxis_title="Radius (mm)",
        yaxis_title="Thickness (mm)",
        zaxis_title="Crystallinity χ"
    ),
    width=800,
    height=600
)
fig.show()""",

"""# --------------------------------------------------------------
# 1️⃣3️⃣  Interactive temperature evolution slider (Plotly)
# --------------------------------------------------------------
# `sim.history_T` contains temperature snapshots (recorded every 5 steps)
temp_history = sim.history_T
num_frames = len(temp_history)

def make_surface(frame_idx):
    return go.Surface(
        x=R, y=Z, z=temp_history[frame_idx],
        colorscale="Turbo",
        cmin=np.min(temp_history),
        cmax=np.max(temp_history),
        showscale=False
    )

# Initial figure
fig = go.Figure(data=[make_surface(0)])

# Slider steps
steps = []
for i in range(num_frames):
    step = dict(
        method="update",
        args=[{"z": [temp_history[i]]}],
        label=f"t={i*params.dt*5:.2f}s"   # 5 steps between recordings
    )
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    title="Temperature Evolution (use slider)",
    scene=dict(
        xaxis_title="Radius (mm)",
        yaxis_title="Thickness (mm)",
        zaxis_title="Temperature (K)"
    ),
    sliders=sliders,
    width=800,
    height=600
)
fig.show()""",

"""# --------------------------------------------------------------
# 1️⃣4️⃣  Parameter sweep: mould temperature vs. warp‑risk
# --------------------------------------------------------------
T_mold_range = np.linspace(288, 353, 12)   # 15 °C → 80 °C (K)
sweep_results = []

for T_mold_K in T_mold_range:
    p = CoolingParams()
    p.T_init         = params.T_init
    p.T_mold         = float(T_mold_K)
    p.dt             = params.dt
    p.total_time     = params.total_time
    p.avrami_k0      = params.avrami_k0
    p.avrami_n       = params.avrami_n
    p.chi_max        = params.chi_max
    p.warp_temp_coeff = params.warp_temp_coeff
    p.warp_chi_coeff  = params.warp_chi_coeff

    s   = ArtifexCoolingSim(config, device=DEVICE)
    res = s.simulate_cooling(p)

    sweep_results.append({
        "T_mold_C":    T_mold_K - 273.15,
        "max_risk":    res["max_warp_risk"],
        "avg_chi":     res["avg_chi_groove"],
        "is_ok":       res["is_ok"],
    })

T_mold_C = [r["T_mold_C"] for r in sweep_results]
max_risk  = [r["max_risk"]  for r in sweep_results]
avg_chi   = [r["avg_chi"]   for r in sweep_results]
colors    = ["green" if r["is_ok"] else "red" for r in sweep_results]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Peak warp‑risk
ax = axes[0]
ax.bar(T_mold_C, max_risk, width=3.5, color=colors,
       edgecolor="black", linewidth=0.5)
ax.axhline(15.0, color="gray", linestyle="--", label="Quality threshold")
ax.set_xlabel("Mould temperature (°C)")
ax.set_ylabel("Max warp‑risk score")
ax.set_title("Peak warp‑risk vs. mould temperature")
ax.legend()

# Mean groove crystallinity
ax = axes[1]
ax.bar(T_mold_C, avg_chi, width=3.5, color=colors,
       edgecolor="black", linewidth=0.5)
ax.axhline(0.15, color="gray", linestyle="--", label="χ threshold")
ax.set_xlabel("Mould temperature (°C)")
ax.set_ylabel("Mean groove crystallinity χ")
ax.set_title("Groove crystallinity vs. mould temperature")
ax.legend()

plt.tight_layout()
plt.show()""",

"""| Upgrade | Why it matters |
|---|---|
| **Nakamura kinetics from DSC data** | Replace placeholder Avrami constants with a physics‑based model. |
| **Thermoelastic / eigenstrain plate solver** | Turn `warp_risk` into a real out‑of‑plane deflection (mm). |
| **Asymmetric mould BCs** | Real injection tools often have different cooling on top vs. bottom. |
| **Isaac Sim USD export** | Export the fields to a USD mesh for high‑fidelity visualisation in Omniverse. |
| **Adaptive time‑step control** | Auto‑adjust `dt` to stay safely within the Fourier stability limit. |
"""
]


nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {"id": "badge"},
   "source": [
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tuesdaythe13th/artifex-warp-newton-sim/blob/main/artifex_colab_interactive.ipynb)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "intro"},
   "source": [
    "# ARTIFEX DISC COOLING (Interactive Dashboard)\n",
    "Run explicit finite element calculations natively through Warp kernels for extreme GPU performance."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

for i, code in enumerate(cells_text):
    cell_type = "markdown" if i == len(cells_text) - 1 else "code"
    
    nb["cells"].append({
        "cell_type": cell_type,
        "metadata": {"id": f"cell_{i}"},
        "source": [line + "\n" if j < len(code.split("\n")) - 1 else line for j, line in enumerate(code.split("\n"))],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    })

with open("artifex_colab_interactive.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
