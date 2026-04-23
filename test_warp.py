import warp as wp
import warp.fem as fem
import numpy as np

wp.init()

DISC_DIAMETER = 0.305       # meters (12-inch LP)
DISC_THICKNESS = 0.0019     # meters (1.9 mm)
MELT_TEMP = 270.0           # °C
MOLD_TEMP = 95.0            # °C
COOLING_TIME = 20.0         # seconds
THERMAL_CONDUCTIVITY = 0.20 # W/m·K (PET)
DENSITY = 1350              # kg/m³
SPECIFIC_HEAT = 1200        # J/kg·K
CRYSTAL_PEAK_TEMP = 160.0
MAX_CRYSTAL_RATE = 0.001
T_G = 75.0

reso_radial = 32
reso_angular = 32
reso_thick = 4

grid = fem.Grid3D(
    res=wp.vec3i(reso_radial, reso_angular, reso_thick),
    bounds_lo=wp.vec3(-DISC_DIAMETER/2, -DISC_DIAMETER/2, 0.0),
    bounds_hi=wp.vec3(DISC_DIAMETER/2, DISC_DIAMETER/2, DISC_THICKNESS)
)
scalar_space = fem.make_polynomial_space(grid, degree=1)

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, k: float):
    return k * wp.dot(fem.grad(u, s), fem.grad(v, s))

@wp.kernel
def crystallinity_step(
    temp: wp.array(dtype=float),
    crystal: wp.array(dtype=float),
    dt: float,
    peak_temp: float,
    max_rate: float,
    Tg: float
):
    tid = wp.tid()
    T = temp[tid]
    chi = crystal[tid]
    if T < Tg:
        rate = 0.0
    else:
        sigma = 40.0
        temp_factor = wp.exp(-(T - peak_temp) * (T - peak_temp) / (2.0 * sigma * sigma))
        rate = max_rate * temp_factor * (1.0 - chi)
    crystal[tid] = chi + rate * dt

@wp.kernel
def update_temp_kernel(
    temp: wp.array(dtype=float),
    residual: wp.array(dtype=float),
    M_diag: float,
    dt: float,
    mold_temp: float,
    reso_radial: int,
    reso_angular: int,
    reso_thick: int
):
    tid = wp.tid()
    layer_size = (reso_radial + 1) * (reso_angular + 1)
    z_idx = tid // layer_size
    if z_idx == 0 or z_idx == reso_thick:
        temp[tid] = mold_temp
    else:
        temp[tid] -= dt * residual[tid] / M_diag

def solve_cooling(n_steps: int = 2):
    dt = COOLING_TIME / n_steps
    temp_array = wp.full(scalar_space.node_count(), MELT_TEMP, dtype=float)
    crystal_array = wp.zeros(scalar_space.node_count(), dtype=float)
    
    domain = fem.Cells(geometry=grid)
    test = fem.make_test(space=scalar_space, domain=domain)
    
    node_vol = float((DISC_DIAMETER/reso_radial) * (DISC_DIAMETER/reso_angular) * (DISC_THICKNESS/reso_thick))
    M_diag_val = float(DENSITY * SPECIFIC_HEAT * node_vol)
    
    temp_field = scalar_space.make_field() 
    
    for step in range(n_steps):
        temp_field.dof_values = temp_array
        # In Warp FEMA, make_test / integrate works similarly.
        residual = fem.integrate(
            diffusion_form,
            fields={"u": temp_field, "v": test},
            values={"k": THERMAL_CONDUCTIVITY}
        )
        wp.launch(update_temp_kernel, dim=scalar_space.node_count(), 
                  inputs=[temp_array, residual, M_diag_val, dt, MOLD_TEMP, reso_radial, reso_angular, reso_thick])
        wp.launch(crystallinity_step, dim=scalar_space.node_count(), 
                  inputs=[temp_array, crystal_array, dt, CRYSTAL_PEAK_TEMP, MAX_CRYSTAL_RATE, T_G])
        
    return temp_array, crystal_array

temp, cryst = solve_cooling(n_steps=2)
print("Finished. Temp max:", float(wp.max(temp).numpy()))
