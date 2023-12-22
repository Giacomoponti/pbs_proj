import taichi as ti
import numpy as np
import math

from multigridsolver import MultigridPCGPoissonSolver

ti.init(default_fp=ti.f32, arch=ti.x64, kernel_profiler=True)

grid_width = 256
grid_height = grid_width
window_width_resolution = 512
window_height_resolution = window_width_resolution

gravity = -9.81
flip_viscosity = 0.0

SOLID = 2
AIR = 1
FLUID = 0

shoot_count = 20
reserved_particles = 20

dt = 0.1
frame_counter = 0


@ti.func
def clamp(x, a, b):
    return max(a, min(b, x))

@ti.func
def sample(data, u, v, ox, oy, grid_width, grid_height):
    s, t = u - ox, v - oy
    i, j = clamp(int(s), 0, grid_width - 1), clamp(int(t), 0, grid_height - 1)
    ip, jp = clamp(i + 1, 0, grid_width - 1), clamp(j + 1, 0, grid_height - 1)
    s, t = clamp(s - i, 0.0, 1.0), clamp(t - j, 0.0, 1.0)
    return \
        (data[i, j] * (1 - s) + data[ip, j] * s) * (1 - t) + \
        (data[i, jp] * (1 - s) + data[ip, jp] * s) * t

@ti.func
def splat(data, weights, f, u, v, ox, oy, grid_width, grid_height):
    s, t = u - ox, v - oy
    i, j = clamp(int(s), 0, grid_width - 1), clamp(int(t), 0, grid_height - 1)
    ip, jp = clamp(i + 1, 0, grid_width - 1), clamp(j + 1, 0, grid_height - 1)
    s, t = clamp(s - i, 0.0, 1.0), clamp(t - j, 0.0, 1.0)
    data[i, j] += f * (1 - s) * (1 - t)
    data[ip, j] += f * (s) * (1 - t)
    data[i, jp] += f * (1 - s) * (t)
    data[ip, jp] += f * (s) * (t)
    weights[i, j] += (1 - s) * (1 - t)
    weights[ip, j] += (s) * (1 - t)
    weights[i, jp] += (1 - s) * (t)
    weights[ip, jp] += (s) * (t)

pressure = ti.field(dtype=ti.f32, shape=(grid_width, grid_height))
new_pressure = ti.field(dtype=ti.f32, shape=(grid_width, grid_height))
divergence = ti.field(dtype=ti.f32, shape=(grid_width, grid_height))
marker_field = ti.field(dtype=ti.i32, shape=(grid_width, grid_height))

velocity_field_ux = ti.field(dtype=ti.f32, shape=(grid_width + 1, grid_height))
velocity_field_uy = ti.field(dtype=ti.f32, shape=(grid_width, grid_height + 1))
new_velocity_field_ux = ti.field(dtype=ti.f32, shape=(grid_width + 1, grid_height))
new_velocity_field_uy = ti.field(dtype=ti.f32, shape=(grid_width, grid_height + 1))
saved_velocity_field_ux = ti.field(dtype=ti.f32, shape=(grid_width + 1, grid_height))
saved_velocity_field_uy = ti.field(dtype=ti.f32, shape=(grid_width, grid_height + 1))

particle_position_field = ti.Vector.field(2, dtype=ti.f32, shape=(grid_width * 2, grid_height * 2))
particle_velocity_field = ti.Vector.field(2, dtype=ti.f32, shape=(grid_width * 2, grid_height * 2))
particle_flag_field = ti.field(dtype=ti.i32, shape=(grid_width * 2, grid_height * 2))

validity_field = ti.field(dtype=ti.i32, shape=(grid_width + 1, grid_height + 1))
new_validity_field = ti.field(dtype=ti.i32, shape=(grid_width + 1, grid_height + 1))

color_field_to_display = ti.Vector.field(3, dtype=ti.f32, shape=(window_width_resolution, window_height_resolution))

next_particle_insert_counter = ti.field(dtype=ti.i32, shape=())

ps = MultigridPCGPoissonSolver(marker_field, grid_width, grid_height)

@ti.kernel
def iterative_pressure_update(dt: ti.f32) -> ti.f32:
    residual = 0.0

    for x, y in pressure:
        if FLUID == marker_field[x, y]:
            b = -divergence[x, y] / dt
            non_solid_neighbors = 0
            # Check for non-solid cells in x-Direction
            if x != 1 and x != grid_width - 1:
                non_solid_neighbors += 2
            elif x == 1:
                non_solid_neighbors += 1
            else:
                non_solid_neighbors += 1

            # Check for non-solid cells in y-Direction
            if y != 1 and y != grid_height - 1:
                non_solid_neighbors += 2
            elif y == 1:
                non_solid_neighbors += 1
            else:
                non_solid_neighbors += 1

            new_pressure[x, y] = (
                b
                + pressure[x - 1, y]
                + pressure[x + 1, y]
                + pressure[x, y - 1]
                + pressure[x, y + 1]
            ) / 4
        else:
            new_pressure[x, y] = pressure[x, y]

    # Compute residual
    for x, y in pressure:
        if FLUID == marker_field[x, y]:

            b = -divergence[x, y] / dt

            cell_residual = 0.0

            non_solid_neighbors = 0
            # Check for non-solid cells in x-Direction
            if x != 1 and x != grid_width - 1:
                non_solid_neighbors += 2
            elif x == 1:
                non_solid_neighbors += 1
            else:
                non_solid_neighbors += 1

            # Check for non-solid cells in y-Direction
            if y != 1 and y != grid_height - 1:
                non_solid_neighbors += 2
            elif y == 1:
                non_solid_neighbors += 1
            else:
                non_solid_neighbors += 1

            cell_residual = b - (
                4 * new_pressure[x, y]
                - new_pressure[x - 1, y]
                - new_pressure[x + 1, y]
                - new_pressure[x, y - 1]
                - new_pressure[x, y + 1]
            )

            residual += cell_residual * cell_residual

    residual = ti.lang.ops.sqrt(residual)
    residual /= (grid_width - 2) * (grid_height - 2)
    return residual

def solve(dt: ti.f32):
    tolerance = 1e-5
    residual = tolerance + 1
    for i in range(300):
        if residual <= tolerance:
            return
        else:
            residual = iterative_pressure_update(dt)
            pressure.copy_from(new_pressure)

@ti.func
def is_marker_a_fluid(i, j):
    return i >= 0 and i < grid_width and j >= 0 and j < grid_height and FLUID == marker_field[i, j]


@ti.func
def is_marker_a_solid(i, j):
    return i < 0 or i >= grid_width or j < 0 or j >= grid_height or SOLID == marker_field[i, j]


@ti.func
def is_marker_air(i, j):
    return i >= 0 and i < grid_width and j >= 0 and j < grid_height and AIR == marker_field[i, j]


@ti.func
def interpolate_velocity(pos, ux, uy):
    _ux = sample(ux, pos.x, pos.y, 0.0, 0.5, grid_width + 1, grid_height)
    _uy = sample(uy, pos.x, pos.y, 0.5, 0.0, grid_width, grid_height + 1)
    return ti.Vector([_ux, _uy])


@ti.kernel
def advect_markers(dt: ti.f32):
    for i, j in particle_position_field:
        if 1 == particle_flag_field[i, j]:
            midpos = particle_position_field[i, j] + interpolate_velocity(particle_position_field[i, j], velocity_field_ux, velocity_field_uy) * (dt * 0.5)
            particle_position_field[i, j] += interpolate_velocity(midpos, velocity_field_ux, velocity_field_uy) * dt


@ti.kernel
def apply_markers():
    for i, j in marker_field:
        if SOLID != marker_field[i, j]:
            marker_field[i, j] = AIR

    for m, n in particle_position_field:
        if 1 == particle_flag_field[m, n]:
            i = clamp(int(particle_position_field[m, n].x), 0, grid_width-1)
            j = clamp(int(particle_position_field[m, n].y), 0, grid_height-1)
            if (SOLID != marker_field[i, j]):
                marker_field[i, j] = FLUID


@ti.kernel
def add_external_forces(dt: ti.f32):
    for i, j in velocity_field_uy:
        velocity_field_uy[i, j] += gravity * dt


@ti.kernel
def apply_boundary_conditions():
    for i, j in marker_field:
        if SOLID == marker_field[i, j]:
            velocity_field_ux[i, j] = 0.0
            velocity_field_ux[i + 1, j] = 0.0
            velocity_field_uy[i, j] = 0.0
            velocity_field_uy[i, j + 1] = 0.0

@ti.kernel
def calculate_divergence(velocity_field_ux: ti.template(), velocity_field_uy: ti.template(), div: ti.template(), marker_field: ti.template()):
    for i, j in div:
        if FLUID == marker_field[i, j]:
            du_dx = velocity_field_ux[i, j] - velocity_field_ux[i + 1, j]
            du_dy = velocity_field_uy[i, j] - velocity_field_uy[i, j + 1]
            div[i, j] = du_dx + du_dy


def solve_pressure():
    divergence.fill(0.0)
    calculate_divergence(velocity_field_ux, velocity_field_uy, divergence, marker_field)

    pressure.fill(0.0)
    new_pressure.fill(0.0)
    ps.solve(pressure, divergence)
    # solve(dt) too unstable


@ti.kernel
def apply_pressure():
    for i, j in velocity_field_ux:
        if is_marker_a_fluid(i - 1, j) or is_marker_a_fluid(i, j):
            velocity_field_ux[i, j] += pressure[i - 1, j] - pressure[i, j]

    for i, j in velocity_field_uy:
        if is_marker_a_fluid(i, j - 1) or is_marker_a_fluid(i, j):
            velocity_field_uy[i, j] += pressure[i, j - 1] - pressure[i, j]


@ti.kernel
def mark_valid_velocity_field_ux():
    for i, j in velocity_field_ux:
        if is_marker_a_fluid(i - 1, j) or is_marker_a_fluid(i, j):
            validity_field[i, j] = 1
        else:
            validity_field[i, j] = 0


@ti.kernel
def mark_valid_velocity_field_uy():
    for i, j in velocity_field_uy:
        if is_marker_a_fluid(i, j - 1) or is_marker_a_fluid(i, j):
            validity_field[i, j] = 1
        else:
            validity_field[i, j] = 0


@ti.kernel
def diffuse_quantity(dst: ti.template(), src: ti.template(),
                     validity_field_dst: ti.template(), validity_field: ti.template()):
    for i, j in dst:
        if 0 == validity_field[i, j]:
            sum = 0.0
            count = 0
            for m, n in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                if 1 == validity_field[i + m, j + n]:
                    sum += src[i + m, j + n]
                    count += 1
            if count > 0:
                dst[i, j] = sum / float(count)
                validity_field_dst[i, j] = 1


def extrapolate_velocities():
    mark_valid_velocity_field_ux()
    for i in range(10):
        new_velocity_field_ux.copy_from(velocity_field_ux)
        new_validity_field.copy_from(validity_field)
        diffuse_quantity(velocity_field_ux, new_velocity_field_ux, validity_field, new_validity_field)

    mark_valid_velocity_field_uy()
    for i in range(10):
        new_velocity_field_uy.copy_from(velocity_field_uy)
        new_validity_field.copy_from(validity_field)
        diffuse_quantity(velocity_field_uy, new_velocity_field_uy, validity_field, new_validity_field)


@ti.kernel
def update_from_grid():
    for i, j in velocity_field_ux:
        saved_velocity_field_ux[i, j] = velocity_field_ux[i, j] - saved_velocity_field_ux[i, j]

    for i, j in velocity_field_uy:
        saved_velocity_field_uy[i, j] = velocity_field_uy[i, j] - saved_velocity_field_uy[i, j]

    for m, n in particle_position_field:
        if 1 == particle_flag_field[m, n]:
            gvel = interpolate_velocity(particle_position_field[m, n], velocity_field_ux, velocity_field_uy)
            dvel = interpolate_velocity(particle_position_field[m, n], saved_velocity_field_ux, saved_velocity_field_uy)
            particle_velocity_field[m, n] = flip_viscosity * gvel + \
                (1.0 - flip_viscosity) * (particle_velocity_field[m, n] + dvel)


@ti.kernel
def transfer_to_grid(weights_ux: ti.template(), weights_uy: ti.template()):
    for m, n in particle_velocity_field:
        if 1 == particle_flag_field[m, n]:
            x, y = particle_position_field[m, n].x, particle_position_field[m, n].y
            u, v = particle_velocity_field[m, n].x, particle_velocity_field[m, n].y
            splat(velocity_field_ux, weights_ux, u, x, y, 0.0, 0.5, grid_width + 1, grid_height)
            splat(velocity_field_uy, weights_uy, v, x, y, 0.5, 0.0, grid_width, grid_height + 1)

    for i, j in weights_ux:
        if weights_ux[i, j] > 0.0:
            velocity_field_ux[i, j] /= weights_ux[i, j]

    for i, j in weights_uy:
        if weights_uy[i, j] > 0.0:
            velocity_field_uy[i, j] /= weights_uy[i, j]

def save_velocities():
    saved_velocity_field_ux.copy_from(velocity_field_ux)
    saved_velocity_field_uy.copy_from(velocity_field_uy)

def step():
    global frame_counter
    frame_counter += 1

    add_external_forces(dt)
    apply_boundary_conditions()

    extrapolate_velocities()
    apply_boundary_conditions()

    solve_pressure()
    apply_pressure()

    extrapolate_velocities()
    apply_boundary_conditions()

    # flip
    update_from_grid()
    advect_markers(dt)

    apply_markers()

    velocity_field_ux.fill(0.0)
    velocity_field_uy.fill(0.0)
    new_velocity_field_ux.fill(0.0)
    new_velocity_field_uy.fill(0.0)
    transfer_to_grid(new_velocity_field_ux, new_velocity_field_uy)

    saved_velocity_field_ux.copy_from(velocity_field_ux)
    saved_velocity_field_uy.copy_from(velocity_field_uy)

    apply_boundary_conditions_to_particles()


@ti.kernel
def copy_marker_particles_to_color_field():
    for i, j in color_field_to_display:
        # fill each pixel
        marker_index_x = int((i + 0.5) * grid_width / window_width_resolution)
        marker_index_y = int((j + 0.5) * grid_height / window_height_resolution)
        # color_field_to_display[i, j] = pressure[marker_index_x, marker_index_y]
        if marker_field[marker_index_x, marker_index_y] == FLUID:
            # show fluids in blue
            color_field_to_display[i, j] = (90/255,188/255,216/255)
        elif marker_field[marker_index_x, marker_index_y] == SOLID:
            # show solids in red
            color_field_to_display[i, j] = (230/255,90/255,120/255)
        else:
            # show background in white
            color_field_to_display[i, j] = (1, 1, 1)

@ti.kernel 
def init_classic():
    #add a cube of water at center of the screen
    for i, j in pressure:
        pressure[i, j] = 0
        divergence[i, j] = 0

        if (i > 83 and i < 163 and j > 120 and j < 200):
            marker_field[i, j] = FLUID
        else:
            marker_field[i, j] = AIR

        if (i == 0 or i == grid_width-1 or j == 0 or j == grid_height-1):
            marker_field[i, j] = SOLID

@ti.kernel
def apply_boundary_conditions_to_particles():
    for n, m in particle_position_field:
        if 1 == particle_flag_field[n, m]:
            if particle_position_field[n, m].x > 2*grid_width:
                particle_position_field[n, m].x = 2*grid_width
                particle_velocity_field[n, m].x = 0.0
            if particle_position_field[n, m].y > 2*grid_height:
                particle_position_field[n, m].y = 2*grid_height
                particle_velocity_field[n, m].y = 0.0
            if particle_position_field[n, m].x < 0.0:
                particle_position_field[n, m].x = 0.0
                particle_velocity_field[n, m].x = 0.0
            if particle_position_field[n, m].y < 0.0:
                particle_position_field[n, m].y = 0.0
                particle_velocity_field[n, m].y = 0.0

@ti.kernel
def init_particles():
    for m, n in particle_position_field:
        i, j = m // 2, n // 2
        particle_position_field[m, n] = [0.0, 0.0]
        if FLUID == marker_field[i, j]:
            particle_flag_field[m, n] = 1

            x = i + ((m % 2) + 0.5) / 2.0
            y = j + ((n % 2) + 0.5) / 2.0

            particle_position_field[m, n] = [x, y]

    # for n, m in particle_position_field:
    #     if n >= 2*grid_width - reserved_particles and m >= 2*grid_width - reserved_particles:
    #         particle_velocity_field[n, m] = ti.Vector([0.0, 0])
    #         #initialize the particle position
    #         i = 2*grid_width - n - reserved_particles//2
    #         j = 2*grid_height - m - reserved_particles//2
    #         particle_position_field[n, m] = ti.Vector([grid_width // 2 + i, grid_height // 2 + j])

    #         #update the marker_field
    #         marker_field[grid_width // 2 + i, grid_height // 2 + j] = FLUID

    #         #update the particle flag
    #         particle_flag_field[n, m] = 1


def initialize():
    velocity_field_ux.fill(0.0)
    velocity_field_uy.fill(0.0)
    particle_flag_field.fill(0)
    particle_position_field.fill(ti.Vector([0.0, 0.0]))
    particle_velocity_field.fill(ti.Vector([0.0, 0.0]))

    init_classic()
    init_particles()
    
def main():

    window = ti.ui.Window("FLIP Simulation", (window_width_resolution, window_height_resolution), vsync=False)
    canvas = window.get_canvas()

    initialize()

    paused = False

    while window.running:

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused


        if not paused:
            step()

        copy_marker_particles_to_color_field()

        canvas.set_image(color_field_to_display)
        window.show()


# if __name__ == 'main':
main()
