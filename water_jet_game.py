import taichi as ti
import numpy as np
import math

# import Box2D
# from Box2D.b2 import (
#     edgeShape,
#     circleShape,
#     fixtureDef,
#     polygonShape,
#     revoluteJointDef,
#     contactListener,
# )

# import gym
# from gym import spaces
# from gym.utils import seeding

ti.init(default_fp=ti.f32, arch=ti.x64, kernel_profiler=True)

grid_width = 256
grid_height = grid_width
window_width_resolution = 512
window_height_resolution = window_width_resolution

use_flip = True
save_results = False

gravity = -9.81
flip_viscosity = 0.0

SOLID = 2
AIR = 1
FLUID = 0

shoot_count = 20
reserved_particles = 20

dt = 0.01
frame_counter = 0


class Fountain: 
    def __init__(self, width):
        self.width = width
        self.orientation = 0
        self.offset = 0
        self.angle = 0 # in radiant


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


@ti.data_oriented
class MultigridPCGPoissonSolver:
    def __init__(self, marker, grid_width, grid_height):
        shape = (grid_width, grid_height)
        self.grid_width, self.grid_height = shape
        print(f'grid_width, grid_height = {grid_width}, {grid_height}')

        self.dim = 2
        self.max_iters = 300
        self.n_mg_levels = 1
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.use_multigrid = False

        def _res(l): return (grid_width // (2**l), grid_height // (2**l))

        self.r = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # M^-1 r
        self.d = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # temp
        self.f = [marker] + [ti.field(dtype=ti.i32, shape=_res(_))
                             for _ in range(self.n_mg_levels - 1)]  # marker
        self.L = [ti.Vector.field(6, dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # -L operator

        self.x = ti.field(dtype=ti.f32, shape=shape)  # solution
        self.p = ti.field(dtype=ti.f32, shape=shape)  # conjugate gradient
        self.Ap = ti.field(dtype=ti.f32, shape=shape)  # matrix-vector product
        self.alpha = ti.field(dtype=ti.f32, shape=())  # step size
        self.beta = ti.field(dtype=ti.f32, shape=())  # step size
        self.sum = ti.field(dtype=ti.f32, shape=())  # storage for reductions

        for _ in range(self.n_mg_levels):
            print(f'r[{_}].shape = {self.r[_].shape}')
        for _ in range(self.n_mg_levels):
            print(f'L[{_}].shape = {self.L[_].shape}')

    @ti.func
    def is_marker_a_fluid(self, f, i, j, grid_width, grid_height):
        return i >= 0 and i < grid_width and j >= 0 and j < grid_height and FLUID == f[i, j]

    @ti.func
    def is_marker_a_solid(self, f, i, j, grid_width, grid_height):
        return i < 0 or i >= grid_width or j < 0 or j >= grid_height or SOLID == f[i, j]

    @ti.func
    def is_marker_air(self, f, i, j, grid_width, grid_height):
        return i >= 0 and i < grid_width and j >= 0 and j < grid_height and AIR == f[i, j]

    @ti.func
    def neighbor_sum(self, L, x, f, i, j, grid_width, grid_height):
        ret = x[(i - 1 + grid_width) % grid_width, j] * L[i, j][2]
        ret += x[(i + 1 + grid_width) % grid_width, j] * L[i, j][3]
        ret += x[i, (j - 1 + grid_height) % grid_height] * L[i, j][4]
        ret += x[i, (j + 1 + grid_height) % grid_height] * L[i, j][5]
        return ret

    # -L matrix : 0-diagonal, 1-diagonal inverse, 2...-off diagonals
    @ti.kernel
    def init_L(self, l: ti.template()):
        _grid_width, _grid_height = self.grid_width // (2**l), self.grid_height // (2**l)
        for i, j in self.L[l]:
            if FLUID == self.f[l][i, j]:
                s = 4.0
                s -= float(self.is_marker_a_solid(self.f[l], i - 1, j, _grid_width, _grid_height))
                s -= float(self.is_marker_a_solid(self.f[l], i + 1, j, _grid_width, _grid_height))
                s -= float(self.is_marker_a_solid(self.f[l], i, j - 1, _grid_width, _grid_height))
                s -= float(self.is_marker_a_solid(self.f[l], i, j + 1, _grid_width, _grid_height))
                self.L[l][i, j][0] = s
                self.L[l][i, j][1] = 1.0 / s
            self.L[l][i, j][2] = float(
                self.is_marker_a_fluid(self.f[l], i - 1, j, _grid_width, _grid_height))
            self.L[l][i, j][3] = float(
                self.is_marker_a_fluid(self.f[l], i + 1, j, _grid_width, _grid_height))
            self.L[l][i, j][4] = float(
                self.is_marker_a_fluid(self.f[l], i, j - 1, _grid_width, _grid_height))
            self.L[l][i, j][5] = float(
                self.is_marker_a_fluid(self.f[l], i, j + 1, _grid_width, _grid_height))

    def solve(self, x, rhs):
        tol = 1e-12

        self.r[0].copy_from(rhs)
        self.x.fill(0.0)

        self.Ap.fill(0.0)
        self.p.fill(0.0)

        for l in range(1, self.n_mg_levels):
            self.downsample_f(self.f[l - 1], self.f[l],
                              self.grid_width // (2**l), self.grid_height // (2**l))
        for l in range(self.n_mg_levels):
            self.L[l].fill(0.0)
            self.init_L(l)

        self.sum[None] = 0.0
        self.reduction(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        #print(f"init rtr = {initial_rTr}")

        if initial_rTr < tol:
            #print(f"converged: init rtr = {initial_rTr}")
            pass
        else:
            # r = b - Ax = b    since x = 0
            # p = r = r + 0 p
            self.z[0].copy_from(self.r[0])

            self.update_p()

            self.sum[None] = 0.0
            self.reduction(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iter = 0
            for i in range(self.max_iters):
                # alpha = rTr / pTAp
                self.apply_L(0, self.p, self.Ap)

                self.sum[None] = 0.0
                self.reduction(self.p, self.Ap)
                pAp = self.sum[None]

                self.alpha[None] = old_zTr / pAp

                # x = x + alpha p
                # r = r - alpha Ap
                self.update_x_and_r()

                # check for convergence
                self.sum[None] = 0.0
                self.reduction(self.r[0], self.r[0])
                rTr = self.sum[None]
                print(rTr)
                if rTr < initial_rTr * tol:
                    break

                # z = M^-1 r
                self.z[0].copy_from(self.r[0])

                # beta = new_rTr / old_rTr
                self.sum[None] = 0.0
                self.reduction(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                self.beta[None] = new_zTr / old_zTr

                # p = z + beta p
                self.update_p()
                old_zTr = new_zTr

                iter = i
            #print(f'converged to {rTr} in {iter} iters')

        x.copy_from(self.x)

    @ti.kernel
    def apply_L(self, l: ti.template(), x: ti.template(), Ax: ti.template()):
        _grid_width, _grid_height = self.grid_width // (2**l), self.grid_height // (2**l)
        for i, j in Ax:
            if FLUID == self.f[l][i, j]:
                r = x[i, j] * self.L[l][i, j][0]
                r -= self.neighbor_sum(self.L[l], x,
                                       self.f[l], i, j, _grid_width, _grid_height)
                Ax[i, j] = r

    @ti.kernel
    def reduction(self, p: ti.template(), q: ti.template()):
        for I in ti.grouped(p):
            if FLUID == self.f[0][I]:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x_and_r(self):
        a = float(self.alpha[None])
        for I in ti.grouped(self.p):
            if FLUID == self.f[0][I]:
                self.x[I] += a * self.p[I]
                self.r[0][I] -= a * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if FLUID == self.f[0][I]:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def downsample_f(self, f_fine: ti.template(), f_coarse: ti.template(),
                     grid_width: ti.template(), grid_height: ti.template()):
        for i, j in f_coarse:
            i2 = i * 2
            j2 = j * 2

            if AIR == f_fine[i2, j2] or AIR == f_fine[i2 + 1, j2] or \
               AIR == f_fine[i2, j2 + 1] or AIR == f_fine[i2 + 1, j2 + 1]:
                f_coarse[i, j] = AIR
            else:
                if FLUID == f_fine[i2, j2] or FLUID == f_fine[i2 + 1, j2] or \
                   FLUID == f_fine[i2 + 1, j2] or FLUID == f_fine[i2 + 1, j2 + 1]:
                    f_coarse[i, j] = FLUID
                else:
                    f_coarse[i, j] = SOLID

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
        print(residual)

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
    # ps.solve(pressure, divergence)
    solve(dt)


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

def step(input, fountain : ti.template()):
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

    # reposition ball particles
    reset_ball_particles()
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
def init_ballgame():
    next_particle_insert_counter[None] = 0

    for i, j in pressure:
        pressure[i, j] = 0
        divergence[i, j] = 0

        if (j > grid_height - 10 and j < grid_height - 2 and i > 70 and i < grid_width - 70):
            marker_field[i, j] = AIR
            #pass
        else:
            marker_field[i, j] = AIR

        if (i == 0 or i == grid_width-1 or j == 0 or j == grid_height-1):
            marker_field[i, j] = SOLID

        #add a ball to the scene
        # if (i - grid_width // 2) ** 2 + (j - grid_height // 2) ** 2 < 100:
        #     marker_field[i, j] = SOLID

@ti.kernel
def reset_ball_particles():
    average_position = ti.Vector([0., 0.,])
    average_velocity = ti.Vector([0., 0.,])
    counter = 0
    for n, m in particle_position_field:
        if n >= 2*grid_width - reserved_particles and m >= 2*grid_width - reserved_particles:
            average_velocity += particle_velocity_field[n, m]
            average_position += particle_position_field[n, m]
            counter += 1
    average_position /= counter
    average_velocity /= counter
    for n, m in particle_position_field:
        if n >= 2*grid_width - reserved_particles and m >= 2*grid_width - reserved_particles:
            particle_velocity_field[n, m] = average_velocity
            #initialize the particle position
            i = 0.5 + n - (2*grid_width - reserved_particles) - reserved_particles//2
            j = 0.5 + m - (2*grid_height - reserved_particles) - reserved_particles//2
            particle_position_field[n, m] = average_position + ti.Vector([i, j])

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

    for n, m in particle_position_field:
        if n >= 2*grid_width - reserved_particles and m >= 2*grid_width - reserved_particles:
            particle_velocity_field[n, m] = ti.Vector([0.0, 0])
            #initialize the particle position
            i = 2*grid_width - n - reserved_particles//2
            j = 2*grid_height - m - reserved_particles//2
            particle_position_field[n, m] = ti.Vector([grid_width // 2 + i, grid_height // 2 + j])

            #update the marker_field
            marker_field[grid_width // 2 + i, grid_height // 2 + j] = FLUID

            #update the particle flag
            particle_flag_field[n, m] = 1


def initialize():
    velocity_field_ux.fill(0.0)
    velocity_field_uy.fill(0.0)
    particle_flag_field.fill(0)
    particle_position_field.fill(ti.Vector([0.0, 0.0]))
    particle_velocity_field.fill(ti.Vector([0.0, 0.0]))

    init_ballgame()
    init_particles()

#seems like the water disppears when another call of this function is made
@ti.kernel
def shoot_water(offset: int, angle: float):
    #rotation for fountain
    rotation = ti.Matrix([[ti.cos(angle), -ti.sin(angle)], [ti.sin(angle), ti.cos(angle)]])
    vel_vector = rotation @ ti.Vector([0, 50])

    #shoot water into the scene from the bottom of the screen
    for j in range (20):
        #marker_field[grid_width // 2 + i, 100] = FLUID
        #marker_field[grid_width // 2 - i, 100] = FLUID
        for i in range (20 - 2*j):
        #update the particle velocity 
            particle_velocity_field[offset + grid_width // 2 + i, j] = vel_vector
            particle_velocity_field[offset + grid_width // 2 - i, j] = vel_vector
            #initialize the particle position
            particle_position_field[offset + grid_width // 2 + i, j] = ti.Vector([offset + grid_width // 2 + i, j])
            particle_position_field[offset + grid_width // 2 - i, j] = ti.Vector([offset + grid_width // 2 - i, j])

            #update the marker_field
            marker_field[offset + grid_width // 2 + i, j] = FLUID
            marker_field[offset + grid_width // 2 - i, j] = FLUID

            #update the particle flag
            particle_flag_field[offset + grid_width // 2 + i, j] = 1
            particle_flag_field[offset + grid_width // 2 - i, j] = 1

            #update the pressure
            pressure[offset + grid_width // 2 + i, j] = 0
            pressure[offset + grid_width // 2 - i, j] = 0

            #update the divergence
            divergence[offset + grid_width // 2 + i, j] = 0
            divergence[offset + grid_width // 2 - i, j] = 0

            #update the velocity
            velocity_field_ux[offset + grid_width // 2 + i, j] = 0
            velocity_field_ux[offset + grid_width // 2 - i, j] = 0

            velocity_field_uy[offset + grid_width // 2 + i, j] = 0
            velocity_field_uy[offset + grid_width // 2 - i, j] = 0

    # my attempt at shooting water. kinda fails, kinda works.    
    # for n, m in particle_position_field:
    #     if n >= next_particle_insert_counter[None] and n < next_particle_insert_counter[None] + shoot_count and m >= next_particle_insert_counter[None] and m < next_particle_insert_counter[None] + shoot_count:
    #         i = next_particle_insert_counter[None] - n + shoot_count//2
    #         j = next_particle_insert_counter[None] - m
    #         particle_velocity_field[n, m] = ti.Vector([0.0, 200])
    #         #initialize the particle position
    #         particle_position_field[n, m] = ti.Vector([grid_width // 2 + i, j])

    #         #update the marker_field
    #         marker_field[grid_width // 2 + i, j] = FLUID

    #         #update the particle flag
    #         particle_flag_field[n, m] = 1

    #         #update the pressure
    #         pressure[grid_width // 2 + i, j] = 0

    #         #update the divergence
    #         divergence[grid_width // 2 + i, j] = 0

    #         #update the velocity
    #         velocity_field_ux[grid_width // 2 + i, j] = 0
    #         velocity_field_uy[grid_width // 2 + i, j] = 0
    # next_particle_insert_counter[None] += shoot_count
    # if next_particle_insert_counter[None] + shoot_count + reserved_particles >= 2 * grid_width:
    #     next_particle_insert_counter[None] = 0   
    
def main():

    window = ti.ui.Window("Water Jet Game", (window_width_resolution, window_height_resolution), vsync=False)
    canvas = window.get_canvas()

    initialize()

    paused = False
    input = 0
    #create fountain object
    fountain = Fountain(11 // 2)

    while window.running:

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused
            elif e.key == "d":
                fountain.offset += 8
                #print(fountain.offset)
            elif e.key == "a":
                fountain.offset -= 8    
            elif e.key == "q":
                fountain.angle += 0.1
            elif e.key == "e":
                fountain.angle -= 0.1
            if e.key == ti.ui.SPACE:
                #print("shooting water")
                input = 1

        if input == 1:
            shoot_water(fountain.offset, fountain.angle)

        if not paused:
            step(input, fountain)
        input = 0

        copy_marker_particles_to_color_field()

        canvas.set_image(color_field_to_display)
        window.show()


# if __name__ == 'main':
main()
