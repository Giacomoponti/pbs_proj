import taichi as ti
import numpy as np
import math

SOLID = 2
AIR = 1
FLUID = 0

# adapted from here https://gitee.com/citadel2020/taichi_demos/blob/master/mgpcgflip/flip.md
# and here https://gitee.com/citadel2020/taichi_demos/tree/master
# the standard Gauss-Seidel solver didn't work for us.

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