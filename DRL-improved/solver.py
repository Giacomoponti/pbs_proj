import taichi as ti
from abc import ABCMeta, abstractmethod

VELOCITY_LIMIT = 70.0

class DoubleBuffers:
    def __init__(self, resolution, n_channel):
        if n_channel == 1:
            self.current = ti.field(ti.f32, shape=resolution)
            self.next = ti.field(ti.f32, shape=resolution)
        else:
            self.current = ti.Vector.field(n_channel, ti.f32, shape=resolution)
            self.next = ti.Vector.field(n_channel, ti.f32, shape=resolution)

    def swap(self):
        self.current, self.next = self.next, self.current

    def reset(self):
        self.current.fill(0)
        self.next.fill(0)

@ti.data_oriented
class Solver(metaclass=ABCMeta):
    def __init__(self, boundary_condition):
        self._bc = boundary_condition
        self._resolution = boundary_condition.get_resolution()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def get_fields(self):
        pass

    @ti.func
    def is_wall(self, i, j):
        return self._bc.is_wall(i, j)

    @ti.func
    def is_fluid_domain(self, i, j):
        return self._bc.is_fluid_domain(i, j)


@ti.kernel
def limit_field(field: ti.template(), limit: ti.f32):
    for i, j in field:
        norm = field[i, j].norm()
        if norm > limit:
            field[i, j] = limit * (field[i, j] / norm)


@ti.kernel
def clamp_field(field: ti.template(), low: ti.f32, high: ti.f32):
    for i, j in field:
        field[i, j] = ti.min(ti.max(field[i, j], low), high)

class MacSolver(Solver):
    """Maker And Cell method"""

    def __init__(
        self,
        boundary_condition,
        pressure_updater,
        advect_function,
        dt,
        Re,
        vorticity_confinement=None,
    ):
        super().__init__(boundary_condition)

        self._advect = advect_function
        self.dt = dt
        self.Re = Re

        self.pressure_updater = pressure_updater
        self.vorticity_confinement = vorticity_confinement

        self.v = DoubleBuffers(self._resolution, 2)  # velocity
        self.p = DoubleBuffers(self._resolution, 1)  # pressure

    def update(self):
        self._bc.set_velocity_boundary_condition(self.v.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self.v.swap()

        if self.vorticity_confinement is not None:
            self.vorticity_confinement.apply(self.v)
            self.v.swap()

        self.pressure_updater.update(self.p, self.v.current)

        limit_field(self.v.current, VELOCITY_LIMIT)

    def get_fields(self):
        return self.v.current, self.p.current

    @ti.kernel
    def _update_velocities(self, vn: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in vn:
            if self.is_fluid_domain(i, j):
                vn[i, j] = vc[i, j] + self.dt * (
                    -self._advect(vc, vc, i, j)
                    - ti.Vector(
                        [
                            diff_x(pc, i, j),
                            diff_y(pc, i, j),
                        ]
                    )
                    + (diff2_x(vc, i, j) + diff2_y(vc, i, j)) / self.Re
                )