import taichi as ti
from solver import MacSolver
from Ball import Ball
from Pipe import Pipe


class FluidSimulator: 
    #initializer
    def __init__(self, solver): 
        self.ball = Ball(1.5, 0.1)
        self.pipe = Pipe((0.5, 0))
        self.water = MACgrid()
        self.solver = solver
        self.rbg_buffer = ti.Vector.field(3, dtype=ti.f32, shape=solver.grid_shape)  
        self._wall_color = ti.Vector([0.5, 0.7, 0.5])
    
    #step function
    def step(self):
        self.ball.advance(self.solver.dt)
        self.solver.step()
        self.solver.update()


    def create():
        solver = MacSolver(
                boundary_condition, pressure_updater, advect_upwind, dt, re, vorticity_confinement
            )
    
        return FluidSimulator(solver)  
    
    def render(self):
        self._render_background()
        self._render_ball()
        self._render_pipe()
        self._render_water()
        return self.rbg_buffer.to_numpy()