import taichi as ti
from solver import MacSolver



class FluidSimulator: 
    #initializer
    def __init__(self, solver): 
        self.solver = solver
        self.rbg_buffer = ti.Vector.field(3, dtype=ti.f32, shape=solver.grid_shape)  
        self._wall_color = ti.Vector([0.5, 0.7, 0.5])
    
    #step function
    def step(self):
        self.solver.update()


    def create():
        solver = MacSolver(
                boundary_condition, pressure_updater, advect_upwind, dt, re, vorticity_confinement
            )
    
        return FluidSimulator(solver)  