import argparse
from pathlib import Path
import taichi as ti
import taichi.math as tm
from taichi.examples.patterns import taichi_logo
#fountain class
from fountain import Fountain

# configuration and hyper parameters
window_height = 512
window_width = window_height
shape = (window_width, window_height)
stepsize_delta_t = 0.05
rho = 1.0
g = ti.Vector([0, -9.81])
color_decay = 0.001 # set 0 for no color decay/fading
color_multiplier = 1 - color_decay


#shift of the fountain 
fountain_offset = 0


# init step for taichi lang
ti.init(arch=ti.cpu)

# define fields
# we need two velocity fields, one before the update u_{i-1} and one after the update u_{i}.
# Most implementations swap the fields after each update so that the old field can be recycled as a new-new field.
velocity_field_u = ti.Vector.field(n=2, dtype=float, shape=shape)
new_velocity_field_u = ti.Vector.field(n=2, dtype=float, shape=shape)
# color field
# Actually q is the color as well as the velocity
color_field_q = ti.Vector.field(n=3, dtype=float, shape=shape)
new_color_field_q = ti.Vector.field(n=3, dtype=float, shape=shape)
display_inverted_color_field = ti.Vector.field(n=3, dtype=float, shape=shape)
# divergence field for intermediate results for the pressure solver
divergence_field = ti.field(dtype=float, shape=shape)
pressure_field_p = ti.field(dtype=float, shape=shape)

#field for boundary conditions
bc_mask = ti.field(dtype=float, shape=shape)

# pressure solver
# from: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py
@ti.kernel
def fill_laplacian_matrix(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(window_width, window_height):
        row = i * window_width + j
        center = 0.0
        if j != 0:
            A[row, row - 1] += -1.0
            center += 1.0
        if j != window_height - 1:
            A[row, row + 1] += -1.0
            center += 1.0
        if i != 0:
            A[row, row - window_height] += -1.0
            center += 1.0
        if i != window_width - 1:
            A[row, row + window_height] += -1.0
            center += 1.0
        A[row, row] += center

N = window_width * window_height
K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
F_b = ti.ndarray(ti.f32, shape=N)

fill_laplacian_matrix(K)
L = K.build()
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(L)
solver.factorize(L)


@ti.func
def field_copy(src: ti.template(), dst: ti.template()):
    for I in ti.grouped(src):
        dst[I] = src[I]

# begin bilinear interpolation. copied from ti example 6
@ti.func
def sample(vf, u, v):
    i, j = int(u), int(v)
    # Nearest
    i = ti.max(0, ti.min(shape[0] - 1, i))
    j = ti.max(0, ti.min(shape[1] - 1, j))
    return vf[i, j]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return (1 - frac) * vl + frac * vr

@ti.func
def bilerp(vf, coord):
    u = coord[0]
    v = coord[1]
    # use -0.5 to decide where bilerp performs in cells
    s, t = u - 0.5, v - 0.5
    iu, iv = int(s), int(t)
    a = sample(vf, iu + 0.5, iv + 0.5)
    b = sample(vf, iu + 1.5, iv + 0.5)
    c = sample(vf, iu + 0.5, iv + 1.5)
    d = sample(vf, iu + 1.5, iv + 1.5)
    # fract
    fu, fv = s - iu, t - iv
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)
# end bilinear interpolation

@ti.kernel
def update_advection_step():

    #temporarily set color to 0 for boundary conditions, to prevent color from leaking out
    for i, j in color_field_q:
        if bc_mask[i, j] == 1:
            color_field_q[i, j] = 0


    # according to ETH lecture 4 slide 36
    for i, j in velocity_field_u:
    
        #ADDED: set velocity at boundary condition
        if bc_mask[i, j] == 1:

            velocity_field_u[i, j] = ti.Vector([0, 0])#ti.Vector([0, -9.81])#*stepsize_delta_t
            # if bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
            #     velocity_field_u[i + 1, j] = -velocity_field_u[i - 1, j]
            # elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
            #     velocity_field_u[i - 1, j] = -velocity_field_u[i + 1, j]
            # elif bc_mask[i, j - 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
            #     velocity_field_u[i, j + 1] = -velocity_field_u[i, j - 1]
            # elif bc_mask[i, j + 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
            #     velocity_field_u[i, j - 1] = -velocity_field_u[i, j + 1]  

            #falling ball, when collides with ground it explodes lol 
            bc_mask[i, j] = 0
            bc_mask[i, int(j - 9.81*stepsize_delta_t)] = 1


        # 1. Dertime velocity u_ij at grid point
        velocity_u_ij = velocity_field_u[i, j]
        # 2. Integrate position for a timestep of - delta_t
        # find x_ij via MAC
        position_now = ti.Vector([i, j]) + ti.Vector([0.5, 0.5]) # +0.5 because of MAC to be in the middle of the cell
        # x_source = x_ij - delta_t * u_ij
        # TODO: here we could use a Runge Kutta 3 backtrace step if we wanted
        position_source = position_now - velocity_u_ij * stepsize_delta_t
        # 3. Interpolate q at x_source to obtain q_source
        # q seems to be the color value

        
        #this interpolation creates problem in aroiund the ball as the color is interpolated from the sphere too
        color_q_source = bilerp(color_field_q, position_source)

        # if bc_mask[i, j] == 1:
        #     bc_mask[i, j] = 0
        #     new_x = int(position_source[0])
        #     new_y = int(position_source[1])
        #     bc_mask[new_x, new_y] = 1
        #     color_q_source = ti.Vector([0, 255, 255])
        
        # 4. Assign q_ij = q_source for next time step
        new_color_field_q[i, j] = color_q_source * color_multiplier
        # redo step 3 for velocity as well. Actually q is the color as well as the velocity
        # 3. Interpolate q at x_source to obtain q_source
        # q seems to be the color value
        velocity_u_source = bilerp(velocity_field_u, position_source)
        # 4. Assign q_ij = q_source for next time step
        new_velocity_field_u[i, j] = velocity_u_source * color_multiplier

    #reset color to red for boundary conditions
    for i, j in color_field_q:
        if bc_mask[i, j] == 1:
            new_color_field_q[i, j] = ti.Vector([0, 255, 255])

        
    # set the new fields to the old pointers.
    field_copy(new_color_field_q, color_field_q)
    field_copy(new_velocity_field_u, velocity_field_u)

    # apply boundary conditions
    # TODO: apply boundary conditions



@ti.kernel
def update_externalforces_step(offset : ti.template(), angle : ti.template()):
    # I think the only external force is gravity.
    # according to ETH lecture 4 slide 36,
    # we can use a simple forward Euler integration
    for i, j in velocity_field_u:
        velocity_field_u[i, j] = velocity_field_u[i, j] + stepsize_delta_t*g/rho
    # apparently, only do it for u and not q -> ignore the color field here
    
    # artificial force for the input
    fountain_pixel_width = 11 // 2  + 10

    # seems like source only moves when offset becomes lerger than fountain width
    for k in range(-fountain_pixel_width, fountain_pixel_width+1):
        #rotaion for fountain
        rotation = ti.Matrix([[ti.cos(angle), -ti.sin(angle)], [ti.sin(angle), ti.cos(angle)]])

        velocity_field_u[int(window_width/2 + offset)-k, 0] = rotation @ ti.Vector([0, 900])
        color_field_q[int(window_width/2 + offset)-k, 0] = 1

    # apply boundary conditions
    # TODO: apply boundary conditions

@ti.kernel
def compute_divergence_field():
    # from https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py
    for i, j in velocity_field_u:
        vl = sample(velocity_field_u, i - 1, j)
        vr = sample(velocity_field_u, i + 1, j)
        vb = sample(velocity_field_u, i, j - 1)
        vt = sample(velocity_field_u, i, j + 1)
        vc = sample(velocity_field_u, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == window_width - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == window_height - 1:
            vt.y = -vc.y
        divergence_field[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5

@ti.kernel
def fill_fb_array(f_b_object: ti.types.ndarray()):
    for I in ti.grouped(divergence_field):
        f_b_object[I[0] * window_width + I[1]] = -divergence_field[I]

@ti.kernel
def fill_pressure_field_back_in(x: ti.types.ndarray()):
    for I in ti.grouped(pressure_field_p):
        pressure_field_p[I] = x[I[0] * window_width + I[1]]

@ti.kernel
def update_velocities_from_pressure():
    # update velocities
    for i, j in velocity_field_u:
        pl = sample(pressure_field_p, i - 1, j)
        pr = sample(pressure_field_p, i + 1, j)
        pb = sample(pressure_field_p, i, j - 1)
        pt = sample(pressure_field_p, i, j + 1)
        velocity_field_u[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


def update_pressure_step():
    # solve linear system on sparse matrix to solve for pressure,
    # then use explicit Euler to update next step
    update_pressure_bc(pressure_field_p, bc_mask)
    # see: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py
    # first compute the divergence.
    # This is the right hand side of the 2D MAC grid equation on ETH lecture 4 slide 39
    compute_divergence_field()
    # second build the sparse matrix and solve for that
    fill_fb_array(F_b)
    x = solver.solve(F_b)
    fill_pressure_field_back_in(x)
    # update pressure 
    update_velocities_from_pressure()
    # swap_velocity_fields() the above for loop does not depend on the velocity field,
    # so we can update the old field directly without needing to swap

@ti.kernel
def update_pressure_bc(pressure_field_p: ti.template(), bc_mask: ti.template()):
    # update the pressure at boundary conditions
    for i, j in pressure_field_p:
        if bc_mask[i, j] == 1:
            pressure_field_p[i, j] = 0
        # if bc_mask[i, j] == 1:
        #     if bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
        #         pressure_field_p[i, j] = pressure_field_p[i - 1, j]
        #     elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
        #         pressure_field_p[i, j] = pressure_field_p[i + 1, j]
        #     elif bc_mask[i, j - 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
        #         pressure_field_p[i, j] = pressure_field_p[i, j - 1]
        #     elif bc_mask[i, j + 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
        #         pressure_field_p[i, j] = pressure_field_p[i, j + 1]
        #     elif bc_mask[i - 1, j] == 0 and bc_mask[i, j + 1] == 0:
        #         pressure_field_p[i, j] = (pressure_field_p[i - 1, j] + pressure_field_p[i, j + 1]) / 2.0
        #     elif bc_mask[i + 1, j] == 0 and bc_mask[i, j + 1] == 0:
        #         pressure_field_p[i, j] = (pressure_field_p[i + 1, j] + pressure_field_p[i, j + 1]) / 2.0
        #     elif bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 0:
        #         pressure_field_p[i, j] = (pressure_field_p[i - 1, j] + pressure_field_p[i, j - 1]) / 2.0
        #     elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 0:
        #         pressure_field_p[i, j] = (pressure_field_p[i + 1, j] + pressure_field_p[i, j - 1]) / 2.0

@ti.kernel
def set_inverted_colorfield():
    for I in ti.grouped(color_field_q):
        display_inverted_color_field[I] = 1-color_field_q[I]

def update_all_steps(fountain : ti.template()):
    # Split the Navier-Stokes equation into four parts. See ETH lecture 4 slide 30
    # Treat each term seperately, in the following order
    # 1) Advection
    # 2) Viscosity (curl)
    # 3) External forces (or mouse interaction)
    # 4) Pressure (use a pressure solver)
    # Apply boundary conditions either at each step or at the very end

    update_advection_step()
    # ignore viscosity. note from lecture:
    # "numerical dissipation due to Semi-Lagrangian advection is often sufficient"
    #print("offset: ", fountain.offset)
    update_externalforces_step(fountain.offset, fountain.angle)
    update_pressure_step()

    set_inverted_colorfield()

@ti.kernel
def init_fields():
    pressure_field_p.fill(0)
    velocity_field_u.fill(0)
    color_field_q.fill(0)
    # for i, j in ti.ndrange(shape[0] * 4, shape[1] * 4):
    #     # 4x4 super sampling:
    #     ret = taichi_logo(ti.Vector([i, j]) / (shape[0] * 4))
    #     color_field_q[i // 4, j // 4] += ret / 16


    #fill boundary condition mask and set the ball in the middle
    bc_mask.fill(0)
    for i in range(0, window_width):
        for j in range(0, window_height):
            if (i - window_width/2)**2 + (j - window_height/2)**2 < 30**2:
                bc_mask[i, j] = 1
                color_field_q[i, j] = 1


def main():
    
    window = ti.ui.Window("Simulation", (window_width, window_height), vsync=False)
    canvas = window.get_canvas()

    init_fields()

    paused = False
    #create fountain object
    fountain = Fountain(11 // 2)

    while window.running:
        
        if not paused:
            # update step
            update_all_steps(fountain)

        if window.get_event(ti.ui.PRESS):
            e = window.event
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

        canvas.set_image(display_inverted_color_field)
        window.show()

if __name__ == "__main__":
    main()
