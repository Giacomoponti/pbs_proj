import argparse
from pathlib import Path
import numpy as np
import taichi as ti
from taichi.examples.patterns import taichi_logo
import time

print_time_spent = False

# configuration and hyper parameters
display_multiplier = 1
window_height = 100
window_width = window_height
shape = (window_width, window_height)
N = window_width * window_height
dx = 1.0
stepsize_delta_t = 0.01
rho = 1.0
g = ti.Vector([0, -9.81])
color_decay = 0 #0.001 # set 0 for no color decay/fading
color_multiplier = 1 - color_decay

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
supersize_color_field = ti.Vector.field(n=3, dtype=float, shape=(window_width*display_multiplier, window_height*display_multiplier))
# divergence field for intermediate results for the pressure solver
divergence_field = ti.field(dtype=float, shape=shape)
pressure_field_p = ti.field(dtype=float, shape=shape)

#field for boundary conditions
bc_mask = ti.field(dtype=float, shape=shape)

# rigid body, geometry
# let's start with a square
box2d_width = int(window_width/8)
box2d_width2 = box2d_width*box2d_width
box2d_width_half = box2d_width//2
mass = 1.0
imatrix = mass / 12 * (box2d_width2 + box2d_width2)
imatrixInv = 1.0/imatrix
rbd_com = ti.Vector.field(n=2, dtype=float, shape=())
rbd_angle = ti.field(dtype=float, shape=())
rbd_angular_momentum = ti.field(dtype=float, shape=())
rbd_linear_velocity = ti.Vector.field(n=2, dtype=float, shape=())

# static geometry
nodal_rigid_phi = ti.field(dtype=float, shape=shape)
rigid_u_weights = ti.Vector.field(n=2, dtype=float, shape=shape)
solid_u = ti.Vector.field(n=2, dtype=float, shape=shape)
# pressure solver fields  
u_valid = ti.Vector.field(n=2, dtype=int, shape=shape)
old_u_valid = ti.Vector.field(n=2, dtype=int, shape=shape)
liquid_phi = ti.field(dtype=float, shape=shape)
u_weights = ti.Vector.field(n=2, dtype=float, shape=shape)
base_trans_x = ti.field(dtype=float, shape=shape)
base_trans_y = ti.field(dtype=float, shape=shape)
base_rot_z = ti.field(dtype=float, shape=shape)
# viscosity solver fields
u_vol = ti.Vector.field(n=4, dtype=float, shape=shape)
viscosity = ti.field(dtype=float, shape=shape)
# marker particles
particles = ti.Vector.field(n=2, dtype=float, shape=(2*N))
particle_radius = dx / 1.414 # sqrt(2)

nonzeros_count_for_rows = ti.field(dtype=int, shape=(N))

@ti.func
def to_local_box_coord(pos):
    return pos - rbd_com[None]

@ti.func
def to_global_coord_from_box(pos):
    return pos + rbd_com[None]

@ti.func
def is_inside_square_box(pos):
    return pos[0] >= -box2d_width/2 and pos[0] <= box2d_width/2 and pos[1] >= -box2d_width/2 and pos[1] <= box2d_width/2

@ti.func
def project_out_of_square_box(pos):
    if is_inside_square_box(pos):
        distance_left = pos[0] + box2d_width/2
        distance_right = box2d_width/2 - pos[0]
        distance_bottom = pos[1] + box2d_width/2
        distance_top = box2d_width/2 - pos[1]

        if distance_bottom <= distance_right and distance_bottom <= distance_left and distance_bottom <= distance_top:
            pos[1] = -box2d_width/2
        elif distance_right <= distance_left and distance_right <= distance_bottom and distance_right <= distance_top:
            pos[0] = box2d_width/2
        elif distance_left <= distance_right and distance_left <= distance_bottom and distance_left <= distance_top:
            pos[0] = -box2d_width/2
        else:
            pos[1] = box2d_width/2
    return pos

@ti.func
def get_signed_distance_function_to_box(pos):
    return local_signed_distance_function_to_box(pos - rbd_com[None])

@ti.func
def local_signed_distance_function_to_box(pos_local):
    # negative if pos is inside
    # positive if pos is outside
    pos = pos_local

    # compute distances
    distance_left = - pos[0] - box2d_width/2
    distance_right = -box2d_width/2 + pos[0]
    distance_bottom = -pos[1] - box2d_width/2
    distance_top = -box2d_width/2 + pos[1]

    # how many are positives
    positive_one = 0.0
    positive_one_set = False
    positive_two = 0.0
    i = 0
    if distance_left >= 0:
        if not positive_one_set:
            positive_one = distance_left
            positive_one_set = True
        else:
            positive_two = distance_left
        i += 1
    if distance_right >= 0:
        if not positive_one_set:
            positive_one = distance_right
            positive_one_set = True
        else:
            positive_two = distance_right
        i += 1
    if distance_bottom >= 0:
        if not positive_one_set:
            positive_one = distance_bottom
            positive_one_set = True
        else:
            positive_two = distance_bottom
        i += 1
    if distance_top >= 0:
        if not positive_one_set:
            positive_one = distance_top
            positive_one_set = True
        else:
            positive_two = distance_top
        i += 1

    return_val = 0.0
    if i == 2:
        return_val = ti.math.sqrt(positive_one*positive_one + positive_two*positive_two)
    elif i == 1:
        return_val = positive_one
    else:
        return_val = max(max(distance_left, distance_right), max(distance_bottom, distance_top))
    return return_val

@ti.kernel
def advect_particles():
    # move particles in/through the fluid
    # perform backtrace as in lecture
    # we could use Runge Kutta 2, 3, or 4
    for i in particles:
        particle_pos = particles[i]
        # use RK 2
        start_velocity = bilerp(velocity_field_u, particle_pos)
        midpoint_pos = particle_pos + 0.5*stepsize_delta_t*start_velocity
        mid_velocity = bilerp(velocity_field_u, midpoint_pos)
        new_particle_pos = particle_pos + stepsize_delta_t*mid_velocity

        # TODO: boundary condition. maybe

        # test for collision with the rigid body
        pos_local = to_local_box_coord(new_particle_pos)
        pos_projected = project_out_of_square_box(pos_local)
        new_pos_projected = to_global_coord_from_box(pos_projected)

        particles[i] = new_pos_projected

        if new_pos_projected[0] < 0 or new_pos_projected[0] >= window_width or new_pos_projected[1] < 0 or new_pos_projected[1] >= window_height:
            # reposition "lost" particles at the fountain again
            particles[i] = ti.Vector([ti.random()*window_width/8+window_width/16*7, ti.random()*window_width/16])

    # adjust particles that drifted to close to each other
    # min_distance = 0.5*dx
    # for i,j in ti.ndrange(window_width, window_width):
    #     if i == j:
    #         continue
    #     dist = (particles[i] - particles[j]).norm()
    #     if dist < min_distance:
    #         direction = (particles[i] - particles[j]).normalized()
    #         particles[i] -= 0.5*(dist - min_distance)*direction
    #         particles[j] += 0.5*(dist - min_distance)*direction

@ti.kernel
def advance_rbd():
    rbd_com[None] += rbd_linear_velocity[None]*stepsize_delta_t
    rbd_angle[None] += rbd_angular_momentum[None]*stepsize_delta_t
    # apply force
    rbd_linear_velocity[None] += g*stepsize_delta_t
    # apply rotation
    # Updating angular velocity omega
    # NB: omega = (I^-1)L
    rbd_angular_momentum[None] = imatrixInv * rbd_angular_momentum[None]; 

@ti.func
def fraction_inside_levelset(phi_left, phi_right):
    return_val = 0.0
    if phi_left >= 0 and phi_right >= 0:
        # all empty
        return_val = 0
    elif phi_left < 0 and phi_right < 0:
        # all full
        return_val = 1
    elif phi_left >= 0:
        return_val = 1 - phi_left / (phi_left - phi_right)
    else:
        return_val = phi_left / (phi_left - phi_right)
    return return_val

@ti.kernel
def update_rigid_body_fields():
    # update level set from current position
    for i, j in nodal_rigid_phi:
        location = ti.Vector([i*dx, j*dx])
        nodal_rigid_phi[i, j] = get_signed_distance_function_to_box(location)

    # compute face area fractions from distance field
    for i, j in rigid_u_weights:
        rigid_bottom = nodal_rigid_phi[i, j]
        rigid_top = nodal_rigid_phi[i, j + 1] if j + 1 < window_height else 0
        u_fraction = fraction_inside_levelset(rigid_bottom, rigid_top)
        rigid_left = nodal_rigid_phi[i, j]
        rigid_right = nodal_rigid_phi[i + 1, j] if i + 1 < window_height else 0
        v_fraction = fraction_inside_levelset(rigid_left, rigid_right)
        rigid_u_weights[i, j] = ti.Vector([u_fraction, v_fraction])

    # TODO: recompute the grid-based "effective" masses per axis, so that we can exactly balance in hydrostatic scenarios.

@ti.func
def get_barycentric(x, i_low, i_high):
    s = ti.math.floor(x)
    i = int(x)
    f = 0.0
    if i < i_low:
        i, f = i_low, 0
    elif i > i_high-2:
        i, f = i_high-2, 1
    else:
        i, f = i, x - s
    return i, f

@ti.kernel
def compute_phi():
    # estimate the liquid signed distance
    # estimate from particles
    liquid_phi.fill(3*dx)
    for p in particles:
        pos = particles[p]
        # determine containing cell
        i, fx = get_barycentric(pos[0] / dx - 0.5, 0, window_width)
        j, fy = get_barycentric(pos[1] / dx - 0.5, 0, window_height)

        # compute distance to surrounding few points, keep if it's the minimum
        for j_off in range(j-2, j+3):
            for i_off in range(i-2, i+3):
                if i_off < 0 or i_off >= window_width or j_off < 0 or j_off >= window_height:
                    continue

                position = ti.Vector([(i_off+0.5)*dx, (j_off+0.5)*dx])
                phi_temp = (pos - position).norm() - 1.02*particle_radius
                liquid_phi[i_off, j_off] = min(liquid_phi[i_off, j_off], phi_temp)

    # extrapolate phi into solids if nearby
    for i, j in liquid_phi:
        if liquid_phi[i, j] < 0.5*dx:
            if i + 1 < window_width and j + 1 < window_height:
                solid_phi_val = 0.25*(nodal_rigid_phi[i,j] + nodal_rigid_phi[i+1, j] + nodal_rigid_phi[i, j+1] + nodal_rigid_phi[i+1, j+1])
                if solid_phi_val < 0:
                    liquid_phi[i, j] = -0.5*dx


@ti.func
def rungekutta2(pos):
    velocity = bilerp(velocity_field_u, pos)
    velocity = bilerp(velocity_field_u, pos - 0.5*stepsize_delta_t*velocity)
    return pos - stepsize_delta_t*velocity

@ti.kernel
def advect():
    for i, j in velocity_field_u:
        # semi-Lagrangian advection
        new_velocity_field_u[i, j] = bilerp(velocity_field_u, rungekutta2(ti.Vector([i, j]) + ti.Vector([0.5, 0.5])))
    field_copy(new_velocity_field_u, velocity_field_u)

@ti.func
def clamp(n, minimum, maximum): 
    return max(minimum, min(n, maximum))

@ti.kernel
def compute_pressure_weights():
    # compute finite-volume style face-weights for fluid from nodal signed distances
    for i, j in u_weights:
        u_weight = 1 - fraction_inside_levelset(nodal_rigid_phi[i, j+1], nodal_rigid_phi[i, j]) - rigid_u_weights[i, j][0]
        v_weight = 1 - fraction_inside_levelset(nodal_rigid_phi[i+1, j], nodal_rigid_phi[i, j]) - rigid_u_weights[i, j][1]
        u_weights[i, j] = ti.Vector([clamp(u_weight, 0, 1), clamp(v_weight, 0, 1)])

@ti.kernel
def fill_matrix_to_solve(K: ti.types.sparse_matrix_builder(), F_b: ti.types.ndarray()):
    for i, j in base_trans_x:
        # translation coupling
        base_trans_x[i, j] = ((rigid_u_weights[i + 1, j] - rigid_u_weights[i, j]) / dx)[0]
        base_trans_y[i, j] = ((rigid_u_weights[i, j + 1] - rigid_u_weights[i, j]) / dx)[1]

        # rotation coupling
        pos = ti.Vector([(i+0.5)*dx, (j+0.5)*dx])
        rad = pos - rbd_com[None]
        base_rot_z[i, j] = rad[0]*base_trans_y[i,j] - rad[1]*base_trans_x[i,j]

    any_liquid_surface = False


    nonzeros_count_for_rows.fill(0)
    # build the linear system for pressure
    for i, j in ti.ndrange((1, window_width-1), (1, window_height-1)):
        index = i + window_width*j
        center_phi = liquid_phi[i, j]
        if center_phi < 0:
            # right neighbour
            term = u_weights[i+1, j][0]*stepsize_delta_t / (dx*dx)
            if term > 0:
                right_phi = liquid_phi[i+1, j]
                if right_phi < 0:
                    K[index, index] += term
                    K[index, index+1] += -term
                    nonzeros_count_for_rows[index] += 2
                else:
                    theta = fraction_inside_levelset(center_phi, right_phi)
                    if theta < 0.01:
                        theta = 0.01
                    K[index, index] += term/theta
                    nonzeros_count_for_rows[index] += 1
                    any_liquid_surface = True
                F_b[index] += -(u_weights[i+1, j][0]*velocity_field_u[i + 1, j][0] / dx)

            # left neighbour
            term = u_weights[i, j][0]*stepsize_delta_t / (dx*dx)
            if term > 0:
                left_phi = liquid_phi[i-1, j]
                if left_phi < 0:
                    K[index, index] += term
                    K[index, index-1] += -term
                    nonzeros_count_for_rows[index] += 2
                else:
                    theta = fraction_inside_levelset(center_phi, left_phi)
                    if theta < 0.01:
                        theta = 0.01
                    K[index, index] += term/theta
                    nonzeros_count_for_rows[index] += 1
                    any_liquid_surface = True
                F_b[index] = +(u_weights[i, j][0]*velocity_field_u[i, j][0] / dx)

            # top neighbour
            term = u_weights[i, j+1][1]*stepsize_delta_t / (dx*dx)
            if term > 0:
                top_phi = liquid_phi[i, j+1]
                if top_phi < 0:
                    K[index, index] += term
                    K[index, index+window_width] += -term
                    nonzeros_count_for_rows[index] += 2
                else:
                    theta = fraction_inside_levelset(center_phi, top_phi)
                    if theta < 0.01:
                        theta = 0.01
                    K[index, index] += term/theta
                    nonzeros_count_for_rows[index] += 1
                    any_liquid_surface = True
                F_b[index] = -(u_weights[i, j+1][1]*velocity_field_u[i, j+1][1] / dx)

            # bottom neighbour
            term = u_weights[i, j][1]*stepsize_delta_t / (dx*dx)
            if term > 0:
                bottom_phi = liquid_phi[i, j-1]
                if bottom_phi < 0:
                    K[index, index] += term
                    K[index, index - window_width] += -term
                    nonzeros_count_for_rows[index] += 2
                else:
                    theta = fraction_inside_levelset(center_phi, bottom_phi)
                    if theta < 0.01:
                        theta = 0.01
                    K[index, index] += term/theta
                    nonzeros_count_for_rows[index] += 1
                    any_liquid_surface = True
                F_b[index] = +(u_weights[i, j][1]*velocity_field_u[i, j][1] / dx)

    for i, j in ti.ndrange(window_width, window_height):
        index = i + window_width*j
        center_phi = liquid_phi[i, j]
        if center_phi < 0:
            # RHS contributions
            # translation
            F_b[index] -= rbd_linear_velocity[None][0] * base_trans_x[i, j]
            F_b[index] -= rbd_linear_velocity[None][1] * base_trans_y[i, j]
            # rotation
            F_b[index] -= rbd_angular_momentum[None] * base_rot_z[i, j]

    for i, j, k, m in ti.ndrange(window_width, window_height, window_width, window_height):
        index = i + window_width*j
        center_phi = liquid_phi[i, j]
        if center_phi < 0:
            # LHS matrix contributions
            val = 0.0
            other_phi = liquid_phi[k, m]
            if other_phi < 0:
                # translation 
                val += stepsize_delta_t * base_trans_x[i, j] * base_trans_x[k, m] / mass
                val += stepsize_delta_t * base_trans_y[i, j] * base_trans_y[k, m] / mass
                # rotation
                val += stepsize_delta_t * base_rot_z[i, j] * base_rot_z[k, m] / mass
                if abs(val) > 0.000001:
                    index = i + window_width*j
                    K[index, k + window_width*m] += val
                    nonzeros_count_for_rows[index] += 1

    for i in ti.ndrange(N):
        if nonzeros_count_for_rows[i] == 0:
            K[i, i] += 1.0
            F_b[i] = 0

@ti.kernel
def update_pressure_after_solver(pressure: ti.types.ndarray()):
    # apply the velocity update
    for i, j in ti.ndrange((1, window_width-1), (0, window_height)):
        index = i + window_width*j
        if u_weights[i, j][0] > 0 and (liquid_phi[i,j] < 0 or liquid_phi[i-1, j] < 0):
            theta = 1.0
            if liquid_phi[i, j] >= 0 or liquid_phi[i-1, j] >= 0:
                theta = fraction_inside_levelset(liquid_phi[i-1, j], liquid_phi[i,j])
            if theta < 0.01:
                theta = 0.01
            velocity_field_u[i,j][0] -= stepsize_delta_t * (pressure[index] - pressure[index-1]) / dx / theta
            u_valid[i,j][0] = 1
        else:
            velocity_field_u[i,j][0] = 0
            u_valid[i,j][0] = 0
    for i, j in ti.ndrange((0, window_width), (1, window_height-1)):
        index = i + window_width*j
        if u_weights[i, j][1] > 0 and (liquid_phi[i,j] < 0 or liquid_phi[i, j-1] < 0):
            theta = 1.0
            if liquid_phi[i, j] >= 0 or liquid_phi[i, j-1] >= 0:
                theta = fraction_inside_levelset(liquid_phi[i, j-1], liquid_phi[i,j])
            if theta < 0.01:
                theta = 0.01
            velocity_field_u[i,j][1] -= stepsize_delta_t * (pressure[index] - pressure[index-window_width]) / dx / theta
            u_valid[i,j][1] = 1
        else:
            velocity_field_u[i,j][1] = 0
            u_valid[i,j][1] = 0

    # apply the pressure update to the rigid body
    for i,j in velocity_field_u:
        index = i + window_width*j
        center_phi = liquid_phi[i,j]
        if center_phi < 0:
            rbd_linear_velocity[None][0] += stepsize_delta_t * base_trans_x[i,j] * pressure[index] / mass
            rbd_linear_velocity[None][1] += stepsize_delta_t * base_trans_y[i,j] * pressure[index] / mass
            rbd_angular_momentum[None] += stepsize_delta_t * base_rot_z[i,j] * pressure[index]

def solve_pressure():
    # an implementation of the variational pressure projection solve for static geometry
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    F_b = ti.ndarray(ti.f32, shape=N)
    fill_matrix_to_solve(K, F_b)

    # solve the system
    L = K.build()

    #print(L)

    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)

    pressure = solver.solve(F_b)

    # fill_pressure_field_back_in(pressure)
    # update_velocities_from_pressure()
    update_pressure_after_solver(pressure)


@ti.kernel
def extrapolate_step():
    field_copy(velocity_field_u, new_velocity_field_u)
    field_copy(u_valid, old_u_valid)
    # extrapolate u
    for i, j in ti.ndrange((1, window_width-1), (1, window_height-1)):
        sum = 0.0
        count = 0
        if not old_u_valid[i, j][0]:
            if old_u_valid[i+1, j][0]:
                sum += velocity_field_u[i+1, j][0]
                count += 1
            if old_u_valid[i-1, j][0]:
                sum += velocity_field_u[i-1, j][0]
                count += 1
            if old_u_valid[i, j+1][0]:
                sum += velocity_field_u[i, j+1][0]
                count += 1
            if old_u_valid[i, j-1][0]:
                sum += velocity_field_u[i, j-1][0]
                count += 1
            if count:
                new_velocity_field_u[i, j][0] = sum / count
                u_valid[i, j][0] = 1
    # extrapolate v
    for i, j in ti.ndrange((1, window_width-1), (1, window_height-1)):
        sum = 0.0
        count = 0
        if not old_u_valid[i, j][1]:
            if old_u_valid[i+1, j][1]:
                sum += velocity_field_u[i+1, j][1]
                count += 1
            if old_u_valid[i-1, j][1]:
                sum += velocity_field_u[i-1, j][1]
                count += 1
            if old_u_valid[i, j+1][1]:
                sum += velocity_field_u[i, j+1][1]
                count += 1
            if old_u_valid[i, j-1][1]:
                sum += velocity_field_u[i, j-1][1]
                count += 1
            if count:
                new_velocity_field_u[i, j][1] = sum / count
                u_valid[i, j][1] = 1
    field_copy(new_velocity_field_u, velocity_field_u)

def extrapolate():
    for l in range(10):
        extrapolate_step()

@ti.kernel
def recompute_solid_velocity():
    pass
    # for i, j in ti.ndrange(window_width, window_height-1):
    #     pos = ti.Vector([i*dx, (j + 0.5)*dx])
    #     if (0.5*(nodal_solid_phi[i, j] + nodal_rigid_phi[i, j+1]) < 0.5*(nodal_rigid_phi[i, j] + no))


@ti.kernel
def color_particles():
    display_inverted_color_field.fill(1.0)
    for p in particles:
        pos = particles[p]
        pos_x_int = int(pos[0])
        pos_y_int = int(pos[1])
        if pos_x_int >= 0 and pos_x_int < window_width and pos_y_int >= 0 and pos_y_int < window_height:
            display_inverted_color_field[int(pos[0]), int(pos[1])].xzy = (90/255,188/255,216/255)

    for I in ti.grouped(display_inverted_color_field):
        if is_inside_square_box(to_local_box_coord(I)):
            display_inverted_color_field[I].xzy = (255/255,10/255,10/255)

def advance():
    # print(velocity_field_u)
    # print(particles)
    # print(rbd_com[None])

    # 1) passively advect particles
    start_time = time.process_time()
    advect_particles()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "advect_particles": {end_time-start_time:7.5f}s')
    start_time = time.process_time()

    # 2) time integrate rigid body
    advance_rbd()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "advance_rbd": {end_time-start_time:7.5f}s')
    start_time = time.process_time()

    # TODO: 3) process collisions between rigid body and boundaries

    # 4) recompute the distance fields and face areas
    update_rigid_body_fields()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "update_rigid_body_fields": {end_time-start_time:7.5f}s')
    start_time = time.process_time()

    # 5) estimate the liquid signed distance
    compute_phi()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "compute_phi": {end_time-start_time:7.5f}s')
    start_time = time.process_time()

    # 6) advance the velocity
    advect()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "advect": {end_time-start_time:7.5f}s')
    start_time = time.process_time()
    # update_advection_step()

    # 7) update external forces
    update_externalforces_step()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "update_externalforces_step": {end_time-start_time:7.5f}s')
    start_time = time.process_time()

    # 8) apply projection
    # 8.1) Compute finite-volume type face area weight for each velocity sample.
    compute_pressure_weights()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "compute_pressure_weights": {end_time-start_time:7.5f}s')
    start_time = time.process_time()
    # 8.2) Set up and solve the variational pressure solve, in either SPD or indefinite forms.
    solve_pressure()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "solve_pressure": {end_time-start_time:7.5f}s')
    start_time = time.process_time()
    # update_pressure_step()
    # 9) TODO: apply boundary conditions again and extrapolate

    # 10) color fields that have particles
    color_particles()
    end_time = time.process_time()
    if print_time_spent:
            print(f'process time for "color_particles": {end_time-start_time:7.5f}s')
    start_time = time.process_time()

    extrapolate()


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
#print(L)
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
            velocity_field_u[i, j] = ti.Vector([0, 0])
            continue
                
        # 1. Determine velocity u_ij at grid point
        velocity_u_ij = velocity_field_u[i, j]
        # 2. Integrate position for a timestep of - delta_t
        # find x_ij via MAC
        position_now = ti.Vector([i, j]) + ti.Vector([0.5, 0.5]) # +0.5 because of MAC to be in the middle of the cell
        # x_source = x_ij - delta_t * u_ij
        # TODO: here we could use a Runge Kutta 3 backtrace step if we wanted
        position_source = position_now - velocity_u_ij * stepsize_delta_t
        # 3. Interpolate q at x_source to obtain q_source
        # q seems to be the color value
        color_q_source = bilerp(color_field_q, position_source) #this interpolation creates problem in around the ball as the color is interpolated from the sphere too
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
def update_externalforces_step():
    # I think the only external force is gravity.
    # according to ETH lecture 4 slide 36,
    # we can use a simple forward Euler integration
    for i, j in velocity_field_u:
        velocity_field_u[i, j] = velocity_field_u[i, j] + stepsize_delta_t*g/rho
    # apparently, only do it for u and not q -> ignore the color field here

    # artificial force for the input
    fountain_pixel_width = 11 // 2
    for k in range(-fountain_pixel_width, fountain_pixel_width+1):
        velocity_field_u[int(window_width/2)-k, 0] = ti.Vector([0, 500.0])
        color_field_q[int(window_width/2)-k, 0] = 1

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
            if bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                pressure_field_p[i, j] = pressure_field_p[i - 1, j]
            elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                pressure_field_p[i, j] = pressure_field_p[i + 1, j]
            elif bc_mask[i, j - 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                pressure_field_p[i, j] = pressure_field_p[i, j - 1]
            elif bc_mask[i, j + 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                pressure_field_p[i, j] = pressure_field_p[i, j + 1]
            elif bc_mask[i - 1, j] == 0 and bc_mask[i, j + 1] == 0:
                pressure_field_p[i, j] = (pressure_field_p[i - 1, j] + pressure_field_p[i, j + 1]) / 2.0
            elif bc_mask[i + 1, j] == 0 and bc_mask[i, j + 1] == 0:
                pressure_field_p[i, j] = (pressure_field_p[i + 1, j] + pressure_field_p[i, j + 1]) / 2.0
            elif bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 0:
                pressure_field_p[i, j] = (pressure_field_p[i - 1, j] + pressure_field_p[i, j - 1]) / 2.0
            elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 0:
                pressure_field_p[i, j] = (pressure_field_p[i + 1, j] + pressure_field_p[i, j - 1]) / 2.0

@ti.kernel
def set_inverted_colorfield():
    # for I in ti.grouped(color_field_q):
    #     display_inverted_color_field[I] = 1-color_field_q[I]

    threshold = 0.5
    for I in ti.grouped(color_field_q):
        if is_inside_square_box(to_local_box_coord(ti.Vector(I))):
            display_inverted_color_field[I].xzy = (255/255,10/255,10/255)
        elif bc_mask[I] == 1:
            display_inverted_color_field[I] = (1, 0, 0)
        elif color_field_q[I].norm() > threshold:
            # Mark as part of the surface
            # color of water is somewhere among the following choices:
            #0f5e9c (15,94,156)
            #2389da (35,137,218)
            #1ca3ec (28,163,236)
            #5abcd8 (90,188,216)
            #74ccf4 (116,204,244)
            display_inverted_color_field[I].xzy = (90/255,188/255,216/255)
        else:
            # Mark as air or background
            display_inverted_color_field[I] = 1.0

def update_all_steps():
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
    update_externalforces_step()
    update_pressure_step()

    set_inverted_colorfield()

@ti.kernel
def init_fields():
    rbd_com[None] = ti.Vector([window_width/2, window_height*3/4])
    rbd_angle[None] = -ti.math.pi / 2
    rbd_linear_velocity[None] = ti.Vector([0, 0])
    rbd_angular_momentum[None] = 0

    for p in particles:
        particles[p] = ti.Vector([ti.random()*window_width/8+window_width/16*7, ti.random()*window_width/8])

    # for i,j in velocity_field_u:
    #     velocity_field_u[i,j] = ti.random()
    velocity_field_u.fill(0)

    pressure_field_p.fill(0)
    color_field_q.fill(0)

    nodal_rigid_phi.fill(0)
    rigid_u_weights.fill(0)
    solid_u.fill(0)
    u_valid.fill(0)
    liquid_phi.fill(0)
    u_weights.fill(0)
    base_trans_x.fill(0)
    base_trans_y.fill(0)
    base_rot_z.fill(0)



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

    window = ti.GUI("Simulation", res=(window_width*display_multiplier, window_height*display_multiplier))
    init_fields()


    video_manager = ti.tools.VideoManager(output_dir="./", framerate=24, automatic_build=False)


    paused = False
    while window.running:
    # for frame_counter in range(200):
        start_time_main = time.process_time()
        if not paused:
            # update step
            # update_all_steps()
            advance()
        end_time_main = time.process_time()
        if print_time_spent:
            print(f'process time for "advance": {end_time_main-start_time_main:7.5f}s')
        start_time_main = time.process_time()


        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused
            elif e.key == ti.GUI.SPACE:
                init_fields()

        # for i, j in ti.ndrange(shape[0] * display_multiplier, shape[1] * display_multiplier):
        #     supersize_color_field[i, j] = display_inverted_color_field[i // display_multiplier, j // display_multiplier]

        window.set_image(display_inverted_color_field)
        
        # window.rect(topleft=[(rbd_com[None][0]-box2d_width_half)/window_width, (rbd_com[None][1]+box2d_width_half)/window_width], bottomright=[(rbd_com[None][0]+box2d_width_half)/window_width, (rbd_com[None][1]-box2d_width_half)/window_width], color=0xFF00FF)
        window.show()

        img = display_inverted_color_field.to_numpy()
        video_manager.write_frame(img)


        end_time_main = time.process_time()
        if print_time_spent:
            print(f'process time for "draw": {end_time_main-start_time_main:7.5f}s')
    
    video_manager.make_video(gif=True, mp4=True)

if __name__ == "__main__":
    main()
