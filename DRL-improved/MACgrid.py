
class MACgrid:

    def __init__(self, width, length):
        self.width = width
        self.length = length 
        self.h = h 
        self.rest_u = 0
        self.rest_v = 0
        #horizontal velocity field 
        self.vel_u = ti.field(dtype=ti.f32, shape=(width + 1, length))
        #vertical velocity grid 
        self.vel_v = ti.field(dype=ti.f32, shape=(width, length + 1))
        #pressuere grid
        self.pressure = ti.field(dtype=ti.f32, shape=(width, length))

    def U(pos): 
        interpolate(pos / h - ti.Vector(0, 0.5), self.vel_u)

    def V(pos):
        interpolate(pos / h - ti.Vector(0.5, 0), self.vel_v)

    def velocity(pos):
        return ti.Vector([U(pos), V(pos)])    

    def get_barycentric(p):
        i = int(p)
        f = p - i
        return i, f

    def interpolate(pos, grid):
        i, f = get_barycentric(pos)
        return grid[i] * (1 - f) + grid[i + 1] * f    