import taichi as ti 
#Pipe class for shooting water into the scene
class Pipe:
    def __init__(self, pos):
        self.pos = pos

    def update(self, dt, input):
        if (input == 'left'):
            self.pos[0] -= 0.01

        if (input == 'right'):
            self.pos[0] += 0.01

        if (input == 'space'):
            #shoot water (still to implement)      
            pass
