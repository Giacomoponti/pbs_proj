import taichi as ti
import math as m
#ball class
class Ball:

    def __init__(self, density, radius):
        self.density = density;
        self.r = radius;
        self.mass = density * m.pi * radius * radius;
        self.center = ti.Vector(0, 0);
        self.linear_velocity = ti.Vector(0, 0);
        self.angular_velocity = 0;
        self.angular_momentum = 0;
        self.inertia = (self.mass/2.) * self.r * self.r;

    def isInside(self, point):
        return (point - self.center).norm() < self.r;

    def update(self, dt):
        self.center += self.linear_velocity * dt;
        self.angular_momentum += self.angular_velocity * dt;
        self.angular_velocity = self.angular_momentum / self.inertia;