import taichi as ti 
from fluidSimulator import FluidSimulator


def main():

    resolution = 256
    #initialize ti
    ti.init(arch=ti.cpu)
    #initialize window
    window = ti.ui.Window("Fluid Simulator", (resolution, 2*resolution), vsync=False)
    canvas = window.get_canvas()
    #initialize the fluid simulator
    fluid_sym = FluidSimulator.create()

    pause = False
    while window.running:
        #if not paused take a step
        if not pause:
            fluid_sym.step()
        #catch a pause, pres p to pause
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == "p":
                paused = not paused
                
        #render the image     
        img = fluid_sym.render()
        canvas.set_image(img)
        window.show()


if __name__ == "__main__":
    main()