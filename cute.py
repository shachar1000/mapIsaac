import noise
import numpy as np
from PIL import Image
import cv2
from filter import filter_circle, filter_ellipse, filter_square
from scipy.misc import toimage

shape = (600,600)
scale = 0.1
octaves = 11
persistence = 0.5
lacunarity = 2.0
seed = np.random.randint(0,100)

world = np.zeros(shape)

x_idx = np.linspace(0, 1, shape[0])
y_idx = np.linspace(0, 1, shape[1])
world_x, world_y = np.meshgrid(x_idx, y_idx)

world = np.vectorize(noise.snoise2)(world_x/scale,
                        world_y/scale,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        base=seed)


print(world)


# print(noise.pnoise2(-0.1, -0.1, octaves=octaves,
# persistence=persistence,
# lacunarity=lacunarity,
# repeatx=600,
# repeaty=600,
# base=seed))

world_noise = np.zeros_like(world)

for i in range(shape[0]):
    for j in range(shape[1]):
        world_noise[i][j] = (world[i][j] * filter_square[i][j])
        # if world_noise[i][j] > 0:
        #     world_noise[i][j] *= 20

# max = np.max(world_noise)
# world_noise = world_noise / max

# world_print = np.floor((world_noise+0.5) * 255).astype(np.uint8) 
# Image.fromarray(world_print, mode='L').show()
toimage(world_noise).show()


lightblue = [0,191,255]
blue = [65,105,225]
green = [34,139,34]
darkgreen = [0,100,0]
sandy = [210,180,140]
beach = [238, 214, 175]
snow = [255, 250, 250]
mountain = [139, 137, 137]


threshold = 0

def add_color(world):
    color_world = np.zeros(world.shape+(3,))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if world[i][j] < threshold + 0.05:
                color_world[i][j] = blue
            elif world[i][j] < threshold + 0.055:
                color_world[i][j] = sandy
            elif world[i][j] < threshold + 0.1:
                color_world[i][j] = beach
            elif world[i][j] < threshold + 0.25:
                color_world[i][j] = green
            elif world[i][j] < threshold + 0.6:
                color_world[i][j] = darkgreen
            elif world[i][j] < threshold + 0.7:
                color_world[i][j] = mountain
            elif world[i][j] < threshold + 1.0:
                color_world[i][j] = snow
    return color_world
    
#print(world.min())    
#world = np.floor((world + .5) * 255).astype(np.uint8) # <- Normalize world first
color_world = add_color(world_noise)
cv2.imshow("Simple_black", cv2.cvtColor(color_world.astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
