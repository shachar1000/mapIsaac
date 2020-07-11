import math
import numpy as np
from PIL import Image
from scipy.misc import toimage

shape = (600, 600)
cx, cy = 300, 300
filter_circle = np.zeros_like(np.zeros(shape))

# for y in range(filter_circle.shape[0]):
#     for x in range(filter_circle.shape[1]):
#         filter_circle[y][x] = math.sqrt(abs(x - cx)**2+abs(y - cy)**2)

# max = np.max(filter_circle)
# print(max)
# filter_circle = filter_circle / max
# filter_circle -= 0.5
# filter_circle *= 2.0
# filter_circle = -filter_circle
# # 
# for y in range(filter_circle.shape[0]):
#     for x in range(filter_circle.shape[1]):
#         if filter_circle[y][x] > 0:
#             filter_circle[y][x] *= 20

# # # get it between 0 and 1
# max = np.max(filter_circle)
# filter_circle = filter_circle / max

x_ = np.linspace(-1, 1, shape[0])
y_ = np.linspace(-1, 1, shape[1])

xx, yy = np.meshgrid(x_, y_)
filter_circle = np.sqrt(xx**2 + yy**2) # since origin 0



#filter_circle = np.floor((filter_circle) * 255).astype(np.uint8) 
#Image.fromarray(filter_circle, mode='L').show()
# radius = 300^2
# toimage(np.clip(radius-filter_circle, 0, 1)).show()
print(np.max(filter_circle)) # sqrt of 2
filter_circle = np.clip(1**2 - filter_circle, 0, 1)
toimage(filter_circle).show()


a = 300 #semi major axis of ellipse
b = 150 # minor
h = cx
k = cy
filter_ellipse = np.zeros(shape)
for y in range(filter_ellipse.shape[0]):
    for x in range(filter_ellipse.shape[1]):
        filter_ellipse[y, x] = (((x-h) * math.cos(0) + (y - k) * math.sin(0))**2) / (a ** 2) \
         + ((((x-h) * math.sin(0) + (y - k) * math.cos(0))**2) / b **2)

filter_ellipse = np.clip(1.0-filter_ellipse, 0, 1)
toimage(filter_ellipse).show()


x_ = np.linspace(-1, 1, shape[0])
y_ = np.linspace(-1, 1, shape[1])

xx, yy = np.meshgrid(x_, y_)
c = 0.5
filter_square = np.maximum(np.absolute(xx),np.absolute(yy))
# for y in yy:
#     for x in xx:
#         filter_square[y, x] = max((abs(x), abs(y)))
filter_square = np.clip(c - filter_square, 0, 1)
toimage(filter_square).show()

np.savetxt("ellipse_gradient.txt", filter_ellipse)
np.savetxt("square_gradient.txt", filter_square)
np.savetxt("circle_gradient.txt", filter_circle)
