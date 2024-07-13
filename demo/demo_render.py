import cv2
import imageio
import imageio.v3 as iio
import numpy as np
import math
from mesh_augmentator import MeshModel

output_dim = 224

def render_rect(mesh, rect, img):
    x, y, w, h = rect
    x1, y1 = mesh.project_point(x, y)
    x2, y2 = mesh.project_point(x + w, y)
    x3, y3 = mesh.project_point(x + w, y + h)
    x4, y4 = mesh.project_point(x, y + h)

    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2)
    cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), (0, 0, 255), thickness=2)
    cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), thickness=2)
    cv2.line(img, (int(x4), int(y4)), (int(x1), int(y1)), (0, 0, 255), thickness=2)    

def render_depth(sample, background, rects):
    h, w, dc = sample.shape
    dw = w // 30
    dh = h // 30
    frames = []
    for i in range(1, 200, 10):
        mesh = MeshModel(dw, dh, sample, False, False)
        mesh.lens_radius = i # small radius => small depth simulation

        mesh.cylynder_vertical(R = w * 2)
        mesh.shift(0, 0, mesh.get_best_object_distance())

        mesh.set_output_size(output_dim, output_dim)
        output_image = mesh.render(background)
        frames.append(np.copy(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)))        
        mesh.free()

    l = len(frames)
    for c in range(1, l):
        frames.append(np.copy(frames[l - c]))

    iio.imwrite('depth.gif', frames, duration=300, loop=0)
        

def render_light(sample, background_template, rects):
    h, w, dc = sample.shape
    dw = w // 30
    dh = h // 30
    frames = []    
    for i in range(1, 400, 20):
        mesh = MeshModel(dw, dh, sample, True, False)
        mesh.set_light_position(0, i, -100)

        mesh.cylynder_vertical(R = w * 8)
        mesh.shift(0, 0, mesh.get_best_object_distance())

        mesh.set_output_size(output_dim, output_dim)
        output_image = mesh.render(background)
        frames.append(np.copy(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)))        
        mesh.free()

    l = len(frames)
    for c in range(1, l):
        frames.append(np.copy(frames[l - c]))

    iio.imwrite('light.gif', frames, duration=200, loop=0)

def render_shadow(sample, background, rects):
    h, w, dc = sample.shape
    dw = w // 30
    dh = h // 30
    frames = []    
    for i in range(1, 200, 10):        
        mesh = MeshModel(dw, dh, sample, True, True)
        mesh.set_light_position(0, -200, -100)
        mesh.set_shadow_y(-i)        

        mesh.cylynder_vertical(R = w * 8)
        mesh.shift(0, 0, mesh.get_best_object_distance())

        mesh.set_output_size(output_dim, output_dim)
        output_image = mesh.render(background)
        frames.append(np.copy(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)))        
        mesh.free()

    l = len(frames)
    for c in range(1, l):
        frames.append(np.copy(frames[l - c]))

    iio.imwrite("shadow.gif", frames, duration=200, loop=0)
        
def render_light_diameter(sample, background, rects):
    h, w, dc = sample.shape
    dw = w // 30
    dh = h // 30
    frames = []    
    for i in range(1, 200, 10):        
        mesh = MeshModel(dw, dh, sample, True, True)
        mesh.set_light_position(0, -200, -100)
        mesh.set_shadow_y(-100)
        mesh.set_light_diameter(i)

        mesh.cylynder_vertical(R = w * 8)
        mesh.shift(0, 0, mesh.get_best_object_distance())

        mesh.set_output_size(output_dim, output_dim)
        output_image = mesh.render(background)
        frames.append(np.copy(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)))        
        mesh.free()

    l = len(frames)
    for c in range(1, l):
        frames.append(np.copy(frames[l - c]))

    iio.imwrite("light_diameter.gif", frames, duration=200, loop=0)

def render_rotate_and_cylynder(sample, background, rects):
    h, w, dc = sample.shape
    dw = w // 30
    dh = h // 30
    frames = []    
    for i in range(-30, 30, 3):        
        mesh = MeshModel(dw, dh, sample, True, True)
        if i != 0:
            mesh.cylynder_vertical(R = w * 4 / (i / 30))
        else:
            mesh.cylynder_vertical(R = w * 10)
            
        mesh.shift(0, 0, mesh.get_best_object_distance())
        mesh.rotate(math.radians(i))
        mesh.set_output_size(output_dim, output_dim)
        output_image = mesh.render(background)
        for rect in rects:
            render_rect(mesh, rect, output_image)
            
        frames.append(np.copy(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)))        
        mesh.free()

    l = len(frames)
    for c in range(1, l):
        frames.append(np.copy(frames[l - c]))

    iio.imwrite("rotate.gif", frames, duration=200, loop=0)    

#prepare data
sample = cv2.imread('sample.jpg')
h, w, dc = sample.shape
k = min(output_dim / h, output_dim / w) * 2.2
sample = cv2.resize(sample, (int(w * k), int(h * k)), interpolation = cv2.INTER_LINEAR)
h, w, dc = sample.shape
background = cv2.imread('wood.jpg')
background = cv2.resize(background, (output_dim, output_dim), interpolation = cv2.INTER_LINEAR)

rects = []
with open("rects.txt") as file:    
    for line in [line.rstrip() for line in file]:
        rects.append([float(a) * k for a in line.split(':')[1].split(';')])

render_depth(sample, background, rects)
render_light(sample, background, rects)
render_shadow(sample, background, rects)
render_light_diameter(sample, background, rects)
render_rotate_and_cylynder(sample, background, rects)
