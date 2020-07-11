import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import matplotlib.pyplot as plt
import pprint
import itertools
from test_polys import fix_polygons, bad
from scipy.spatial import distance
from sklearn import neighbors
import noise
from scipy.misc import toimage
import math
import random
import collections

points = np.random.rand(2200, 2)


shapeOfGradient = "circle"

# fucking stupid ass  donkey shit way to find circumcenter of triangle....   And this shit will be in bagrut FFS
def triangle_csc(pts):
    # diffs = x2-x1 and y2-y1 for 2 vertices of the triangle
    diffs = np.diff(pts[:3], axis=0) # the :3 is because we want only 2 vertices and not 3 for the calculations
    slopes = [(diffs[i][1]/diffs[i][0]) for i in range(2)]
    means = [[(pts[i][0]+pts[i+1][0])/2, (pts[i][1]+pts[i+1][1])/2] for i in range(2)]
    slopesOfPerpendicularBisectors = [(-1)/slopes[i] for i in range(len(slopes))]
    #y=mx+b   =>    b=y-mx
    b = [means[i][1]-(slopesOfPerpendicularBisectors[i]*means[i][0]) for i in range(2)]
    # m1x+b=m2x+b
    x = ((b[1]-b[0])/(slopesOfPerpendicularBisectors[0]-slopesOfPerpendicularBisectors[1]))
    y = (x*slopesOfPerpendicularBisectors[0])+b[0]
    return (x, y)

points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)
delauny = Delaunay(points)
triangles = delauny.points[delauny.vertices]
circum_centers = np.array([triangle_csc(tri) for tri in triangles])



circum_centers = circum_centers[np.logical_not(np.isnan(circum_centers))]
circum_centers = np.split(circum_centers, len(circum_centers)/2)


vor = Voronoi(circum_centers)

# plot


polygons = []
# colorize
for region in vor.regions:
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        polygons.append(polygon)
    
# def centroids(polys):
#     # n sides of all polygons
#     #polys = np.array([np.array([np.array(polys[i][j]) for j in range(len(polys[i]))]) for i in range(len(polys))])
# 
#     # ok, so in python we have a list of polygons and every polygon is a list of points
#     # we need to convert from python to numpy 
# 
#     polys = polys[1:]
#     polys = np.array(polys)
#     polys = np.array([np.array(polys[i]) for i in range(len(polys))])
#     polys = np.array([np.array(np.array([value for value in polys[i]])) for i in range(len(polys))])
#     #pprint.pprint(polys)
#     lengths = [polys[i].shape[0] for i in range(len(polys))]
#     sumX = [np.sum([polys[i][:, 0]]) for i in range(len(polys))]
#     sumY = [np.sum([polys[i][:, 1]]) for i in range(len(polys))]
#     centroids = [(sumX[i]/lengths[i], sumY[i]/lengths[i]) for i  in range(len(polys))]
#     return centroids
    
    
def centroidsFunc(polys):
    #polys = polys[1:]
    polys = list(filter(lambda poly: len(poly) > 0, polys))
    centroids = []
    for i in range(len(polys)):
        sumXY = [sum(point[coor] for point in polys[i]) for coor in [0, 1]]
        centroids.append([sumXY[coor]/len(polys[i]) for coor in [0, 1]])
    
    # otherwise extremely slow
    return list(filter(lambda centroid: centroid[0] < 100 and centroid[0] > -100, centroids))
        
     

for i in range(30):
    vor = Voronoi(centroidsFunc(polygons))
    polygons = []
    for region in vor.regions:
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            polygons.append(polygon)


            
            
# for y in range(len(polygons)):
#     for x in range(len(polygons[y])):
#         #print(polygons[y][x])
#         #print(list(itertools.permutations(polygons[y][x])))
#         print(list(itertools.starmap(lambda pointX, pointY: point[0] < 1 and point[0] > 0, list(itertools.permutations(polygons[y][x])))))     

    
def display(vor):
    #voronoi_plot_2d(vor, show_vertices=False, show_points=False)
    polygons = []
    for region in vor.regions:
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            polygons.append(polygon)
    #polygons = [point.tolist() for point in line for line in polygon for polygon in polygons]        
    polygons = list(filter(None, polygons[:]))
    #print(polygons[0:3])
            
    polygons_fixed = fix_polygons(polygons[:])
    #polygons_fixed = list(filter(None, polygons_fixed))
    
    # now we need to remove polygons that have values below 0 or above 1
    polygons_fixed = [polygon for polygon in polygons_fixed if [point[coor] > 1 or point[coor] < 0 for coor in [0, 1] for point in polygon].count(True) == 0]
    #  do something about corner polys
    
    polygons_fixed = list(filter(None, polygons_fixed[:]))
    # let's say we want to find the neighbors of the 45
    # for i in range(len(polygons_fixed)):
    #     for j in range(len(polygons_fixed)):
    #         if j == i:
    #             pass
    #         else:
    #             #print(len(centroidsFunc([polygons_fixed[i]])))
    #             dist = distance.euclidean(centroidsFunc([polygons_fixed[i]])[0], centroidsFunc([polygons_fixed[j]])[0])
    
    #################################################################################################################################################
    # id = 100
    # centroids_fixed = np.array([np.array(centroid) for centroid in centroidsFunc(polygons_fixed)])
    # print(centroids_fixed)
    # tree = neighbors.KDTree(centroids_fixed, leaf_size=2)
    # # if the polygon borders white (x/y = 0/1) then we want to decrease k
    # amountOfBorderPoints = [[point[coor] in [0, 1] for coor in [0, 1]].count(True) > 0 for point in polygons_fixed[id]].count(True)
    # dist, ind = tree.query([centroids_fixed[id]], k=len(polygons_fixed[id])+1-(amountOfBorderPoints//2))
    # 
    # for i in range(len(polygons_fixed[:])):
    #     if i in ind and i is not id:
    #         plt.fill(*zip(*polygons_fixed[i]), 'b')
    #     elif i == id:
    #         plt.fill(*zip(*polygons_fixed[i]), "g")
    #     else:
    #         plt.fill(*zip(*polygons_fixed[i]), "r")
    ##################################################################################################################################################
    scale = 0.25
    octaves = 10
    persistence = 0.4 # התגבשות
    # lower persistance = more grouped instead of break
    lacunarity = 2 # irregularity around 2 is good
    seed = np.random.randint(0,100)
    
    data_model = []
    centroids_fixed = np.array([np.array(centroid) for centroid in centroidsFunc(polygons_fixed)])
    tree = neighbors.KDTree(centroids_fixed, leaf_size=40)
    ellipse_gradient = np.loadtxt('{}_gradient.txt'.format(shapeOfGradient))        
    for i in range(len(polygons_fixed)):
        id = i
        # if the polygon borders white (x/y = 0/1) then we want to decrease k
        amountOfBorderPoints = [[point[coor] in [0, 1] for coor in [0, 1]].count(True) > 0 for point in polygons_fixed[id]].count(True)
        dist, ind = tree.query([centroids_fixed[id]], k=len(polygons_fixed[id])+1-(amountOfBorderPoints//2))    
        
        indicesInGradient = [int(math.floor(coor*600)) for coor in centroids_fixed[i]]
            
        data_model.append({
            "poly": polygons_fixed[i],
            "neighbor_indices": ind,
            "centroid": centroids_fixed[i],
            "index": i,
            "visited": False,
            "visitedForLake": False,
            "nation": None,
            # also heightmap
            "heightmap": noise.snoise2(
                centroids_fixed[i][0]/scale,
                centroids_fixed[i][1]/scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=seed)*ellipse_gradient[indicesInGradient[1]][indicesInGradient[0]]
        })
    
    threshold = -0.02
        
    for i in range(len(data_model)):
        data_model[i]["type"] = "ocean" if data_model[i]["heightmap"] < 0.05 + threshold else None
    #data_model_no_ocean = list(filter(lambda data: data['type'] is not "ocean", data_model))
    for i in range(len(data_model)):
        if data_model[i]["type"] != "ocean":
            #print(data_model[i]["neighbor_indices"])
            neighborIsOcean = [data_model[indice]["type"] == "ocean" for indice in data_model[i]["neighbor_indices"][0]]
            data_model[i]["type"] = "shore" if (neighborIsOcean.count(True) > 0) else "mainland"  
        
        
    for data in data_model:
        data["heightmap"] /= max([data["heightmap"] for data in data_model]) 
            
    colorsForHeight = np.array([
        [0.05, [65,105,225]],
        [0.055, [210,180,140]],
        [0.1, [238, 214, 175]],
        [0.25, [34,139,34]],
        [0.6, [0, 100, 0]],
        [0.7, [139, 137, 137]],
        [1, [255, 250, 250]]
    ])
    for i in range(len(colorsForHeight)):
        colorsForHeight[i][1] = np.array(colorsForHeight[:, 1][i])/255        
    
    
    ################## choose nation centers
    num_nations = 10
    cores = []
    data_model_no_ocean = list(filter(lambda data: data['type'] is not "ocean", data_model))
    for i in range(num_nations):
        if i is 0:
            cores.append(random.choice(data_model_no_ocean))
        else:
            while True:
                choice = random.choice(data_model_no_ocean)
                if np.sum(np.array([distance.euclidean(choice["centroid"], core["centroid"]) for core in cores]) < 0.1) == 0:
                    cores.append(choice)
                    break
    
    # now bfs
    queues = [collections.deque([root]) for root in cores]
    while np.sum(np.array([len(queue) for queue in queues])) > 0:
        #print(np.sum(np.array([len(queue) for queue in queues]))) 
        for i in range(len(queues)):
            try:
                vertex = queues[i].popleft()
            except:
                pass
            if vertex["visited"] is False:    
                vertex["visited"] = True
                vertex["nation"] = i
                queues[i].extend([data_model[indice] for indice in vertex["neighbor_indices"][0] if data_model[indice]["type"] is not "ocean" and data_model[indice]["visited"] is False and data_model[indice]["nation"] is None])
                           
    
    
            
    
            
    # colors = [list(np.random.choice(range(256), size=3)/255) for i in range(num_nations)]      
    # for i in range(len(data_model)):
    #     if data_model[i]["type"] is not "ocean":
    #         if data_model[i]["nation"] is not None:
    #             plt.fill(*zip(*data_model[i]["poly"]), color=colors[data_model[i]["nation"]])
    #         else:
    #             plt.fill(*zip(*data_model[i]["poly"]), 'g')      
    #     else:        
    #         plt.fill(*zip(*data_model[i]["poly"]), 'b')
    
    ####################################################################
    ocean_id_bfs = []
    data_model_ocean = list(filter(lambda data: data['type'] is "ocean", data_model))
    queue_ocean = collections.deque([random.choice(data_model_ocean)])
    while queue_ocean:
        vertex = queue_ocean.popleft()
        if vertex["visitedForLake"] is False:
            vertex["visitedForLake"] = True
            ocean_id_bfs.append(vertex["index"])
            indices = [indice for indice in vertex["neighbor_indices"][0]]
            queue_ocean.extend(list(filter(lambda dictt: dictt['index'] in indices, data_model_ocean)))        
    print(len(ocean_id_bfs))
    
    for data in data_model:
        if data["type"] is "ocean" and data["index"] not in ocean_id_bfs:
            data["type"] = "lake"
    
    # now for shores vs lake shores
    data_model_shores = list(filter(lambda data: data['type'] is "shore", data_model))
    for data in data_model_shores:
        indices = [indice for indice in data["neighbor_indices"][0]]
        neighborss = list(filter(lambda dictt: dictt['index'] in indices, data_model_ocean))
        types = [neigh["type"] for neigh in neighborss]
        print(types)
        if types.count("ocean") is 0:
            data["type"] = "lakeShore"
    
    
    for i in range(len(data_model)):
        #plt.fill(*zip(*data_model[i]["poly"]), 'g' if data_model[i]["heightmap"] > 0.05 else "b")
    
        if i in [core["index"] for core in cores]:
            plt.fill(*zip(*data_model[i]["poly"]), 'r')  
        else:    
            #plt.fill(*zip(*data_model[i]["poly"]), color=[color[1] for color in colorsForHeight if data_model[i]["heightmap"] < threshold + color[0]][0])
            if data_model[i]["type"] == "mainland":    
                plt.fill(*zip(*data_model[i]["poly"]), "g")
            elif data_model[i]["type"] == "ocean":
                plt.fill(*zip(*data_model[i]["poly"]), "b")
            elif data_model[i]["type"] == "shore":
                plt.fill(*zip(*data_model[i]["poly"]), "y")
            elif data_model[i]["type"] == "lake":
                plt.fill(*zip(*data_model[i]["poly"]), "w")
            elif data_model[i]["type"] == "lakeShore":    
                   plt.fill(*zip(*data_model[i]["poly"]), "r")
    
    # for i in range(len(polygons_fixed)):
    #     plt.fill(*zip(*polygons_fixed[i]), "r")
    
        #plt.fill(*zip(*polygons_fixed[i]), "r")           
    # fix the range of axes
    plt.xlim([0,1]), plt.ylim([0,1])    
    plt.show()
    
    
    #print(len(ellipse_gradient))
    #toimage(ellipse_gradient).show()
    

display(vor)
