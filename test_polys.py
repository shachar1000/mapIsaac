
from pprint import pprint
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt

def countX(lst, x):
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count 

polygons = [
# [
#     [0.1, 0.2],
#     [0.3, -3],
#     [0.6, -3],
#     [0.8, 0.3]
# ],
[
    [1, -1],
    [3, -1],
    [3, -3],
    [1, -3]
],
#[[0.34850857, 0.17631741], [0.331479  , 0.16892157], [0.3212696 , 0.17397311], [0.32596671, 0.2009352 ], [0.33629901, 0.20283098], [0.35058281, 0.18845382]],
[np.array([ 0.3920254 , -0.00246165]), [ 0.39779363, -0.04307638], [ 0.41450237, -0.04454898], [0.43288276, 0.0044384 ], [0.41896231, 0.0096856 ]]

]

# we can use itertools.combinations which is just permutations in a specific length but in order of appearance 
def adjacent(list, n):
    groups = [list[i:i + n] for i in range(len(list) + 1 - n)]
    groups.append([list[::-1][0], list[0]])
    return groups
    
def intersection(PA1, PB1, PA2, PB2):
    verticalStacking = np.vstack([PA1, PB1, PA2, PB2])
    add1AsZ = np.hstack((verticalStacking, np.ones((4, 1))))
    lines = [np.cross(add1AsZ[i], add1AsZ[i+1]) for i in [0, 2]]
    x, y, z = np.cross(lines[0], lines[1])
    return [x/z, y/z]
    
# now for each pair we want to check of only one in the pair has bad coors

def bad(point):
    numTrue = collections.Counter([point[coor] > 1 or point[coor] < 0 for coor in [0, 1]])[True]
    if numTrue is 1:
        lineDict = {
            "1" : [lambda point: point[0] > 1, [[1, 0], [1, 1]]],
            "2" : [lambda point: point[0] < 0, [[0, 0], [0, 1]]],
            "3" : [lambda point: point[1] > 1, [[0, 1], [1, 1]]],
            "4" : [lambda point: point[1] < 0, [[0, 0], [1, 0]]]
        }
        for key, value in lineDict.items():
            if lineDict[key][0](point) is True:
                return lineDict[key][1]
        # very clever debugging        
        print(point[1])
    else:
        return False


# print(bad([ 0.3920254,  -0.00246165]))
# print("reeee")

def fix_polygons(polygons):
    #polygons = [point.tolist() for polygon in polygons for point in polygon if type(point) == np.ndarray else point]
    
    for y, polygon in enumerate(polygons):
        for x, point in enumerate(polygon):
            polygons[y][x] = point.tolist() if type(point) == np.ndarray else point
    
    for i in range(len(polygons)):
        adj = adjacent(polygons[i], 2)
        #print(adj)
        
        for count, pair in enumerate(adj):
            # if only one false it means the other is array
            # collection.Counter doesn't work with matrix..... only lists
            if countX([bad(point) for point in pair], False) == 1:
                # we need to know which is the second line...
                # so we need to filter the false away from the list so we are left with the line
                
                #print("lol")
                #print([bad(point) for point in pair])
                
                
                line = list(filter(lambda x: type(x) == type([]), [bad(point) for point in pair]))[0]
                #line = line[0]
                
                #line = line[0]
                #line = [[0, 0],[1, 0]]
                inter = intersection(pair[0], pair[1], line[0], line[1])
                indexOfBad = [i for i in range(len(pair)) if bad(pair[i]) is False][0]
                adj[count][1-indexOfBad] = inter
            elif countX([bad(point) for point in pair], False) == 0:
                adj[count] = "removed"
  
        flat = list(itertools.chain(*[point for point in adj if point is not "removed"]))
        
        # this works only if order is not a concern...........
        #flat_no_duplicates = list(set(map(tuple, flat)))  
        #print(flat)
        flat_no_duplicates = [e for i,e in enumerate(flat) if e not in flat[:i]]
        
        polygons[i] = flat_no_duplicates
        #print(flat)  

        
    return polygons

if __name__ == '__main__':
    polygons_fixed = fix_polygons(polygons[:])
    #polygons_fixed = list(filter(None, polygons_fixed))
    #print(polygons_fixed)
    #print(polygons_fixed)
    for i in range(len(polygons)):
        plt.fill(*zip(*polygons[i]), "r")
        #try: # polygon may get removed in fixed which would result in index error since we are looping unfixed
        plt.fill(*zip(*polygons_fixed[i]), "b")
        # except:
        #     pass
    plt.xlim([0,1]), plt.ylim([0,1])    
    plt.show()
