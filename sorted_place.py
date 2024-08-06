import numpy as np

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def nearest_neighbor_tsp(start, points):
    visited = [start]
    unvisited = points[:]
    current = start
    
    while unvisited:
        next_point = min(unvisited, key=lambda p: distance(current, p))
        unvisited.remove(next_point)
        visited.append(next_point)
        current = next_point
        
    return visited
