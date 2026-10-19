#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 19:52:03 2026

@author: dfadda
"""

def get_axes(polygon):
    """Calculates perpendicular axes (normals) for all edges of a polygon."""
    axes = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]  # Next vertex (wraps around)
        
        # Edge vector
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Perpendicular normal vector (-y, x)
        normal = (-edge[1], edge[0])
        
        # Normalize the vector
        length = (normal[0]**2 + normal[1]**2) ** 0.5
        if length != 0:
            axes.append((normal[0] / length, normal[1] / length))
    return axes

def project(polygon, axis):
    """Projects all vertices of a polygon onto a single axis."""
    dots = [v[0] * axis[0] + v[1] * axis[1] for v in polygon]
    return min(dots), max(dots)

def overlap(proj1, proj2):
    """Checks if two 1D projections overlap."""
    return min(proj1[1], proj2[1]) >= max(proj1[0], proj2[0])

def convex_polygons_intersect(poly1, poly2):
    """Returns True if two convex polygons intersect using SAT."""
    # Combine the normal axes from both shapes to check against
    axes = get_axes(poly1) + get_axes(poly2)
    
    for axis in axes:
        proj1 = project(poly1, axis)
        proj2 = project(poly2, axis)
        
        # If we find even one separating axis, they definitely DO NOT intersect
        if not overlap(proj1, proj2):
            return False
            
    return True