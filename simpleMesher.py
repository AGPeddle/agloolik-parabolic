from __future__ import division
from __future__ import absolute_import
import pickle
import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
from six.moves import range
import matplotlib.pyplot as pt

def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]

def main():
    points = []
    facets = []

    circ_start = len(points)
    points.extend(
            (3 * np.cos(angle), 3 * np.sin(angle))
            for angle in np.linspace(0, 2*np.pi, 30, endpoint=False))

    #boundaryPts1 = points[0:20]
    #boundaryFacets1 = []
    #boundaryFacets1.extend(round_trip_connect(21,31))
    boundaryPts1 = points[0:31]
    boundaryFacets1 = []
    facets.extend(round_trip_connect(circ_start, len(points)-1))

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0)/3
        max_area = 0.001 + (la.norm(bary, np.inf)-1)*0.1
        return bool(area > max_area)

    def refinement2(vertices, area):
        return(area > 0.25)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    mesh = triangle.build(info, refinement_func = refinement2)

    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    boundary_points1 = []
    i = 0
    for meshPt in mesh_points:
        for point in boundaryPts1:
            if point == tuple(meshPt):
                boundary_points1.append(i)
                break
        i+=1

    mesh_out = [mesh_points, mesh_tris, [boundary_points1],[boundaryFacets1]]
    with open('mesh2.msh','wb') as outFile:
        pickle.dump(mesh_out,outFile)

    pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    pt.show()

if __name__ == "__main__":
        main()
