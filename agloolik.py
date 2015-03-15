#!/usr/bin/env/ python3
"""
Author: Adam G. Peddle
Contact: ap553@exeter.ac.uk
Version: 0.1
"""

import numpy as np
import scipy
from numpy import fft
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import pylab
import json
import logging
import collections.abc
import time
import sys
import pickle

class Control(dict):
    """
    Simple container class to hold relevant parameters for the control
    and execution of the code. Primarly populated from the control file.

    v1.0 AGP    28 Feb 2015
    """

    def __init__(self, controlFileName):

        with open(controlFileName) as controlFile:
            controlData = json.load(controlFile)

            super(Control,self).__setitem__('Equation',controlData['Control']['Equation'])
            super(Control,self).__setitem__('c',controlData['Control']['c'])

            super(Control,self).__setitem__('meshFile',controlData['Meshing']['meshfile'])
            super(Control,self).__setitem__('Dirichlet_functions',controlData['Meshing']['Dirichlet_functions'])
            super(Control,self).__setitem__('Neumann_functions',controlData['Meshing']['Neumann_functions'])

            super(Control,self).__setitem__('logLevel',controlData['Output']['loggingLevel'])
            super(Control,self).__setitem__('outFileStem',controlData['Output']['outFileStem'])
            super(Control,self).__setitem__('nPlots',controlData['Output']['nPlots'])
            try:
                super(Control,self).__setitem__('outSuffix','_' + controlData['Output']['outSuffix'])
            except KeyError:
                 super(Control,self).__setitem__('outSuffix','')

    #def __getitem__(self, name):
    #    safeFalses = ('outSuffix','c')
    #    if super(Control, self).__getitem__(name) or name in safeFalses:
    #        return super(Control, self).__getitem__(name)
    #    else:
    #        logging.error("Required parameter '{}' not present in control file".format(name))
    #        raise AttributeError("Required parameter '{}' not present in control file".format(name))

class Element:
    """
    Implements an individual element. The Geometry class is composed of some
    arbitrary number of Elements and calls though to computations performed
    at the local level.

    v1.0 AGP    28 Feb 2015
    """

    def __init__(self, corners, globalIndex):
        self.corners = corners
        self.globalIndex = globalIndex
        self.area = self.compute_area()

    def compute_local_K(self):
        """
        Computes and returns the local stiffness matrix for a linear triangular element.

        v1.0 AGP    28 Feb 2015
        """
        a = np.zeros(3)
        b = np.zeros(3)

        for k in range(3):
            a[k] = self.corners[(k+1)%3][1] - self.corners[(k+2)%3][1]
            b[k] = -(self.corners[(k+1)%3][0] - self.corners[(k+2)%3][0])

        return np.multiply((np.outer(a,a) + np.outer(b,b)),(0.25/self.area))

    def compute_local_M(self):
        """
        Computes and returns the local mass matrix for a linear triangular element.

        v1.0 AGP    03 March 2015
        """
        phi = np.zeros(3)
        midpoint = []
        a = []
        b = []
        c = []
        for k in range(3):
            midpoint.append([(self.corners[(k+1)%3][0] - self.corners[(k+2)%3][0])/2, \
                            (self.corners[(k+1)%3][1] - self.corners[(k+2)%3][1])/2])

            a.append( self.corners[(k+1)%3][1] - self.corners[(k+2)%3][1])
            b.append(-self.corners[(k+1)%3][0] + self.corners[(k+2)%3][0])
            c.append(self.corners[(k+1)%3][0]*self.corners[(k+2)%3][1] - \
                     self.corners[(k+2)%3][0]*self.corners[(k+1)%3][1])

        for i in range(3):
            for j in range(3):
                phi_node =a[i]*midpoint[j][0] + b[i]*midpoint[j][1] + c[i]
                phi[i] += phi_node

        np.multiply(phi,1/(2.0*self.area))
        return np.multiply(np.outer(phi,phi),self.area/3.0)


    def compute_local_F(self):
        """
        Computes and returns the local force matrix for a linear triangular element.

        v1.0 AGP    28 Feb 2015
        """

        return (self.area/3.0)*np.ones(3)

    def compute_area(self):
        """
        Returns the area of a triangular element.
        v1.0 AGP    28 Feb 2015
        """
        return 0.5*abs(self.corners[0][0]*(self.corners[1][1]- \
                            self.corners[2][1]) + \
                            self.corners[1][0]*(self.corners[2][1]- \
                            self.corners[0][1]) + \
                            self.corners[2][0]*(self.corners[0][1]- \
                            self.corners[1][1]))


class Geometry:
    """
    The geometry class contains the element objects which implement their own
    metrics. The geometry class permits an interface with the remainder of the
    code for the computation of the force/stiffness matrices as well as
    keeping track of some housekeeping (like number of elements).

    v1.0 AGP    28 February 2015
    """

    def __init__(self, control):
        with open(control['meshFile'], 'rb') as meshfile:
            try:
                meshData = pickle.load(meshfile)
            except EOFError:
                logging.severe("Meshfile is empty")
                raise

        self.nodes = meshData[0]
        self.triangles = meshData[1]
        self.nNodes = len(self.nodes)
        self.nTriangles = len(self.triangles)

        self.Dirichlet_functions = control['Dirichlet_functions']
        self.Neumann_functions = control['Neumann_functions']

        self._Dirichlet_nodes = meshData[2] #Internal to the element, includes sublists
        self.Dirichlet_nodes = [] #Publicly accessible, single vector
        for nodes in self._Dirichlet_nodes:
            self.Dirichlet_nodes += nodes

        self.Neumann_facets = meshData[3]

        self.elements = []

        for triangle in self.triangles:
            corners = (self.nodes[triangle[0]], \
                       self.nodes[triangle[1]], \
                       self.nodes[triangle[2]])
            self.elements.append(Element(corners,triangle))

    def assemble_stiffness(self):
        """
        Assembles the stiffness matrix. Calls through to local
        stiffness matrix computations applied at the element
        level.

        v1.0 AGP    28 Feb 2015
        """

        stiffness = np.zeros((self.nNodes, self.nNodes))

        for element in self.elements:
            K_0 = element.compute_local_K()
            for i in range(3):
                for j in range(3):
                    stiffness[element.globalIndex[i]][element.globalIndex[j]] += K_0[i][j]
        return stiffness

    def assemble_mass(self):
        """
        Assembles the mass matrix if necessary. Calls through to element level.

        v1.0 AGP    03 March 2015
        """
        mass = np.zeros((self.nNodes,self.nNodes))

        for element in self.elements:
            M_0 = element.compute_local_M()
            for i in range(3):
                for j in range(3):
                    mass[element.globalIndex[i]][element.globalIndex[j]] += M_0[i][j]
        return mass

    def assemble_force(self):
        """
        Assembles the force matrix. As with assemble_stiffness, calls
        through to element level for computation of local force matrices.

        v1.0 AGP    28 Feb 2015
        """

        force = np.zeros((self.nNodes,1))

        for element in self.elements:
            F_0 = element.compute_local_F()
            for i in range(3):
                force[element.globalIndex[i],0] += F_0[i]
        return force

    def enforce_Dirichlet(self,node):
        for i in range(len(self.Dirichlet_functions)):
            if node in self._Dirichlet_nodes[i]:
                x = self.nodes[node][0]
                y = self.nodes[node][1]
                outBC = eval(self.Dirichlet_functions[i])
                return outBC
        logging.warning("Dirichlet Boundary Condition node {} not found".format(node))

    def enforce_Neumann(self):
        Neu_out = np.zeros((self.nNodes,1))

        ctr = 0
        for sublist in self.Neumann_facets:
            for facet in sublist:
                length = np.sqrt((self.nodes[facet[0]][0] - self.nodes[facet[1]][0])**2 + \
                                 (self.nodes[facet[0]][1] - self.nodes[facet[1]][1])**2)

                g1 = 0
                for node in facet:
                    x = self.nodes[node][0]
                    y = self.nodes[node][1]
                    g1 += eval(self.Neumann_functions[ctr])

                integral = (length/4.0)*g1
                for node in facet:
                    Neu_out[node] += integral

            ctr += 1

        return Neu_out


def initialise(controlFileName):
    """
    Handles the initialisation of the program. Sets up logging structure,
    control, geometry (containing elements).

    v1.0 AGP 28 February 2015
    """

    #Create control structure with values from control file
    control = Control(controlFileName)

    with open(control['outFileStem'] + "_log.info", 'w'):
        pass #Clear an existing logfile
    #Then set up the logging
    try: #Check that the loglevel was set properly in the control file
        log_level = getattr(logging, control["logLevel"])
        logReset = False
    except AttributeError as ex:
        control["logLevel"] = logging.INFO
        logReset = True

    logging.basicConfig(filename = control['outFileStem'] + "_log.info", level = control['logLevel'], \
        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S' )

    if logReset: logging.warning('WARNING: Logging Level set to INFO')
    logging.info('Control structure created successfully')
    logging.getLogger().addHandler(logging.StreamHandler())

    geometry = Geometry(control) #Create geometry structure
    logging.info("Initialisation Complete")

    return control, geometry

def output(control, geometry, U):
    """
    Implements the output in the form of relevant plots and datasets (eventually).
    Logging is not implemented through this function. Rather, this is the post-processing
    phase of the computation.

    v1.0 AGP    28 Feb 2015
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(geometry.nodes[:,0],geometry.nodes[:,1],geometry.triangles,U,cmap = cm.coolwarm)
    plt.show()

def main(controlFileName):
    """
    Main function.

    v1.0 AGP    28 Feb 2015
        Calls to initialise, output, and system solution.
    """

    controlFileName = ''.join(controlFileName)
    control, geometry = initialise(controlFileName)

    LHS = geometry.assemble_stiffness()
    if control['c'] != 0:
        LHS += control['c']*geometry.assemble_mass()

    LHS = np.delete(LHS,geometry.Dirichlet_nodes,axis=0)
    LHS_shape = np.shape(LHS)
    Dirichlet_columns = np.zeros(shape=(LHS_shape[0],len(geometry.Dirichlet_nodes)))
    Dir_col = 0
    for column in geometry.Dirichlet_nodes:
        Dirichlet_columns[:,Dir_col] = LHS[:,column]
        Dir_col += 1
    LHS = np.delete(LHS,geometry.Dirichlet_nodes,axis=1)
    logging.info('LHS successfully assembled')

    Dirichlet_bcs = np.zeros((len(geometry.Dirichlet_nodes),1))
    for ctr in range(np.size(Dirichlet_bcs)):
        Dirichlet_bcs[ctr,0] = geometry.enforce_Dirichlet(geometry.Dirichlet_nodes[ctr])

    RHS = geometry.assemble_force() + geometry.enforce_Neumann()
    RHS = np.delete(RHS,geometry.Dirichlet_nodes)
    RHS = np.reshape(RHS,(-1,1)) #Reshapes RHS into an (n,1) array so it will agree below
    RHS -= np.dot(Dirichlet_columns,Dirichlet_bcs)
    logging.info('RHS successfully assembled')

    start = time.time()
    U = np.linalg.solve(LHS,RHS)
    end = time.time()
    logging.info('System solved in {:.8f} seconds'.format(end-start))
    U_full = np.zeros(geometry.nNodes)

    j = 0
    for i in range(geometry.nNodes):
        if i not in geometry.Dirichlet_nodes:
            U_full[i] = U[j]
            j+=1
        else:
            U_full[i] = geometry.enforce_Dirichlet(i)

    output(control, geometry, U_full)
    logging.info("Computation Completed Successfully!")

if __name__ == "__main__":
    main(sys.argv[1:])
    exit()

    try:
        main(sys.argv[1:])
    except FileNotFoundError:
        print("""Control File not found. Please specify the control file.
Proper calling is: python3 agloolik.py control_file_name.ctr""")


"TESTING"
