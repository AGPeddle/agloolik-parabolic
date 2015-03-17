# agloolik-parabolic
FEM for SIMPLE 2nd order Parabolic PDEs

Author: Adam G. Peddle
Current version: 1.0
Date: 17 March 2015

agloolik-parabolic is a method for computation of second-order
linear parabolic PDEs in two spatial dimensions. It uses
linear triangular finite elements.

Invocation of the program requires a proper config file
as well as a mesh file, which may be created by the
*simpleMesher.py* file. Unfortunately, the mesher is still
rather user-unfriendly. Calling takes the form:

python3 agloolik controlFileName.ctr

Agloolik runs only with Python3 and is not backwards-compatible.
Agloolik depends on Numpy and Scipy. The meshing similarly
requires the meshpy package, available from:

http://mathema.tician.de/software/meshpy/

At the moment, there are no known bugs. It should, however, be noted
that agloolik has very limited functionality for parabolic
PDEs at the moment and so boundary conditions are restricted to
homogeneous Dirichlet. The initial condition may be freely specified
in the config file.
