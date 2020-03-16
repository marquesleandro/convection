# ==========================================
# Code created by Leandro Marques at 03/2020
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# This code is used to import .vtk file


# Converting .msh in a python list

import numpy as np

def vtkfile_linear(_file): 

 vtklist = [] 
 with open(_file) as vtkfile:
   for line in vtkfile:
    row = line.split()
    vtklist.append(row[:])

 for i in range(0,len(vtklist)):
  for j in range(0,len(vtklist[i])):
   if vtklist[i][j] == "POINTS":
    npoints = int(vtklist[i][j+1])

    x = np.zeros([npoints,1], dtype = float)
    y = np.zeros([npoints,1], dtype = float)
    for k in range(0,npoints):
     x[k] = float(vtklist[i+k+1][0])
     y[k] = float(vtklist[i+k+1][1])
    continue  

   if vtklist[i][j] == "CELLS":
    nelem = int(vtklist[i][j+1])

    IEN = np.zeros([nelem,3], dtype = int)
    for e in range(0,nelem):
     IEN[e][0] = int(vtklist[i+e+1][1])
     IEN[e][1] = int(vtklist[i+e+1][2])
     IEN[e][2] = int(vtklist[i+e+1][3])
    continue 

   if vtklist[i][j] == "VECTORS":
    vx = np.zeros([npoints,1], dtype = float)
    vy = np.zeros([npoints,1], dtype = float)
    for k in range(0,npoints):
     vx[k] = float(vtklist[i+k+1][0])
     vy[k] = float(vtklist[i+k+1][1])
    continue  

   if vtklist[i][j] == "scalar1":
    scalar1 = np.zeros([npoints,1], dtype = float)
    for k in range(0,npoints):
     scalar1[k] = float(vtklist[i+k+2][0])
    continue  

   if vtklist[i][j] == "scalar2":
    scalar2 = np.zeros([npoints,1], dtype = float)
    for k in range(0,npoints):
     scalar2[k] = float(vtklist[i+k+2][0])
    continue  

   if vtklist[i][j] == "scalar3":
    scalar3 = np.zeros([npoints,1], dtype = float)
    for k in range(0,npoints):
     scalar3[k] = float(vtklist[i+k+2][0])
    continue  

 return npoints, nelem, IEN, x, y, vx, vy, scalar1, scalar2, scalar3
