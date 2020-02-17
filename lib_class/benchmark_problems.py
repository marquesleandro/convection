# ==========================================
# Code created by Leandro Marques at 12/2018
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# This code is used to compute boundary condition


import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

# OBS: para vetor devemos unir eixo x e y no mesmo vetor, logo usar np.row_stack([dirichlet_pts[1],dirichlet_pts[2]])


class Poiseuille:

 # ------------------------------------------------------------------------------------------------------
 # Use:

 # # Applying vx condition
 # condition_xvelocity = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_xvelocity.neumann_condition(mesh.neumann_edges[1])
 # condition_xvelocity.dirichlet_condition(mesh.dirichlet_pts[1])
 # condition_xvelocity.gaussian_elimination(LHS_vx0,mesh.neighbors_nodes)
 # vorticity_ibc = condition_xvelocity.ibc

 # # Applying vy condition
 # condition_yvelocity = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_yvelocity.neumann_condition(mesh.neumann_edges[2])
 # condition_yvelocity.dirichlet_condition(mesh.dirichlet_pts[2])
 # condition_yvelocity.gaussian_elimination(LHS_vy0,mesh.neighbors_nodes)

 # # Applying psi condition
 # condition_streamfunction = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_streamfunction.streamfunction_condition(mesh.dirichlet_pts[3],LHS_psi0,mesh.neighbors_nodes)
 # ------------------------------------------------------------------------------------------------------


 def __init__(_self, _nphysical, _npoints, _x, _y):
  _self.nphysical = _nphysical
  _self.npoints = _npoints
  _self.x = _x
  _self.y = _y
  _self.bc = np.zeros([_self.nphysical,1], dtype = float) 
  _self.benchmark_problem = 'Poiseuille'

  # Velocity vx condition
  _self.bc[0][0] = 0.0
  _self.bc[1][0] = 0.0
  _self.bc[2][0] = 1.0
  _self.bc[3][0] = 0.0

  # Velocity vy condition
  _self.bc[4][0] = 0.0
  _self.bc[5][0] = 0.0
  _self.bc[6][0] = 0.0
  _self.bc[7][0] = 0.0


 def neumann_condition(_self, _neumann_edges):
  _self.bc_neumann = np.zeros([_self.npoints,1], dtype = float) 
  _self.neumann_edges = _neumann_edges 
 
  for i in range(0, len(_self.neumann_edges)):
   line = _self.neumann_edges[i][0] - 1
   v1 = _self.neumann_edges[i][1] - 1
   v2 = _self.neumann_edges[i][2] - 1

   x = _self.x[v1] - _self.x[v2]
   y = _self.y[v1] - _self.y[v2]
   length = np.sqrt(x**2 + y**2)
  
   _self.bc_neumann[v1] += (_self.bc[line]*length) / 2. 
   _self.bc_neumann[v2] += (_self.bc[line]*length) / 2. 


 def dirichlet_condition(_self, _dirichlet_pts):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.dirichlet_pts = _dirichlet_pts
 

  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1

   _self.bc_1[v1] = _self.bc[line]
   _self.bc_1[v2] = _self.bc[line]

   _self.bc_neumann[v1] = 0.0 #Dirichlet condition is preferential
   _self.bc_neumann[v2] = 0.0 #Dirichlet condition is preferential

   _self.ibc.append(v1)
   _self.ibc.append(v2)
   
  _self.ibc = np.unique(_self.ibc)


 def gaussian_elimination(_self, _LHS0, _neighbors_nodes):
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.neighbors_nodes = _neighbors_nodes

  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 


 def streamfunction_condition(_self, _dirichlet_pts, _LHS0, _neighbors_nodes):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.dirichlet_pts = _dirichlet_pts
  _self.neighbors_nodes = _neighbors_nodes

  # Dirichlet condition
  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1

   if line == 8:
    _self.bc_1[v1] = 0.0
    _self.bc_1[v2] = 0.0
 
    _self.ibc.append(v1)
    _self.ibc.append(v2)

   elif line == 11:
    _self.bc_1[v1] = 1.0
    _self.bc_1[v2] = 1.0

    _self.ibc.append(v1)
    _self.ibc.append(v2)

   elif line == 10:
    _self.bc_1[v1] = _self.y[v1]
    _self.bc_1[v2] = _self.y[v2]

    _self.ibc.append(v1)
    _self.ibc.append(v2)

  _self.ibc = np.unique(_self.ibc)


  # Gaussian elimination for psi
  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 

class QuadPoiseuille:

 # ------------------------------------------------------------------------------------------------------
 # Use:

 # # Applying vx condition
 # condition_xvelocity = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_xvelocity.neumann_condition(mesh.neumann_edges[1])
 # condition_xvelocity.dirichlet_condition(mesh.dirichlet_pts[1])
 # condition_xvelocity.gaussian_elimination(LHS_vx0,mesh.neighbors_nodes)
 # vorticity_ibc = condition_xvelocity.ibc

 # # Applying vy condition
 # condition_yvelocity = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_yvelocity.neumann_condition(mesh.neumann_edges[2])
 # condition_yvelocity.dirichlet_condition(mesh.dirichlet_pts[2])
 # condition_yvelocity.gaussian_elimination(LHS_vy0,mesh.neighbors_nodes)

 # # Applying psi condition
 # condition_streamfunction = bc_apply.Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_streamfunction.streamfunction_condition(mesh.dirichlet_pts[3],LHS_psi0,mesh.neighbors_nodes)
 # ------------------------------------------------------------------------------------------------------


 def __init__(_self, _nphysical, _npoints, _x, _y):
  _self.nphysical = _nphysical
  _self.npoints = _npoints
  _self.x = _x
  _self.y = _y
  _self.bc = np.zeros([_self.nphysical,1], dtype = float) 
  _self.benchmark_problem = 'Quad Poiseuille'

  # Velocity vx condition
  _self.bc[0][0] = 0.0
  _self.bc[1][0] = 0.0
  _self.bc[2][0] = 1.0
  _self.bc[3][0] = 0.0

  # Velocity vy condition
  _self.bc[4][0] = 0.0
  _self.bc[5][0] = 0.0
  _self.bc[6][0] = 0.0
  _self.bc[7][0] = 0.0


 def neumann_condition(_self, _neumann_edges):
  _self.bc_neumann = np.zeros([_self.npoints,1], dtype = float) 
  _self.neumann_edges = _neumann_edges 
 
  for i in range(0, len(_self.neumann_edges)):
   line = _self.neumann_edges[i][0] - 1
   v1 = _self.neumann_edges[i][1] - 1
   v2 = _self.neumann_edges[i][2] - 1
   v3 = _self.neumann_edges[i][3] - 1

   x1 = _self.x[v1] - _self.x[v3]
   y1 = _self.y[v1] - _self.y[v3]
   length1 = np.sqrt(x1**2 + y1**2)

   x2 = _self.x[v3] - _self.x[v2]
   y2 = _self.y[v3] - _self.y[v2]
   length2 = np.sqrt(x2**2 + y2**2)
 

   _self.bc_neumann[v1] += (_self.bc[line]*length1) / 2.0 
   _self.bc_neumann[v2] += (_self.bc[line]*length2) / 2.0 
   _self.bc_neumann[v3] += ((_self.bc[line]*length1) / 2.0) + ((_self.bc[line]*length2) / 2.0) 


 def dirichlet_condition(_self, _dirichlet_pts):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.dirichlet_pts = _dirichlet_pts
 

  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1
   v3 = _self.dirichlet_pts[i][3] - 1

   _self.bc_1[v1] = _self.bc[line]
   _self.bc_1[v2] = _self.bc[line]
   _self.bc_1[v3] = _self.bc[line]

   _self.bc_neumann[v1] = 0.0 #Dirichlet condition is preferential
   _self.bc_neumann[v2] = 0.0 #Dirichlet condition is preferential
   _self.bc_neumann[v3] = 0.0 #Dirichlet condition is preferential

   _self.ibc.append(v1)
   _self.ibc.append(v2)
   _self.ibc.append(v3)
   
  _self.ibc = np.unique(_self.ibc)


 def gaussian_elimination(_self, _LHS0, _neighbors_nodes):
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.neighbors_nodes = _neighbors_nodes

  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 


 def streamfunction_condition(_self, _dirichlet_pts, _LHS0, _neighbors_nodes):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.dirichlet_pts = _dirichlet_pts
  _self.neighbors_nodes = _neighbors_nodes

  # Dirichlet condition
  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1
   v3 = _self.dirichlet_pts[i][3] - 1

   if line == 8:
    _self.bc_1[v1] = 0.0
    _self.bc_1[v2] = 0.0
    _self.bc_1[v3] = 0.0
 
    _self.ibc.append(v1)
    _self.ibc.append(v2)
    _self.ibc.append(v3)

   elif line == 11:
    _self.bc_1[v1] = 1.0
    _self.bc_1[v2] = 1.0
    _self.bc_1[v3] = 1.0

    _self.ibc.append(v1)
    _self.ibc.append(v2)
    _self.ibc.append(v3)

   elif line == 10:
    _self.bc_1[v1] = _self.y[v1]
    _self.bc_1[v2] = _self.y[v2]
    _self.bc_1[v3] = _self.y[v3]

    _self.ibc.append(v1)
    _self.ibc.append(v2)
    _self.ibc.append(v3)

  _self.ibc = np.unique(_self.ibc)


  # Gaussian elimination for psi
  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 


class Cavity:

 # ------------------------------------------------------------------------------------------------------
 # Use:

 # # Applying vx condition
 # condition_xvelocity = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_xvelocity.neumann_condition(mesh.neumann_edges[1])
 # condition_xvelocity.dirichlet_condition(mesh.dirichlet_pts[1])
 # condition_xvelocity.gaussian_elimination(LHS_vx0,mesh.neighbors_nodes)
 # vorticity_ibc = condition_xvelocity.ibc

 # # Applying vy condition
 # condition_yvelocity = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_yvelocity.neumann_condition(mesh.neumann_edges[2])
 # condition_yvelocity.dirichlet_condition(mesh.dirichlet_pts[2])
 # condition_yvelocity.gaussian_elimination(LHS_vy0,mesh.neighbors_nodes)

 # # Applying psi condition
 # condition_streamfunction = bc_apply.Cavity(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_streamfunction.neumann_condition(mesh.neumann_edges[3])
 # condition_streamfunction.dirichlet_condition(mesh.dirichlet_pts[3])
 # condition_streamfunction.gaussian_elimination(LHS_psi0,mesh.neighbors_nodes)
 # ------------------------------------------------------------------------------------------------------


 def __init__(_self, _nphysical, _npoints, _x, _y):
  _self.nphysical = _nphysical
  _self.npoints = _npoints
  _self.x = _x
  _self.y = _y
  _self.bc = np.zeros([_self.nphysical,1], dtype = float) 
  _self.benchmark_problem = 'Cavity'

  # Velocity vx condition - pg. 7
  _self.bc[0][0] = 0.0 
  _self.bc[1][0] = 1.0

  # Velocity vy condition - pg. 7
  _self.bc[2][0] = 0.0
  _self.bc[3][0] = 0.0

  # Streamline condition - pg. 7
  _self.bc[4][0] = 0.0



 def neumann_condition(_self, _neumann_edges):
  _self.bc_neumann = np.zeros([_self.npoints,1], dtype = float) 
  _self.neumann_edges = _neumann_edges 
 
  for i in range(0, len(_self.neumann_edges)):
   line = _self.neumann_edges[i][0] - 1
   v1 = _self.neumann_edges[i][1] - 1
   v2 = _self.neumann_edges[i][2] - 1

   x = _self.x[v1] - _self.x[v2]
   y = _self.y[v1] - _self.y[v2]
   length = np.sqrt(x**2 + y**2)
  
   _self.bc_neumann[v1] += (_self.bc[line]*length) / 2. 
   _self.bc_neumann[v2] += (_self.bc[line]*length) / 2. 


 def dirichlet_condition(_self, _dirichlet_pts):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.dirichlet_pts = _dirichlet_pts
 

  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1

   _self.bc_1[v1] = _self.bc[line]
   _self.bc_1[v2] = _self.bc[line]

   _self.bc_neumann[v1] = 0.0 #Dirichlet condition is preferential
   _self.bc_neumann[v2] = 0.0 #Dirichlet condition is preferential

   _self.ibc.append(v1)
   _self.ibc.append(v2)
   
  _self.ibc = np.unique(_self.ibc)


 def gaussian_elimination(_self, _LHS0, _neighbors_nodes):
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.neighbors_nodes = _neighbors_nodes

  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 

class Convection2D:

 # ------------------------------------------------------------------------------------------------------
 # Use:

 # # Applying c condition
 # condition_concentration = bc_apply.Convection(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_concentration.neumann_condition(mesh.neumann_edges[1])
 # condition_concentration.dirichlet_condition(mesh.dirichlet_pts[1])
 # condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)
 # condition_concentration.initial_condition()
 # c = np.copy(condition_concentration.c)
 # vx = np.copy(condition_concentration.vx)
 # vy = np.copy(condition_concentration.vy)
 # ------------------------------------------------------------------------------------------------------


 def __init__(_self, _nphysical, _npoints, _x, _y):
  _self.nphysical = _nphysical
  _self.npoints = _npoints
  _self.x = _x
  _self.y = _y
  _self.bc = np.zeros([_self.nphysical,1], dtype = float) 
  _self.benchmark_problem = 'Convection 2D'

  # Velocity c condition
  _self.bc[0][0] = 0.0
  _self.bc[1][0] = 0.0
  _self.bc[2][0] = 0.0
  _self.bc[3][0] = 0.0


 def neumann_condition(_self, _neumann_edges):
  _self.bc_neumann = np.zeros([_self.npoints,1], dtype = float) 
  _self.neumann_edges = _neumann_edges 
 
  for i in range(0, len(_self.neumann_edges)):
   line = _self.neumann_edges[i][0] - 1
   v1 = _self.neumann_edges[i][1] - 1
   v2 = _self.neumann_edges[i][2] - 1

   x = _self.x[v1] - _self.x[v2]
   y = _self.y[v1] - _self.y[v2]
   length = np.sqrt(x**2 + y**2)
  
   _self.bc_neumann[v1] += (_self.bc[line]*length) / 2. 
   _self.bc_neumann[v2] += (_self.bc[line]*length) / 2. 


 def dirichlet_condition(_self, _dirichlet_pts):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.dirichlet_pts = _dirichlet_pts
 

  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1

   _self.bc_1[v1] = _self.bc[line]
   _self.bc_1[v2] = _self.bc[line]

   _self.bc_neumann[v1] = 0.0 #Dirichlet condition is preferential
   _self.bc_neumann[v2] = 0.0 #Dirichlet condition is preferential

   _self.ibc.append(v1)
   _self.ibc.append(v2)
   
  _self.ibc = np.unique(_self.ibc)


 def gaussian_elimination(_self, _LHS0, _neighbors_nodes):
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.neighbors_nodes = _neighbors_nodes

  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 

 def initial_condition(_self):
  _self.c = np.copy(_self.bc_dirichlet)
  _self.vx = np.zeros([_self.npoints,1], dtype = float)
  _self.vy = np.zeros([_self.npoints,1], dtype = float)

  for i in range(0,_self.npoints):
   _self.vx[i] = _self.y[i]/3.0
   _self.vy[i] = -_self.x[i]/3.0

  a = 0.0
  b = -1.5
  r = 1.0

  for i in range(0, _self.npoints):
   x = _self.x[i] - a
   y = _self.y[i] - b
   lenght = np.sqrt(x**2 + y**2)

   if lenght < r:
    _self.c[i] = r**2 - (_self.x[i] - a)**2 - (_self.y[i] - b)**2


class Wave2D:

 # ------------------------------------------------------------------------------------------------------
 # Use:

 # # Applying c condition
 # condition_concentration = bc_apply.Convection(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_concentration.neumann_condition(mesh.neumann_edges[1])
 # condition_concentration.dirichlet_condition(mesh.dirichlet_pts[1])
 # condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)
 # condition_concentration.initial_condition()
 # c = np.copy(condition_concentration.c)
 # vx = np.copy(condition_concentration.vx)
 # vy = np.copy(condition_concentration.vy)
 # ------------------------------------------------------------------------------------------------------


 def __init__(_self, _nphysical, _npoints, _x, _y):
  _self.nphysical = _nphysical
  _self.npoints = _npoints
  _self.x = _x
  _self.y = _y
  _self.bc = np.zeros([_self.nphysical,1], dtype = float) 
  _self.benchmark_problem = 'Wave 2D'

  # Velocity c condition
  _self.bc[0][0] = 0.0
  _self.bc[1][0] = 0.0
  _self.bc[2][0] = 0.0
  _self.bc[3][0] = 0.0


 def neumann_condition(_self, _neumann_edges):
  _self.bc_neumann = np.zeros([_self.npoints,1], dtype = float) 
  _self.neumann_edges = _neumann_edges 
 
  for i in range(0, len(_self.neumann_edges)):
   line = _self.neumann_edges[i][0] - 1
   v1 = _self.neumann_edges[i][1] - 1
   v2 = _self.neumann_edges[i][2] - 1

   x = _self.x[v1] - _self.x[v2]
   y = _self.y[v1] - _self.y[v2]
   length = np.sqrt(x**2 + y**2)
  
   _self.bc_neumann[v1] += (_self.bc[line]*length) / 2. 
   _self.bc_neumann[v2] += (_self.bc[line]*length) / 2. 


 def dirichlet_condition(_self, _dirichlet_pts):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.dirichlet_pts = _dirichlet_pts
 

  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1

   _self.bc_1[v1] = _self.bc[line]
   _self.bc_1[v2] = _self.bc[line]

   _self.bc_neumann[v1] = 0.0 #Dirichlet condition is preferential
   _self.bc_neumann[v2] = 0.0 #Dirichlet condition is preferential

   _self.ibc.append(v1)
   _self.ibc.append(v2)
   
  _self.ibc = np.unique(_self.ibc)


 def gaussian_elimination(_self, _LHS0, _neighbors_nodes):
  _self.LHS = sps.csr_matrix.copy(_LHS0) #used csr matrix because LHS = lil_matrix + lil_matrix
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.neighbors_nodes = _neighbors_nodes

  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 

 def initial_condition(_self):
  _self.c = np.copy(_self.bc_dirichlet)
  _self.vx = np.zeros([_self.npoints,1], dtype = float)
  _self.vy = np.zeros([_self.npoints,1], dtype = float)

  for i in range(0,_self.npoints):
   _self.vx[i] = 0.0
   _self.vy[i] = 0.0

  a = 0.0
  b = 0.0
  r = 1.0

  for i in range(0, _self.npoints):
   x = _self.x[i] - a
   y = _self.y[i] - b
   lenght = np.sqrt(x**2 + y**2)

   if lenght < r:
    _self.c[i] = r**2 - (_self.x[i] - a)**2 - (_self.y[i] - b)**2







class Half_Poiseuille:

 # ------------------------------------------------------------------------------------------------------
 # Use:

 # # Applying vx condition
 # condition_xvelocity = bc_apply.Half_Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_xvelocity.neumann_condition(mesh.neumann_edges[1])
 # condition_xvelocity.dirichlet_condition(mesh.dirichlet_pts[1])
 # condition_xvelocity.gaussian_elimination(LHS_vx0,mesh.neighbors_nodes)
 # vorticity_ibc = condition_xvelocity.ibc

 # # Applying vy condition
 # condition_yvelocity = bc_apply.Half_Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_yvelocity.neumann_condition(mesh.neumann_edges[2])
 # condition_yvelocity.dirichlet_condition(mesh.dirichlet_pts[2])
 # condition_yvelocity.gaussian_elimination(LHS_vy0,mesh.neighbors_nodes)

 # # Applying psi condition
 # condition_streamfunction = bc_apply.Half_Poiseuille(mesh.nphysical,mesh.npoints,mesh.x,mesh.y)
 # condition_streamfunction.streamfunction_condition(mesh.dirichlet_pts[3],LHS_psi0,mesh.neighbors_nodes)
 # ------------------------------------------------------------------------------------------------------


 def __init__(_self, _nphysical, _npoints, _x, _y):
  _self.nphysical = _nphysical
  _self.npoints = _npoints
  _self.x = _x
  _self.y = _y
  _self.bc = np.zeros([_self.nphysical,1], dtype = float) 
  _self.benchmark_problem = 'Half Poiseuille'

  # Velocity vx condition
  _self.bc[0][0] = 0.0
  _self.bc[1][0] = 0.0
  _self.bc[2][0] = 1.0
  _self.bc[3][0] = 0.0

  # Velocity vy condition
  _self.bc[4][0] = 0.0
  _self.bc[5][0] = 0.0
  _self.bc[6][0] = 0.0
  _self.bc[7][0] = 0.0


 def neumann_condition(_self, _neumann_edges):
  _self.bc_neumann = np.zeros([_self.npoints,1], dtype = float) 
  _self.neumann_edges = _neumann_edges 
 
  for i in range(0, len(_self.neumann_edges)):
   line = _self.neumann_edges[i][0] - 1
   v1 = _self.neumann_edges[i][1] - 1
   v2 = _self.neumann_edges[i][2] - 1

   x = _self.x[v1] - _self.x[v2]
   y = _self.y[v1] - _self.y[v2]
   length = np.sqrt(x**2 + y**2)
  
   _self.bc_neumann[v1] += (_self.bc[line]*length) / 2. 
   _self.bc_neumann[v2] += (_self.bc[line]*length) / 2. 


 def dirichlet_condition(_self, _dirichlet_pts):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.dirichlet_pts = _dirichlet_pts
 

  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1

   _self.bc_1[v1] = _self.bc[line]
   _self.bc_1[v2] = _self.bc[line]

   _self.bc_neumann[v1] = 0.0 #Dirichlet condition is preferential
   _self.bc_neumann[v2] = 0.0 #Dirichlet condition is preferential

   _self.ibc.append(v1)
   _self.ibc.append(v2)
   
  _self.ibc = np.unique(_self.ibc)


 def gaussian_elimination(_self, _LHS0, _neighbors_nodes):
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.neighbors_nodes = _neighbors_nodes

  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 


 def streamfunction_condition(_self, _dirichlet_pts, _LHS0, _neighbors_nodes):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.dirichlet_pts = _dirichlet_pts
  _self.neighbors_nodes = _neighbors_nodes

  # Dirichlet condition
  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1
   v2 = _self.dirichlet_pts[i][2] - 1

   if line == 8:
    _self.bc_1[v1] = 0.0
    _self.bc_1[v2] = 0.0
 
    _self.ibc.append(v1)
    _self.ibc.append(v2)

   elif line == 11:
    _self.bc_1[v1] = 1.0
    _self.bc_1[v2] = 1.0

    _self.ibc.append(v1)
    _self.ibc.append(v2)

   elif line == 10:
    _self.bc_1[v1] = _self.y[v1]
    _self.bc_1[v2] = _self.y[v2]

    _self.ibc.append(v1)
    _self.ibc.append(v2)

  _self.ibc = np.unique(_self.ibc)


  # Gaussian elimination for psi
  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 
 
class Convection1D:

 # ------------------------------------------------------------------------------------------------------
 # Use:

 # # Applying c condition
 # condition_concentration = bc_apply.Convection(mesh.nphysical,mesh.npoints,mesh.x)
 # condition_concentration.neumann_condition(mesh.neumann_pts[1])
 # condition_concentration.dirichlet_condition(mesh.dirichlet_pts[1])
 # condition_concentration.gaussian_elimination(LHS_c0,mesh.neighbors_nodes)
 # condition_concentration.initial_condition()
 # c = np.copy(condition_concentration.c)
 # vx = np.copy(condition_concentration.vx)
 # ------------------------------------------------------------------------------------------------------


 def __init__(_self, _nphysical, _npoints, _x):
  _self.nphysical = _nphysical
  _self.npoints = _npoints
  _self.x = _x
  _self.bc = np.zeros([_self.nphysical,1], dtype = float) 
  _self.benchmark_problem = 'Convection 1D'

  # Velocity c condition
  _self.bc[0][0] = 0.0
  _self.bc[1][0] = 0.0


 def neumann_condition(_self, _neumann_pts):
  _self.bc_neumann = np.zeros([_self.npoints,1], dtype = float) 
  _self.neumann_pts = _neumann_pts 
 
  for i in range(0, len(_self.neumann_pts)):
   line = _self.neumann_pts[i][0] - 1
   v1 = _self.neumann_pts[i][1] - 1

   _self.bc_neumann[v1] += _self.bc[line]


 def dirichlet_condition(_self, _dirichlet_pts):
  _self.bc_dirichlet = np.zeros([_self.npoints,1], dtype = float) 
  _self.ibc = [] 
  _self.bc_1 = np.zeros([_self.npoints,1], dtype = float) #For scipy array solve
  _self.dirichlet_pts = _dirichlet_pts
 

  for i in range(0, len(_self.dirichlet_pts)):
   line = _self.dirichlet_pts[i][0] - 1
   v1 = _self.dirichlet_pts[i][1] - 1

   _self.bc_1[v1] = _self.bc[line]

   _self.bc_neumann[v1] = 0.0 #Dirichlet condition is preferential

   _self.ibc.append(v1)
   
  _self.ibc = np.unique(_self.ibc)


 def gaussian_elimination(_self, _LHS0, _neighbors_nodes):
  _self.LHS = sps.lil_matrix.copy(_LHS0)
  _self.bc_2 = np.ones([_self.npoints,1], dtype = float) 
  _self.neighbors_nodes = _neighbors_nodes

  for mm in _self.ibc:
   for nn in _self.neighbors_nodes[mm]:
    _self.bc_dirichlet[nn] -= float(_self.LHS[nn,mm]*_self.bc_1[mm])
    _self.LHS[nn,mm] = 0.0
    _self.LHS[mm,nn] = 0.0
   
   _self.LHS[mm,mm] = 1.0
   _self.bc_dirichlet[mm] = _self.bc_1[mm]
   _self.bc_2[mm] = 0.0
 

 def initial_condition(_self):
  _self.c = np.copy(_self.bc_dirichlet)
  _self.vx = np.zeros([_self.npoints,1], dtype = float)

  for i in range(0,_self.npoints):
   _self.vx[i] = 1.0


  for i in range(0, _self.npoints):
   x = _self.x[i]

   if x >= 2 and x <= 4:
    alpha = (x - 2)/(3 - 2)
    _self.c[i] = np.sin(alpha*(np.pi/2))







