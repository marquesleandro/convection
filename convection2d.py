# =======================
# Importing the libraries
# =======================

import sys
directory = './lib_class'
sys.path.insert(0, directory)

from tqdm import tqdm
from time import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import search_file
import import_msh
import assembly
import benchmark_problems
import semi_lagrangian 
import export_vtk
import relatory




print '''
               COPYRIGHT                    
 ======================================
 Simulator: %s
 created by Leandro Marques at 02/2019
 e-mail: marquesleandro67@gmail.com
 Gesar Search Group
 State University of the Rio de Janeiro
 ======================================
''' %sys.argv[0]





print ' ------'
print ' INPUT:'
print ' ------'
print ""


# ----------------------------------------------------------------------------
benchmark_problem = 'Convection 1D'
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
print ' (1) - Linear Element'
print ' (2) - Mini Element'
print ' (3) - Quadratic Element'
print ' (4) - Cubic Element'
polynomial_option = int(raw_input(" Enter polynomial degree option above: "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
print '  3 Gauss Points'
print '  4 Gauss Points'
print '  6 Gauss Points'
print ' 12 Gauss Points'
gausspoints = int(raw_input(" Enter Gauss Points Number option above: "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
print ' (1) - Taylor Galerkin Scheme'
print ' (2) - Semi Lagrangian Scheme'
scheme_option = int(raw_input(" Enter simulation scheme option above: "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
nt = int(raw_input(" Enter number of time interations (nt): "))
directory_save = raw_input(" Enter folder name to save simulations: ")
print ""
# ----------------------------------------------------------------------------




print ' ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

# Linear Element
if polynomial_option == 1:
 mesh_name = 'malha_convection2D.msh'
 equation_number = 1

 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Linear2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()


# Mini Element
elif polynomial_option == 2:
 mesh_name = 'malha_convection2D.msh'
 equation_number = 1

 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Mini2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()

# Quad Element
elif polynomial_option == 3:
 mesh_name = 'malha_convection2D_quad.msh'
 equation_number = 1

 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Quad2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()

# Cubic Element
elif polynomial_option == 4:
 mesh_name = 'malha_convection2D_cubic.msh'
 equation_number = 1

 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Cubic2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()



npoints                = msh.npoints
nelem                  = msh.nelem
x                      = msh.x
y                      = msh.y
IEN                    = msh.IEN
neumann_edges            = msh.neumann_edges
dirichlet_pts          = msh.dirichlet_pts
neighbors_nodes        = msh.neighbors_nodes
neighbors_elements     = msh.neighbors_elements
far_neighbors_nodes    = msh.far_neighbors_nodes
far_neighbors_elements = msh.far_neighbors_elements
length_min             = msh.length_min
GL                     = msh.GL
nphysical              = msh.nphysical 


Re = 10000.0
Sc = 10000.0
CFL = 0.5
dt = float(CFL*length_min)


end_time = time()
import_mesh_time = end_time - start_time
print ' time duration: %.1f seconds' %import_mesh_time
print ""





print ' ---------'
print ' ASSEMBLY:'
print ' ---------'

start_time = time()

Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy, polynomial_order = assembly.Element2D(polynomial_option, GL, npoints, nelem, IEN, x, y, gausspoints)


end_time = time()
assembly_time = end_time - start_time
print ' time duration: %.1f seconds' %assembly_time
print ""





print ' --------------------------------'
print ' INITIAL AND BOUNDARY CONDITIONS:'
print ' --------------------------------'

start_time = time()

condition_concentration_LHS0 = sps.lil_matrix.copy(M)/dt
condition_concentration = benchmark_problems.Convection2D(nphysical,npoints,x,y)
condition_concentration.neumann_condition(neumann_edges[1])
condition_concentration.dirichlet_condition(dirichlet_pts[1])
condition_concentration.gaussian_elimination(condition_concentration_LHS0,neighbors_nodes)
condition_concentration.initial_condition()

LHS = condition_concentration.LHS
bc_dirichlet = condition_concentration.bc_dirichlet
bc_neumann = condition_concentration.bc_neumann
bc_2 = condition_concentration.bc_2
c = np.copy(condition_concentration.c)
vx = np.copy(condition_concentration.vx)
vy = np.copy(condition_concentration.vy)




end_time = time()
bc_apply_time = end_time - start_time
print ' time duration: %.1f seconds' %bc_apply_time
print ""





print ' -----------------------------'
print ' PARAMETERS OF THE SIMULATION:'
print ' -----------------------------'

print ' Mesh: %s' %mesh_name
print ' Number of equation: %s' %equation_number
print ' Number of nodes: %s' %npoints
print ' Number of elements: %s' %nelem
print ' Smallest edge length: %f' %length_min
print ' Time step: %s' %dt
print ' Number of time iteration: %s' %nt
print ' Reynolds number: %s' %Re
print ' Schmidt number: %s' %Sc
print ""




print ' ----------------------------'
print ' SOLVE THE LINEARS EQUATIONS:'
print ' ----------------------------'
print ""
print ' Saving simulation in %s' %directory_save
print ""



start_time = time()


# Taylor Galerkin
if scheme_option == 1:
 scheme_name = 'Taylor Galerkin'

 for t in tqdm(range(0, nt)):

  # ------------------------ Export VTK File --------------------------------------
  save = export_vtk.Linear1D(x,IEN,npoints,nelem,c,c,c,vx,vx)
  save.create_dir(directory_save)
  save.saveVTK(directory_save + str(t))
  # -------------------------------------------------------------------------------

  # -------------------------------- Solver ---------------------------------------
  A = np.copy(M)/dt
  RHS = sps.lil_matrix.dot(A,c) - np.multiply(vx,sps.lil_matrix.dot(Gx,c))\
                                - np.multiply(vy,sps.lil_matrix.dot(Gy,c))\
                                - (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,c))\
                                                         + np.multiply(vy,sps.lil_matrix.dot(Kyx,c))))\
                                - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,c))\
                                                         + np.multiply(vy,sps.lil_matrix.dot(Kyy,c))))

  RHS = RHS + (1.0/(Re*Sc))*bc_neumann
  RHS = np.multiply(RHS,bc_2)
  RHS = RHS - bc_dirichlet
  
  c = scipy.sparse.linalg.cg(LHS,RHS,c, maxiter=1.0e+05, tol=1.0e-05)
  c = c[0].reshape((len(c[0]),1))
  # -------------------------------------------------------------------------------


# Semi Lagrangian Linear
elif scheme_option == 2:
 scheme_name = 'Semi Lagrangian'

 if polynomial_option == 1: #Linear Element
  for t in tqdm(range(0, nt)):

   # ------------------------ Export VTK File --------------------------------------
   save = export_vtk.Linear2D(x,y,IEN,npoints,nelem,c,c,c,vx,vx)
   save.create_dir(directory_save)
   save.saveVTK(directory_save + str(t))
   # -------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   c_d = semi_lagrangian.Linear2D(npoints, neighbors_elements, IEN, x, y, vx, vy, dt, c)

   A = np.copy(M)/dt
   RHS = sps.lil_matrix.dot(A,c_d)
 
   RHS = RHS + (1.0/(Re*Sc))*bc_neumann
   RHS = np.multiply(RHS,bc_2)
   RHS = RHS - bc_dirichlet

   c = scipy.sparse.linalg.cg(LHS,RHS,c, maxiter=1.0e+05, tol=1.0e-05)
   c = c[0].reshape((len(c[0]),1))
   # -------------------------------------------------------------------------------

 elif polynomial_option == 2: #Mini Element
  for t in tqdm(range(0, nt)):

   # ------------------------ Export VTK File --------------------------------------
   save = export_vtk.Linear2D(x,y,IEN,npoints,nelem,c,c,c,vx,vx)
   save.create_dir(directory_save)
   save.saveVTK(directory_save + str(t))
   # -------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   c_d = semi_lagrangian.Mini2D(npoints, neighbors_elements, IEN, x, y, vx, vy, dt, c)

   A = np.copy(M)/dt
   RHS = sps.lil_matrix.dot(A,c_d)
 
   RHS = RHS + (1.0/(Re*Sc))*bc_neumann
   RHS = np.multiply(RHS,bc_2)
   RHS = RHS - bc_dirichlet

   c = scipy.sparse.linalg.cg(LHS,RHS,c, maxiter=1.0e+05, tol=1.0e-05)
   c = c[0].reshape((len(c[0]),1))
   # -------------------------------------------------------------------------------

 elif polynomial_option == 3: #Quad Element
  for t in tqdm(range(0, nt)):

   # ------------------------ Export VTK File --------------------------------------
   save = export_vtk.Linear2D(x,y,IEN,npoints,nelem,c,c,c,vx,vx)
   save.create_dir(directory_save)
   save.saveVTK(directory_save + str(t))
   # -------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   c_d = semi_lagrangian.Quad2D(npoints, neighbors_elements, IEN, x, y, vx, vy, dt, c)

   A = np.copy(M)/dt
   RHS = sps.lil_matrix.dot(A,c_d)
 
   RHS = RHS + (1.0/(Re*Sc))*bc_neumann
   RHS = np.multiply(RHS,bc_2)
   RHS = RHS - bc_dirichlet

   c = scipy.sparse.linalg.cg(LHS,RHS,c, maxiter=1.0e+05, tol=1.0e-05)
   c = c[0].reshape((len(c[0]),1))
   # -------------------------------------------------------------------------------

 elif polynomial_option == 4: #Cubic Element
  for t in tqdm(range(0, nt)):

   # ------------------------ Export VTK File --------------------------------------
   save = export_vtk.Linear2D(x,y,IEN,npoints,nelem,c,c,c,vx,vx)
   save.create_dir(directory_save)
   save.saveVTK(directory_save + str(t))
   # -------------------------------------------------------------------------------

   # -------------------------------- Solver ---------------------------------------
   c_d = semi_lagrangian.Cubic2D(npoints, neighbors_elements, IEN, x, y, vx, vy, dt, c)

   A = np.copy(M)/dt
   RHS = sps.lil_matrix.dot(A,c_d)
 
   RHS = RHS + (1.0/(Re*Sc))*bc_neumann
   RHS = np.multiply(RHS,bc_2)
   RHS = RHS - bc_dirichlet

   c = scipy.sparse.linalg.cg(LHS,RHS,c, maxiter=1.0e+05, tol=1.0e-05)
   c = c[0].reshape((len(c[0]),1))
   # -------------------------------------------------------------------------------

else:
 print ""
 print " Error: Simulator Scheme not found"
 print ""
 sys.exit()




end_time = time()
solution_time = end_time - start_time
print ' time duration: %.1f seconds' %solution_time
print ""





print ' ----------------'
print ' SAVING RELATORY:'
print ' ----------------'
print ""
print ' End simulation. Relatory saved in %s' %directory_save
print ""

# -------------------------------- Export Relatory ---------------------------------------
relatory.export(directory_save, sys.argv[0], benchmark_problem, scheme_name, mesh_name, equation_number, npoints, nelem, length_min, dt, nt, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints)

