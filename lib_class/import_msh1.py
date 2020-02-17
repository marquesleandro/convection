# ==========================================
# Code created by Leandro Marques at 01/2020
# Gesar Search Group
# State University of the Rio de Janeiro
# e-mail: marquesleandro67@gmail.com
# ==========================================

# This code is used to import .msh in simulator

import sys
import mesh


def Element1D(_directory, _mesh_name, _equation_number, polynomial_option):

 # Linear Element
 if polynomial_option == 1:
  msh = mesh.Linear1D(_directory, _mesh_name, _equation_number)
  msh.coord()
  msh.ien()

 # Quad Element
 elif polynomial_option == 2:
  msh = mesh.Quad1D(_directory, _mesh_name, _equation_number)
  msh.coord()
  msh.ien()

 else:
  print ""
  print " Error: Polynomial degree not found"
  print ""
  sys.exit()



 return msh.npoints, msh.nelem, msh.x, msh.IEN, msh.neumann_pts, msh.dirichlet_pts, msh.neighbors_nodes, msh.neighbors_elements, msh.far_neighbors_nodes, msh.far_neighbors_elements, msh.length_min, msh.GL, msh.nphysical 




def Element2D(_directory, _mesh_name, _equation_number, polynomial_option):

 # Linear Element
 if polynomial_option == 1:
  msh = mesh.Linear2D(_directory, _mesh_name, _equation_number)
  msh.coord()
  msh.ien()

 # Mini Element
 elif polynomial_option == 2:
  msh = mesh.Mini2D(_directory, _mesh_name, _equation_number)
  msh.coord()
  msh.ien()

 # Quad Element
 elif polynomial_option == 3:
  msh = mesh.Quad2D(_directory, _mesh_name, _equation_number)
  msh.coord()
  msh.ien()

 # Cubic Element
 elif polynomial_option == 4:
  msh = mesh.Cubic2D(_directory, _mesh_name, _equation_number)
  msh.coord()
  msh.ien()

 else:
  print ""
  print " Error: Polynomial degree not found"
  print ""
  sys.exit()



 return msh.npoints, msh.nelem, msh.x, msh.y, msh.IEN, msh.neumann_edges, msh.dirichlet_pts, msh.neighbors_nodes, msh.neighbors_elements, msh.far_neighbors_nodes, msh.far_neighbors_elements, msh.length_min, msh.GL, msh.nphysical 


