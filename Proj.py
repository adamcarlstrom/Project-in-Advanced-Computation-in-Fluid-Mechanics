"""This program is an example file for the course"""
"""DD2365 Advanced Computation in Fluid Mechanics, """
"""KTH Royal Institute of Technology, Stockholm, Sweden."""

# Copyright (C) 2021 Johan Hoffman (jhoffman@kth.se)

# This file is part of the course DD2365 Advanced Computation in Fluid Mechanics
# KTH Royal Institute of Technology, Stockholm, Sweden
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License v2.

# This template is maintained by Johan Hoffman
# Please report problems to jhoffman@kth.se


# Load neccessary modules.
# from google.colab import files

import gc

import numpy as np
import time
from dolfin import *
from mshr import *
import dolfin.common.plotting as fenicsplot
from matplotlib import pyplot as plt

# Define rectangular domain
L = 4
H = 2

# Define circle
xc = 1.0
yc = 0.5*H
rc = 0.2

# Define subdomains (for boundary conditions)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L)

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H)

left = Left()
right = Right()
lower = Lower()
upper = Upper()

# Generate mesh (examples with and without a hole in the mesh)
resolution = 32
#mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
def build_mesh(xc,yc,resolution=32):

  mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(xc,yc),rc), resolution)

  # Local mesh refinement (specified by a cell marker)
  no_levels = 1
  for i in range(0,no_levels):
    cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
    for cell in cells(mesh):
      cell_marker[cell] = False
      p = cell.midpoint()
      if p.distance(Point(xc, yc)) < 0.5:
          cell_marker[cell] = True
    mesh = refine(mesh, cell_marker)

  return mesh

mesh = build_mesh(xc,yc,resolution)

# Define mesh functions (for boundary conditions)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
lower.mark(boundaries, 3)
upper.mark(boundaries, 4)

# plt.figure()
# plot(mesh)
# plt.show()


# Generate finite element spaces (for velocity and pressure)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)

V_mesh = VectorFunctionSpace(mesh, "CG", 1)
mesh_disp = Function(V_mesh)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

u_mesh = TrialFunction(V_mesh)
v_mesh = TestFunction(V_mesh)
a_mesh = inner(grad(u_mesh), grad(v_mesh))*dx
L_mesh = inner(Constant((0.0, 0.0)), v_mesh)*dx



# Define boundary conditions
class DirichletBoundaryLower(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class DirichletBoundaryUpper(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H)

class DirichletBoundaryLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class DirichletBoundaryRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

class DirichletBoundaryObjects(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (not near(x[0], 0.0)) and (not near(x[0], L)) and (not near(x[1], 0.0)) and (not near(x[1], H))

dbc_lower = DirichletBoundaryLower()
dbc_upper = DirichletBoundaryUpper()
dbc_left = DirichletBoundaryLeft()
dbc_right = DirichletBoundaryRight()
dbc_objects = DirichletBoundaryObjects()

# Examples of time dependent and stationary inflow conditions
#uin = Expression('4.0*x[1]*(1-x[1])', element = V.sub(0).ufl_element())
#uin = Expression('1.0 + 1.0*fabs(sin(t))', element = V.sub(0).ufl_element(), t=0.0)
uin = 1.0
bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
bcu_upp0 = DirichletBC(V.sub(0), 0.0, dbc_upper)
bcu_upp1 = DirichletBC(V.sub(1), 0.0, dbc_upper)
bcu_low0 = DirichletBC(V.sub(0), 0.0, dbc_lower)
bcu_low1 = DirichletBC(V.sub(1), 0.0, dbc_lower)
bcu_obj0 = DirichletBC(V.sub(0), 0.0, dbc_objects)
bcu_obj1 = DirichletBC(V.sub(1), 0.0, dbc_objects)

pin = Expression('5.0*fabs(sin(t))', element = Q.ufl_element(), t=0.0)
pout = 0.0
#bcp0 = DirichletBC(Q, pin, dbc_left)
bcp1 = DirichletBC(Q, pout, dbc_right)

#bcu = [bcu_in0, bcu_in1, bcu_upp0, bcu_upp1, bcu_low0, bcu_low1, bcu_obj0, bcu_obj1]
bcu = [bcu_in0, bcu_in1, bcu_upp1, bcu_low1, bcu_obj0, bcu_obj1]
bcp = [bcp1]

# Define measure for boundary integration
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
dx = Measure('dx', domain=mesh)

# Set viscosity
nu = 4.0e-3


# Define iteration functions
# (u0,p0) solution from previous time step
# (u1,p1) linearized solution at present time step
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)

# Define mesh deformation, mesh velocity = w/dt
freq = 0.1
t = 0.0
amp_x = 1.0e-2 # Move in x-direction
amp_y = 0.0
w = Expression(("amp_x*sin(2.0*pi*t*freq)*sin(pi*x[0]/L)","amp_y*sin(2.0*pi*t*freq-0.5*pi)*sin(pi*x[1]/H)"), L=L, H=H, t=t, amp_x=amp_x, amp_y=amp_y, freq=freq, element = V.ufl_element())


# w = mesh_disp / dt

# Set parameters for nonlinear and lienar solvers
num_nnlin_iter = 5
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Time step length
dt = 0.5*mesh.hmin()


# Define variational problem
R_in = 0.4  # Slightly larger than cylinder radius
R_out = 1.5 # Gives a wide buffer zone for the mesh to stretch smoothly

vx = 0.1
vy = 0.0

# Stabilization parameters
h = CellDiameter(mesh)
u_mag = sqrt(dot(u1,u1))
d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
d2 = h*u_mag

# Mean velocities for trapozoidal time stepping
um = 0.5*(u + u0)
um1 = 0.5*(u1 + u0)

# Momentum variational equation on residual form
Fu = inner((u - u0)/dt + grad(um)*(um1-w/dt), v)*dx - p1*div(v)*dx + nu*inner(grad(um), grad(v))*dx \
    + d1*inner((u - u0)/dt + grad(um)*(um1-w/dt) + grad(p1), grad(v)*(um1-w/dt))*dx + d2*div(um)*div(v)*dx
au = lhs(Fu)
Lu = rhs(Fu)

# Continuity variational equation on residual form
Fp = d1*inner((u1 - u0)/dt + grad(um1)*(um1-w/dt) + grad(p), grad(q))*dx + div(um1)*q*dx
ap = lhs(Fp)
Lp = rhs(Fp)


# Define the direction of the force to be computed
phi_x = 1.0 # drag
phi_y = 0.0 # lift

#psi_expression = Expression(("0.0","pow(x[0]-0.5,2.0) + pow(x[1]-1.0,2.0) - pow(0.2,2.0) < 1.e-5 ? 1. : 0."), element = V.ufl_element())
psi_expression = Expression(("near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=phi_x, phi_y=phi_y, element = V.ufl_element())
psi = interpolate(psi_expression, V)

Force = inner((u1 - u0)/dt + grad(um1)*um1, psi)*dx - p1*div(psi)*dx + nu*inner(grad(um1), grad(psi))*dx

#plt.figure()
#plot(psi, title="weight function psi")

# Force normalization
D = 2*rc
normalization = -2.0/D


# Open files to export solution to Paraview
file_u = File("results-NS/u.pvd")
file_p = File("results-NS/p.pvd")

# Set plot frequency
plot_time = 0
plot_freq = 10

# Force computation data
force_array = np.array(0.0)
force_array = np.delete(force_array, 0)
time = np.array(0.0)
time = np.delete(time, 0)
start_sample_time = 1.0

def move_mesh(mesh, current_xc, current_yc):
  # Create a Vector space for the mesh displacement
  V_mesh = VectorFunctionSpace(mesh, "CG", 1)

  # Define Boundary Conditions for the MESH
  # The outer walls of the channel do not move
  bc_walls = DirichletBC(V_mesh, Constant((0.0, 0.0)), "on_boundary && (x[1] < 1e-7 || x[1] > 2.0 - 1e-7 || x[0] < 1e-7 || x[0] > 4.0 - 1e-7)")

  # The cylinder boundary moves by the cylinder velocity * dt
  bc_cyl = DirichletBC(V_mesh, Constant((vx * dt, vy * dt)), dbc_objects) 

  # Solve the Laplace equation for smooth mesh displacement (u_mesh)
  u_mesh = TrialFunction(V_mesh)
  v_mesh = TestFunction(V_mesh)
  mesh_disp = Function(V_mesh)

  a_mesh = inner(grad(u_mesh), grad(v_mesh))*dx
  L_mesh = dot(Constant((0.0, 0.0)), v_mesh) * dx # zero source term

  solve(a_mesh == L_mesh, mesh_disp, [bc_walls, bc_cyl])

  # Physically move the mesh nodes
  ALE.move(mesh, mesh_disp)

  # Update cylinder coordinates for the next step
  current_xc += vx * dt
  current_yc += vy * dt
  return current_xc, current_yc

def remesh(distance_since_remesh_arg, current_xc_arg, current_yc_arg, u0_func, p0_func, u1_func, p1_func):
    global mesh, V, Q, u, p, v, q, au, Lu, ap, Lp, Force, bcu, bcp, ds, u0, p0, u1, p1, dx, w

    # Track how far the cylinder has moved since the last remesh
    distance_since_remesh_arg += sqrt((vx*dt)**2 + (vy*dt)**2)

    mesh_Change = False

    # If the cylinder has moved enough, trigger remesh
    if distance_since_remesh_arg > 0.5:
        mesh_Change = True
        print("Mesh quality degrading. Triggering Remesh and Interpolation...")

        # Build the pristine new mesh at the current location
        mesh = build_mesh(current_xc_arg, current_yc_arg)

        # Re-define dx for the new mesh
        dx = Measure('dx', domain=mesh)

        # Define Function Spaces on the new mesh
        V = VectorFunctionSpace(mesh, "P", 2)
        Q = FunctionSpace(mesh, "P", 1)

        # Transfer the data safely using Interpolate
        u0 = Function(V)
        u0_func.set_allow_extrapolation(True)
        u0.interpolate(u0_func)
        p0 = Function(Q)
        p0_func.set_allow_extrapolation(True)
        p0.interpolate(p0_func)

        # Re-initialize other Functions on the new mesh
        u1 = Function(V)
        p1 = Function(Q)

        # Rebuilding Boundaries
        bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
        bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
        bcu_upp0 = DirichletBC(V.sub(0), 0.0, dbc_upper)
        bcu_upp1 = DirichletBC(V.sub(1), 0.0, dbc_upper)
        bcu_low0 = DirichletBC(V.sub(0), 0.0, dbc_lower)
        bcu_low1 = DirichletBC(V.sub(1), 0.0, dbc_lower)
        bcu_obj0 = DirichletBC(V.sub(0), vx, dbc_objects)
        bcu_obj1 = DirichletBC(V.sub(1), vy, dbc_objects)

        pout = 0.0
        bcp1 = DirichletBC(Q, pout, dbc_right)

        bcu = [bcu_in0, bcu_in1, bcu_upp1, bcu_low1, bcu_obj0, bcu_obj1]
        bcp = [bcp1]

        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)

        dbc_left.mark(boundaries, 1)
        dbc_right.mark(boundaries, 2)
        dbc_upper.mark(boundaries, 3)
        dbc_lower.mark(boundaries, 4)
        dbc_objects.mark(boundaries, 5)

        ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

        # Rebuild Navier-Stokes weak forms
        u = TrialFunction(V)
        p = TrialFunction(Q)
        v = TestFunction(V)
        q = TestFunction(Q)

        h = CellDiameter(mesh)
        u_mag_recalc = sqrt(dot(u0,u0))
        d1_recalc = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag_recalc/h,2.0)))
        d2_recalc = h*u_mag_recalc

        um = 0.5*(u + u0)
        um1 = 0.5*(u1 + u0)

        Fu = inner((u - u0)/dt + grad(um)*(um1-w/dt), v)*dx - p1*div(v)*dx + nu*inner(grad(um), grad(v))*dx \
             + d1_recalc*inner((u - u0)/dt + grad(um)*(um1-w/dt) + grad(p1), grad(v)*(um1-w/dt))*dx + d2_recalc*div(um)*div(v)*dx
        au = lhs(Fu)
        Lu = rhs(Fu)

        Fp = d1_recalc*inner((u1 - u0)/dt + grad(um1)*(um1-w/dt) + grad(p_trial), grad(q))*dx + div(um1)*q*dx
        ap = lhs(Fp)
        Lp = rhs(Fp)

        psi_expression.xc = current_xc_arg
        psi_expression.yc = current_yc_arg

        psi = interpolate(psi_expression, V)
        Force = inner((u1 - u0)/dt + grad(um1)*um1, psi)*dx - p1*div(psi)*dx + nu*inner(grad(um1), grad(psi))*dx

        # Reset trackers and clean up memory
        distance_since_remesh_arg = 0.0

        del Fu, Fp, um, um1, psi
        gc.collect()
        
    return distance_since_remesh_arg, mesh_Change

# Time stepping
T = 11
t = dt
distance_since_remesh = 0
current_xc = xc
current_yc = yc
countDown = 0

while t < T + DOLFIN_EPS:

    #s = 'Time t = ' + repr(t)
    #print(s)

    pin.t = t
    #uin.t = t

    w.t = t
    current_xc, current_yc = move_mesh(mesh,current_xc,current_yc)
    psi_expression.xc = current_xc  # Update the bubble's X position
    psi_expression.yc = current_yc
    psi.interpolate(psi_expression)

    distance_since_remesh, mesh_Change = remesh(distance_since_remesh, current_xc,current_yc, u0, p0, u1, p1)

    # Solve non-linear problem
    k = 0
    while k < num_nnlin_iter:

        # Assemble momentum matrix and vector
        Au = assemble(au)
        bu = assemble(Lu)

        # Compute velocity solution
        [bc.apply(Au, bu) for bc in bcu]
        [bc.apply(u1.vector()) for bc in bcu]
        solve(Au, u1.vector(), bu, "bicgstab", "default")

        # Assemble continuity matrix and vector
        Ap = assemble(ap)
        bp = assemble(Lp)

        # Compute pressure solution
        [bc.apply(Ap, bp) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        solve(Ap, p1.vector(), bp, "bicgstab", prec)

        

        k += 1
        
    if mesh_Change:
      countDown = 20
    # Compute force
    F = assemble(Force)
    if (t > start_sample_time) and countDown <= 0:
      force_array = np.append(force_array, normalization*F)
      time = np.append(time, t)
    else:
      countDown -= 1

    if t > plot_time or mesh_Change:

        s = 'Time t = ' + repr(t)
        print(s)

        # Save solution to file
        # file_u << u1
        # file_p << p1
        
        plt.figure(figsize=(12, 10))

        # Plot solution
        plt.subplot(2, 2, 1)
        # plt.figure()
        plot(u1, title=f"Velocity:{t:.2f}")

        # plt.figure()
        plt.subplot(2, 2, 2)
        plot(p1, title=f"Pressure:{t:.2f}")

        # plt.figure()
        plt.subplot(2, 2, 3)
        plot(mesh, title=f"Mesh:{t:.2f}")

        plot_time += T/plot_freq

        # plt.figure()
        plt.subplot(2, 2, 4)
        plt.title(f"Force:{t:.2f}")
        plt.plot(time, force_array)

    # Update time step
    u0.assign(u1)
    t += dt

force_array = np.append(force_array, normalization*F)
print("Force Array:", force_array)
time = np.append(time, t)

s = 'Time t = ' + repr(t)
print(s)

# Save solution to file
# file_u << u1
# file_p << p1

plt.figure(figsize=(12, 10))

# Plot solution
plt.subplot(2, 2, 1)
# plt.figure()
plot(u1, title=f"Velocity:{t:.2f}")

# plt.figure()
plt.subplot(2, 2, 2)
plot(p1, title=f"Pressure:{t:.2f}")

# plt.figure()
plt.subplot(2, 2, 3)
plot(mesh, title=f"Mesh:{t:.2f}")

plot_time += T/plot_freq

# plt.figure()
plt.subplot(2, 2, 4)
plt.title(f"Force:{t:.2f}")
plt.plot(time, force_array)

plt.show()
#!tar -czvf results-NS.tar.gz results-NS
#files.download('results-NS.tar.gz')