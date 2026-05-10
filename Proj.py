# Based on the FEniCS Navier-Stokes ALE file by Johan Hoffman (
# KTH DD2365 Advanced Computation in Fluid Mechanics ) 
# Further enhanced by author Adam Carlström (adacar@ug.kth.se) 2026

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
import logging

logging.getLogger('FEniCS').setLevel(logging.WARNING)
set_log_level(LogLevel.WARNING) # to suppress dolfin output

# Define rectangular domain
L = 8
H = 4

# Circular dimensions
# xc = 1.0
# yc = 0.5*H
# rc = 0.2

# Train dimensions
xc = 2.0
yc = 0.75*H
yc2 = H - yc
t_L = 2.0
t_H = 0.2
t_R = t_H/2.0

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
resolution = 64
#mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
def build_mesh(xc_top,xc_bot):
  global yc,yc2,t_L,t_H,t_R,L,H,resolution
#   mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(xc,yc),rc), resolution)
  
  p1 = Point(xc_top - t_L/2.0, yc - t_H/2.0)
  p2 = Point(xc_top + t_L/2.0, yc + t_H/2.0)
  body = Rectangle(p1, p2)
  
  head_center = Point(xc_top + t_L/2.0, yc)
  head = Circle(head_center, t_R)
  
  train = body + head
  
  p1 = Point(xc_bot - t_L/2.0, yc2 - t_H/2.0)
  p2 = Point(xc_bot + t_L/2.0, yc2 + t_H/2.0)
  body = Rectangle(p1, p2)
  
  head_center = Point(xc_bot - t_L/2.0, yc2)
  head = Circle(head_center, t_R)
  
  train2 = body + head
  
  domain = Rectangle(Point(0.0, 0.0), Point(L, H)) - train - train2
  mesh = generate_mesh(domain, resolution)

  # Local mesh refinement (specified by a cell marker)
  no_levels = 0 
  buffer = 0.25
  for i in range(0,no_levels):
    cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
    cell_marker.set_all(False)
    for cell in cells(mesh):
      p = cell.midpoint()
      px,py = p[0],p[1]
    #   if p.distance(Point(xc, yc)) < 0.5:
      if    ((xc - t_L/2.0 - buffer*3) < px < (xc + t_L/2.0 + t_R + buffer) and 
            (yc - t_H/2.0 - buffer) < py < (yc + t_H/2.0 + buffer)):
          cell_marker[cell] = True
    mesh = refine(mesh, cell_marker)

  return mesh

mesh = build_mesh(xc,L-xc)

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

vx = 0.1
vy = 0.0


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

y_mid = 0.5 * (yc + yc2)   # = 0.5 * (3.0 + 1.0) = 2.0  — stable, never changes

class DirichletBoundaryTopTrain(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary
                and not near(x[0], 0.0) and not near(x[0], L)
                and not near(x[1], 0.0) and not near(x[1], H)
                and x[1] > y_mid)

class DirichletBoundaryBotTrain(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary
                and not near(x[0], 0.0) and not near(x[0], L)
                and not near(x[1], 0.0) and not near(x[1], H)
                and x[1] < y_mid)

dbc_objects_top = DirichletBoundaryTopTrain()
dbc_objects_bot = DirichletBoundaryBotTrain()

dbc_lower = DirichletBoundaryLower()
dbc_upper = DirichletBoundaryUpper()
dbc_left = DirichletBoundaryLeft()
dbc_right = DirichletBoundaryRight()
dbc_objects = DirichletBoundaryObjects()

# The train pushes the fluid
bcu_top0 = DirichletBC(V.sub(0),  vx, dbc_objects_top)
bcu_top1 = DirichletBC(V.sub(1), 0.0, dbc_objects_top)
bcu_bot0 = DirichletBC(V.sub(0), -vx, dbc_objects_bot)
bcu_bot1 = DirichletBC(V.sub(1), 0.0, dbc_objects_bot)
bcu = [bcu_top0, bcu_top1, bcu_bot0, bcu_bot1]

# Create a subdomain for ALL outer walls to apply pressure
class DirichletBoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[0], L) or near(x[1], 0.0) or near(x[1], H))

dbc_outer = DirichletBoundaryOuter()
pout = 0.0

# Apply 0 pressure to all outer walls (Do-Nothing open boundary)
bcp_outer = DirichletBC(Q, pout, dbc_outer)
bcp = [bcp_outer]

# Define measure for boundary integration
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
dx = Measure('dx', domain=mesh)

# Set viscosity
nu = 8.0e-5

# Re = U*D/nu = 0.1*0.4/8e-6 = 500


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
# amp_x = 1.0e-2 # Move in x-direction
# amp_y = 0.0
# w = Expression(("amp_x*sin(2.0*pi*t*freq)*sin(pi*x[0]/L)","amp_y*sin(2.0*pi*t*freq-0.5*pi)*sin(pi*x[1]/H)"), L=L, H=H, t=t, amp_x=amp_x, amp_y=amp_y, freq=freq, element = V.ufl_element())

w = Function(V) # for global mesh deformation

# Set parameters for nonlinear and lienar solvers
num_nnlin_iter = 5
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Time step length
dt = 0.5*mesh.hmin()


# Define variational problem
R_in = 0.4  # Slightly larger than cylinder radius
R_out = 1.5 # Gives a wide buffer zone for the mesh to stretch smoothly

# Stabilization parameters
h = CellDiameter(mesh)
u_mag = sqrt(dot(u1,u1))
d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
d2 = h*u_mag

# Mean velocities for trapozoidal time stepping
um = 0.5*(u + u0)
um1 = 0.5*(u1 + u0)

# Momentum variational equation on residual form
Fu = inner((u - u0)/dt + grad(um)*(um1-w), v)*dx - p1*div(v)*dx + nu*inner(grad(um), grad(v))*dx \
    + d1*inner((u - u0)/dt + grad(um)*(um1-w) + grad(p1), grad(v)*(um1-w))*dx + d2*div(um)*div(v)*dx
au = lhs(Fu)
Lu = rhs(Fu)

# Continuity variational equation on residual form
Fp = d1*inner((u1 - u0)/dt + grad(um1)*(um1-w) + grad(p), grad(q))*dx + div(um1)*q*dx
ap = lhs(Fp)
Lp = rhs(Fp)


# Define the direction of the force to be computed
phi_x = 1.0 # drag
phi_y = 0.0 # lift

# Create an empty function space for psi
psi = Function(V)

# Apply a value of 1.0 (phi_x) in the x-direction directly to the nodes on the train boundary
bc_psi_x = DirichletBC(V.sub(0), Constant(phi_x), dbc_objects_top)
bc_psi_y = DirichletBC(V.sub(1), Constant(phi_y), dbc_objects_top)

# Apply these boundary conditions to our empty psi vector. 
# This automatically makes psi = 1 on the train boundary and 0 everywhere else!
bc_psi_x.apply(psi.vector())
bc_psi_y.apply(psi.vector())

# Force = inner((u1 - u0)/dt + grad(um1)*um1, psi)*dx - p1*div(psi)*dx + nu*inner(grad(um1), grad(psi))*dx
Force = inner((u1 - u0)/dt + grad(um1)*(um1-w),psi)*dx - p1*div(psi)*dx + nu*inner(grad(um1),grad(psi))*dx


#plt.figure()
#plot(psi, title="weight function psi")

# Force normalization
D = t_H
normalization = -2.0/D


# Open files to export solution to Paraview
file_u = File("results-NS/u.pvd")
file_p = File("results-NS/p.pvd")

# Set plot frequency
plot_time = 0
plot_freq = 18

# Force computation data
force_array = np.array(0.0)
force_array = np.delete(force_array, 0)
time = np.array(0.0)
time = np.delete(time, 0)
start_sample_time = 1.0

def move_mesh(mesh, current_xc_top, current_xc_bot):
    global w, V

    pad_x = 0.8
    pad_y = 0.4
    half_L = t_L / 2.0 + t_R + pad_x

    V_mesh = VectorFunctionSpace(mesh, "Lagrange", 1)

    class OuterMeshBoundary(SubDomain):
        def inside(self, x, on_boundary):
            if near(x[0], 0.0) or near(x[0], L) or near(x[1], 0.0) or near(x[1], H):
                return True
            x_min = min(current_xc_top, current_xc_bot) - half_L
            x_max = max(current_xc_top, current_xc_bot) + half_L
            if x[0] < x_min or x[0] > x_max:
                return True

            # Freeze outside y-extent of both train zones
            # Below bottom train
            if x[1] < yc2 - t_H/2.0 - pad_y:
                return True
            # Above top train
            if x[1] > yc + t_H/2.0 + pad_y:
                return True

            return False

    outer_boundary = OuterMeshBoundary()
    bc_outer   = DirichletBC(V_mesh, Constant((0.0, 0.0)),   outer_boundary,  method="pointwise")
    bc_top     = DirichletBC(V_mesh, Constant(( vx*dt, 0.0)), dbc_objects_top)
    bc_bot     = DirichletBC(V_mesh, Constant((-vx*dt, 0.0)), dbc_objects_bot)

    u_mesh    = TrialFunction(V_mesh)
    v_mesh    = TestFunction(V_mesh)
    mesh_disp = Function(V_mesh)

    a_mesh = inner(grad(u_mesh), grad(v_mesh)) * dx
    L_mesh = dot(Constant((0.0, 0.0)), v_mesh) * dx

    solve(a_mesh == L_mesh, mesh_disp, [bc_outer, bc_top, bc_bot])

    w.assign(project(mesh_disp / dt, V, solver_type="cg", preconditioner_type="jacobi"))
    ALE.move(mesh, mesh_disp)

    return current_xc_top + vx*dt, current_xc_bot - vx*dt

def remesh(current_xc_top, current_xc_bot, u0_func, p0_func, u1_func, p1_func):
    global mesh, V, Q, u, p, v, q, au, Lu, ap, Lp, Force, bcu, bcp, ds, u0, p0, u1, p1, dx, w
    global dbc_objects_top, dbc_objects_bot

    mesh_Change = False
    min_q, max_q = MeshQuality.radius_ratio_min_max(mesh)

    if min_q < 0.2:
        mesh_Change = True
        print(f"Remeshing... min radius ratio: {min_q:.4f}")

        # Correct call — all four positional args, resolution is keyword
        mesh = build_mesh(current_xc_top, current_xc_bot)
        dx = Measure('dx', domain=mesh)
        V  = VectorFunctionSpace(mesh, "Lagrange", 1)
        Q  = FunctionSpace(mesh, "Lagrange", 1)

        u0 = Function(V); u0_func.set_allow_extrapolation(True); u0.interpolate(u0_func)
        p0 = Function(Q); p0_func.set_allow_extrapolation(True); p0.interpolate(p0_func)
        u1 = Function(V)
        p1 = Function(Q)
        w  = Function(V)

        
        class DirichletBoundaryTopTrain(SubDomain):
            def inside(self, x, on_boundary):
                return (on_boundary
                        and not near(x[0], 0.0) and not near(x[0], L)
                        and not near(x[1], 0.0) and not near(x[1], H)
                        and x[1] > yc - t_H/2.0 - 0.1)  # robust: above train body

        class DirichletBoundaryBotTrain(SubDomain):
            def inside(self, x, on_boundary):
                return (on_boundary
                        and not near(x[0], 0.0) and not near(x[0], L)
                        and not near(x[1], 0.0) and not near(x[1], H)
                        and x[1] < yc2 + t_H/2.0 + 0.1)  # robust: below train body
                
        dbc_objects_top = DirichletBoundaryTopTrain()
        dbc_objects_bot = DirichletBoundaryBotTrain()
        
        # Rebuild separate velocity BCs
        bcu = [
            DirichletBC(V.sub(0),  vx, dbc_objects_top),
            DirichletBC(V.sub(1), 0.0, dbc_objects_top),
            DirichletBC(V.sub(0), -vx, dbc_objects_bot),
            DirichletBC(V.sub(1), 0.0, dbc_objects_bot),
        ]
        bcp = [DirichletBC(Q, 0.0, DirichletBoundaryOuter())]

        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        dbc_left.mark(boundaries, 1);  dbc_right.mark(boundaries, 2)
        dbc_upper.mark(boundaries, 3); dbc_lower.mark(boundaries, 4)
        dbc_objects.mark(boundaries, 5)
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

        u = TrialFunction(V); p = TrialFunction(Q)
        v = TestFunction(V);  q = TestFunction(Q)

        h            = CellDiameter(mesh)
        u_mag_r      = sqrt(dot(u1, u1))
        d1_recalc    = 1.0 / sqrt(pow(1.0/dt, 2.0) + pow(u_mag_r/h, 2.0))
        d2_recalc    = h * u_mag_r
        um           = 0.5*(u + u0)
        um1          = 0.5*(u1 + u0)

        Fu = (inner((u-u0)/dt + grad(um)*(um1-w), v)*dx
              - p1*div(v)*dx
              + nu*inner(grad(um), grad(v))*dx
              + d1_recalc*inner((u-u0)/dt + grad(um)*(um1-w) + grad(p1), grad(v)*(um1-w))*dx
              + d2_recalc*div(um)*div(v)*dx)
        au = lhs(Fu); Lu = rhs(Fu)

        Fp = d1_recalc*inner((u1-u0)/dt + grad(um1)*(um1-w) + grad(p), grad(q))*dx + div(um1)*q*dx
        ap = lhs(Fp); Lp = rhs(Fp)

        psi = Function(V)
        DirichletBC(V.sub(0), Constant(phi_x), dbc_objects_top).apply(psi.vector())
        DirichletBC(V.sub(1), Constant(phi_y), dbc_objects_top).apply(psi.vector())

        Force = (inner((u1-u0)/dt + grad(um1)*(um1-w), psi)*dx
                 - p1*div(psi)*dx
                 + nu*inner(grad(um1), grad(psi))*dx)

        del Fu, Fp, um, um1, psi
        gc.collect()

    return mesh_Change


# Time stepping
T = 35 # for 1 full pass of the train
t = dt
last_mesh_change_time = 0
prev_calculated_force = 0.0
last_good_force = 0.0
ema_force = 0.0
ema_alpha = 0.10
t_remesh_start = 0.0
remesh_gap_steps = 40
gap_counter = 0
in_recovery = False
stability_streak = 0
required_stability_streak = 5
tolerance_for_stability = 0.05
recovery_buffer_force = []
recovery_buffer_time = []

current_xc_top = xc
current_xc_bot = L-xc

while t < T + DOLFIN_EPS:

    #s = 'Time t = ' + repr(t)
    #print(s)

    current_xc_top,current_xc_bot = move_mesh(mesh,current_xc_top,current_xc_bot)

    mesh_Change = remesh(current_xc_top,current_xc_bot, u0, p0, u1, p1)

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
        in_recovery = True
        last_mesh_change_time = t
        gap_counter = remesh_gap_steps
        t_remesh_start = t
        stability_streak = 0
        gap_counter = 0
        recovery_buffer_force = []
        recovery_buffer_time = []
        if len(force_array) > 0:
            last_good_force = force_array[-1]
        ema_force = last_good_force

    F = assemble(Force)
    calculated_force = normalization * F

    #Kill the zigzag oscillation (Two-step average)
    if t <= dt + DOLFIN_EPS:  # Catch the very first timestep safely
        prev_calculated_force = calculated_force
        
    stabilized_force = 0.5 * (calculated_force + prev_calculated_force)
    prev_calculated_force = calculated_force

    # Smoothing Logic
    if (t > start_sample_time):
        if in_recovery:
            # STATE: SHOCKWAVE RECOVERY
            # Buffer live values during recovery (don't write to main arrays yet)
            recovery_buffer_force.append(stabilized_force)
            recovery_buffer_time.append(t)

            gap_counter += 1
            
            ema_force = (1 - ema_alpha) * ema_force + ema_alpha * stabilized_force
            
            if abs(ema_force) > 1e-10:
                relative_error = abs((stabilized_force - ema_force) / ema_force)
            else:
                relative_error = 1.0
                
            if relative_error < tolerance_for_stability:
                stability_streak += 1
            else:
                stability_streak = 0

            if stability_streak >= required_stability_streak or gap_counter > 150:
                print(f"Solver stabilized after {gap_counter} steps. Backfilling Hermite curve...")
                
                n_steps = len(recovery_buffer_time)
                hermite_forces = []
                hermite_times = []

                for i in range(1, n_steps + 1):
                    progress = i / float(n_steps)
                    h_weight = 3*(progress**2) - 2*(progress**3)
                    artificial_force = (1.0 - h_weight) * last_good_force + h_weight * ema_force
                    hermite_forces.append(artificial_force)
                    hermite_times.append(recovery_buffer_time[i - 1])  # reuse same timestamps

                # Write Hermite curve in chronological order, then clear buffer
                force_array = np.append(force_array, hermite_forces)
                time       = np.append(time, hermite_times)

                # Clear the buffer and exit recovery
                recovery_buffer_force = []
                recovery_buffer_time = []
                in_recovery = False
                
        else:
            # STATE: NORMAL OPERATION
            force_array = np.append(force_array, stabilized_force)
            time = np.append(time, t)
        
    if t > plot_time or mesh_Change and not in_recovery:
        s = 'Time t = ' + repr(t)
        print(s)

        # Save solution to file
        # file_u << u1
        # file_p << p1
        
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Time t = {t:.2f} & Mesh Change: {mesh_Change}, last mesh change at t = {last_mesh_change_time:.2f}", fontsize=16)

        # Plot solution
        plt.subplot(2, 2, 1)
        # plt.figure()
        plot(u1, title=f"Velocity")

        # plt.figure()
        plt.subplot(2, 2, 2)
        plot(p1, title=f"Pressure")

        # plt.figure()
        plt.subplot(2, 2, 3)
        plot(mesh, title=f"Mesh")

        plot_time += T/plot_freq

        # plt.figure()
        plt.subplot(2, 2, 4)
        plt.title(f"Force")
        plt.plot(time, force_array)

    # Update time step
    u0.assign(u1)
    t += dt

np.set_printoptions(threshold=np.inf)
force_array = np.append(force_array, normalization*F)
time = np.append(time, t)
with open("force.txt", "w") as f:
#   f.write(str(force_array) + "\n" + str(time))
    f.write(str(np.array([force_array,time]).T))

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