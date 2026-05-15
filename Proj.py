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

# Set to true for train crossing scenario
# Set to false for lone train scenario
two_trains = True

# Define rectangular domain
L = 8
H = 3

# Train dimensions
xc = 2.0
yc = 0.6*H
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

# Generate mesh 
resolution = 64

# Build mesh method,using x-position of train(s) as input, to allow for remeshing with updated train position
def build_mesh(xc_top, xc_bot):
    global yc, yc2, t_L, t_H, t_R, L, H, resolution, two_trains

    # Build trains as rectangles with half-circles at the front
    p1 = Point(xc_top - t_L/2.0, yc - t_H/2.0)
    p2 = Point(xc_top + t_L/2.0, yc + t_H/2.0)
    body = Rectangle(p1, p2)
    head_center = Point(xc_top + t_L/2.0, yc)
    head = Circle(head_center, t_R)
    train = body + head

    if two_trains:
        p1 = Point(xc_bot - t_L/2.0, yc2 - t_H/2.0)
        p2 = Point(xc_bot + t_L/2.0, yc2 + t_H/2.0)
        body2 = Rectangle(p1, p2)
        head_center2 = Point(xc_bot - t_L/2.0, yc2)
        head2 = Circle(head_center2, t_R)
        train2 = body2 + head2
        domain = Rectangle(Point(0.0, 0.0), Point(L, H)) - train - train2
    else:
        domain = Rectangle(Point(0.0, 0.0), Point(L, H)) - train

    mesh = generate_mesh(domain, resolution)


    # Local mesh refinement (specified by a cell marker)
    no_levels = 0
    buffer = 0.3
    for i in range(0,no_levels):
        cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_marker.set_all(False)
        for cell in cells(mesh):
            p = cell.midpoint()
            px,py = p[0],p[1]
            #   if p.distance(Point(xc, yc)) < 0.5:
            if    ((xc_top - t_L/2.0 - buffer*2) < px < (xc_top + t_L/2.0 + t_R + buffer) and 
                    (yc - t_H/2.0 - buffer) < py < (yc + t_H/2.0 + buffer)
                    or 
                    (xc_bot - t_L/2.0 - buffer*2) < px < (xc_bot + t_L/2.0 + t_R + buffer) and
                    (yc2 - t_H/2.0 - buffer) < py < (yc2 + t_H/2.0 + buffer)):
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

# Middle of both trains used to separate top and bottom train boundary conditions, as well as to determine mesh deformation zone
y_mid = 0.5 * (yc + yc2)

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
bcu = [bcu_top0, bcu_top1]
if two_trains:
    bcu += [bcu_bot0, bcu_bot1]

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
w = Function(V) # for global mesh deformation

# Set parameters for nonlinear and lienar solvers
num_nnlin_iter = 5
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Time step length
dt = 0.25*mesh.hmin()


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


# --- Drag (phi_x=1) ---
psi_drag = Function(V)
DirichletBC(V.sub(0), Constant(1.0), dbc_objects_top).apply(psi_drag.vector())
DirichletBC(V.sub(1), Constant(0.0), dbc_objects_top).apply(psi_drag.vector())
Force_drag = (inner((u1-u0)/dt + grad(um1)*(um1-w), psi_drag)*dx
              - p1*div(psi_drag)*dx
              + nu*inner(grad(um1), grad(psi_drag))*dx)

# --- Lift (phi_y=1) ---
psi_lift = Function(V)
DirichletBC(V.sub(0), Constant(0.0), dbc_objects_top).apply(psi_lift.vector())
DirichletBC(V.sub(1), Constant(1.0), dbc_objects_top).apply(psi_lift.vector())
Force_lift = (inner((u1-u0)/dt + grad(um1)*(um1-w), psi_lift)*dx
              - p1*div(psi_lift)*dx
              + nu*inner(grad(um1), grad(psi_lift))*dx)


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
drag_array = np.delete(np.array(0.0), 0)
lift_array = np.delete(np.array(0.0), 0)
time = np.delete(np.array(0.0), 0)
start_sample_time = 1.0

# Move the mesh according to the train's velocity, using an ALE formulation
# Mesh deformation is computed by solving a Laplace equation, with Dirichlet BCs set to the train velocity at the train surface and 0 at the outer walls, and then extrapolated to the whole mesh. The mesh velocity w is then used in the ALE formulation of the Navier-Stokes equations, and the mesh is moved using ALE.move().
# The train's x-position is also updated and returned, to allow for remeshing with the train in the new position if needed. The mesh deformation zone is limited to a buffer around the train, to avoid excessive deformation of the whole mesh.
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
    bcs_mesh = [bc_outer, bc_top]
    if two_trains:
        bcs_mesh += [bc_bot]

    u_mesh    = TrialFunction(V_mesh)
    v_mesh    = TestFunction(V_mesh)
    mesh_disp = Function(V_mesh)

    a_mesh = inner(grad(u_mesh), grad(v_mesh)) * dx
    L_mesh = dot(Constant((0.0, 0.0)), v_mesh) * dx

    solve(a_mesh == L_mesh, mesh_disp, bcs_mesh)

    w.assign(project(mesh_disp / dt, V, solver_type="cg", preconditioner_type="jacobi"))
    ALE.move(mesh, mesh_disp)

    return current_xc_top + vx*dt, current_xc_bot - vx*dt

# Remeshing function, called at each time step after mesh movement, to check mesh quality and rebuild the mesh if quality is too low. 
# The current train positions are passed as arguments to allow for building the new mesh with the trains in the correct position.
# The function also rebuilds all necessary functions, boundary conditions, and variational forms that depend on the mesh, and returns a boolean indicating whether a remesh was performed or not.
# The remeshing criterion is based on the minimum radius ratio of the mesh cells. If the minimum radius ratio falls below a threshold (0.2 in this case), the mesh is rebuilt.
def remesh(current_xc_top, current_xc_bot, u0_func, p0_func, u1_func, p1_func):
    global mesh, V, Q, u, p, v, q, au, Lu, ap, Lp, Force_drag, Force_lift, drag_array, lift_array, time, bcu, bcp, ds, u0, p0, u1, p1, dx, w
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
            
        ]
        if two_trains:
            bcu += [
                DirichletBC(V.sub(0), -vx, dbc_objects_bot),
                DirichletBC(V.sub(1), 0.0, dbc_objects_bot)
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

        psi_drag = Function(V)
        DirichletBC(V.sub(0), Constant(1.0), dbc_objects_top).apply(psi_drag.vector())
        DirichletBC(V.sub(1), Constant(0.0), dbc_objects_top).apply(psi_drag.vector())
        Force_drag = (inner((u1-u0)/dt + grad(um1)*(um1-w), psi_drag)*dx
                    - p1*div(psi_drag)*dx
                    + nu*inner(grad(um1), grad(psi_drag))*dx)

        # --- Lift (phi_y=1) ---
        psi_lift = Function(V)
        DirichletBC(V.sub(0), Constant(0.0), dbc_objects_top).apply(psi_lift.vector())
        DirichletBC(V.sub(1), Constant(1.0), dbc_objects_top).apply(psi_lift.vector())
        Force_lift = (inner((u1-u0)/dt + grad(um1)*(um1-w), psi_lift)*dx
                    - p1*div(psi_lift)*dx
                    + nu*inner(grad(um1), grad(psi_lift))*dx)

        del Fu, Fp, um, um1
        gc.collect()

    return mesh_Change


# Time stepping
# T = 40 # for 1 full pass of the train
T = 40
t = dt
last_mesh_change_time = 0
last_good_drag_force = 0.0
last_good_lift_force = 0.0
remesh_gap_steps = 60
gap_counter = 0
in_recovery = False

# Buffers to hold force values during remeshing recovery, to allow for smoothing before writing to main arrays
recovery_buffer_dragforce = []
recovery_buffer_liftforce = []
recovery_buffer_time = []
skip_force_recording = False

# Initial train positions for mesh movement and remeshing
current_xc_top = xc
current_xc_bot = L-xc

while t < T + DOLFIN_EPS:
    # Move the mesh according to the train's velocity, and get updated train positions
    current_xc_top,current_xc_bot = move_mesh(mesh,current_xc_top,current_xc_bot)
    # Check mesh quality and remesh if needed, then rebuild all necessary functions, boundary conditions, and variational forms that depend on the mesh. 
    # Get boolean indicating whether a remesh was performed or not.
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

    # Compute forces and write to arrays, with smoothing logic to handle remeshing shocks
    if mesh_Change:
        in_recovery = True
        last_mesh_change_time = t
        gap_counter = 0
        # Don't write the first post-remesh force values directly to the main arrays, as they will be inaccurate due to the shockwave. 
        # Instead, buffer them and apply smoothing before writing to the main arrays after a few steps.
        recovery_buffer_dragforce = []
        recovery_buffer_liftforce = []
        recovery_buffer_time = []
        # If we have good force values from the previous steps, we can use them as the starting point for the smoothing curve.
        if len(drag_array) > 0:
            last_good_drag_force = drag_array[-1]
            last_good_lift_force = lift_array[-1]
            skip_force_recording = True
    else:
        skip_force_recording = False

    # Compute forces for current time step
    F_drag = assemble(Force_drag)
    F_lift = assemble(Force_lift)
    calc_drag = normalization * F_drag
    calc_lift = normalization * F_lift

    # Only record forces after a certain start time,
    # and handle smoothing of force values during remeshing recovery to avoid shockwave artifacts in the recorded data.
    if (t > start_sample_time) and not skip_force_recording:
        # If we are in the middle of a remeshing recovery, 
        # buffer the force values and apply smoothing before writing to the main arrays, to avoid shockwave artifacts.
        if in_recovery:
            recovery_buffer_dragforce.append(calc_drag)
            recovery_buffer_liftforce.append(calc_lift)
            recovery_buffer_time.append(t)

            gap_counter += 1
            
            # 
            if gap_counter >= remesh_gap_steps:
                tail_drag = recovery_buffer_dragforce[-15:]
                target_force_drag = sorted(tail_drag)[len(tail_drag)//2]  
                tail_lift = recovery_buffer_liftforce[-15:]
                target_force_lift = sorted(tail_lift)[len(tail_lift)//2]        
                n_steps = len(recovery_buffer_time)
                hermite_forces_drag = []
                hermite_forces_lift = []
                hermite_times = []

                for i in range(1, n_steps + 1):
                    progress = i / float(n_steps)
                    h_weight = 3*(progress**2) - 2*(progress**3)
                    artificial_force_drag = (1.0 - h_weight) * last_good_drag_force + h_weight *target_force_drag
                    artificial_force_lift = (1.0 - h_weight) * last_good_lift_force + h_weight *target_force_lift
                    hermite_forces_drag.append(artificial_force_drag)
                    hermite_forces_lift.append(artificial_force_lift)
                    hermite_times.append(recovery_buffer_time[i - 1])  # reuse same timestamps

                # Write Hermite curve in chronological order, then clear buffer
                drag_array = np.append(drag_array, hermite_forces_drag)
                lift_array = np.append(lift_array, hermite_forces_lift)
                time       = np.append(time, hermite_times)

                # Clear the buffer and exit recovery
                recovery_buffer_dragforce = []
                recovery_buffer_liftforce = []
                recovery_buffer_time = []
                in_recovery = False
                
        else:
            drag_array = np.append(drag_array, calc_drag)
            lift_array = np.append(lift_array, calc_lift)
            time = np.append(time, t)

        
    if (t > plot_time and not in_recovery) or (mesh_Change):
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
        plt.title(f"Force: (t = {t:.2f})")
        plt.plot(time, drag_array, color='tab:blue',  label='Drag')
        plt.plot(time, lift_array, color='tab:orange', label='Lift')
        plt.legend()

    # Update time step
    u0.assign(u1)
    t += dt

# np.set_printoptions(threshold=np.inf)
# force_array = np.append(force_array, normalization*F)
# time = np.append(time, t)
# with open("force.txt", "w") as f:
# #   f.write(str(force_array) + "\n" + str(time))
#     f.write(str(np.array([force_array,time]).T))

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
plt.plot(time, drag_array, color='tab:blue',  label='Drag')
plt.plot(time, lift_array, color='tab:orange', label='Lift')
plt.legend()

plt.show()
#!tar -czvf results-NS.tar.gz results-NS
#files.download('results-NS.tar.gz')