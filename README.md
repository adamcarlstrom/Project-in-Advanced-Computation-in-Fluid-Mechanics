# Numerical Simulation of Train Aerodynamics

## Project Overview
This project simulates the highly transient aerodynamic forces generated during a high-speed train crossing scenario. Using a custom **Localized Arbitrary Lagrangian-Eulerian (ALE)** framework built in **FEniCS**, the simulation models the pressure fields and extreme lateral forces (suction and repulsion) that occur when two trains pass each other in opposite directions.

The primary objective is to bypass standard Eulerian/Lagrangian mesh limitations (such as element inversion or the "snowplow" effect) by implementing localized mesh kinematics, periodic topological remeshing, and dynamic force smoothing.

---

## Technical Stack
* **Language:** Python 3
* **CFD Framework:** FEniCS / dolfin
* **Mesh Generation:** mshr
* **Data Visualization:** Matplotlib

---

## Mathematical Framework

### 1. Incompressible Navier-Stokes Equations
Because the simulated train velocities remain well below the Mach 0.3 compressibility threshold, the fluid is modeled using the incompressible Navier-Stokes equations:
* **Momentum:** $\dot{u} + (u \cdot \nabla)u + \nabla p - \Delta u = 0$
* **Continuity:** $\nabla \cdot u = 0$

### 2. Arbitrary Lagrangian-Eulerian (ALE) Formulation
To account for the trains moving through the mesh without failing the solver, the fluid equations are mapped to an ALE framework. The mesh velocity ($w$) is incorporated into the convective term, allowing the finite elements to stretch and deform safely alongside the boundary displacement.

### 3. Finite Element Stabilization
*Assumption Note: Based on the residual variables (`d1`, `d2`) defined in the project code, Streamline Upwind Petrov-Galerkin (SUPG) and Pressure-Stabilizing Petrov-Galerkin (PSPG) stabilization are assumed to be implemented to bypass the LBB condition for equal-order* $P_1-P_1$ *elements and handle convection-dominated flow.*

---

## Methodology & Implementation

### Localized Mesh Kinematics
Instead of global mesh deformation—which causes severe aspect ratio degradation—mesh stretching is isolated to a localized "bubble" around the passing trains. The mesh displacement field is governed by solving a **Laplace equation** ($\nabla \cdot (\gamma \nabla u_{mesh}) = 0$), which acts as a numerical shock-absorber distributing the strain evenly.

### Periodic Topological Remeshing
To completely prevent element inversion as the gap between the trains closes, a geometric element quality threshold is enforced. 
* The solver continuously calculates the minimum radius ratio: $q = 2 \cdot r_{in} / r_{out}$.
* When $q < 0.15$, the solver pauses time integration.
* A pristine, unstretched mesh is generated, and the velocity/pressure fields are mapped onto it using **point-wise interpolation**.

### Post-Remesh Force Smoothing
Interpolation across non-matching meshes inherently introduces transient numerical divergence and pressure spikes. To mitigate this, a dynamic shockwave recovery pipeline is implemented:
* The solver enters a 60-iteration recovery window post-remesh.
* Missing/corrupted force data is bridged using a **Cubic Hermite blending function**: $h(p) = 3p^2 - 2p^3$.
* This guarantees a continuous, physically realistic force history while filtering out mathematical artifacts.

---

## Results & Key Findings

The simulation tracks the crossing interaction across three distinct aerodynamic phases:

1. **Approach Phase (Repulsion):** As the high-pressure noses of the trains meet, the air is violently compressed. The pressure between the trains spikes, causing the vehicles to repel each other. 
   * *Peak Repulsion Coefficient:* $C_L = +4.25$
2. **Parallel Phase (Severe Suction):** As the train bodies align side-by-side, the narrow gap acts as a Venturi channel. The localized fluid velocity skyrockets, plunging the static pressure to an absolute minimum and generating a massive inward pulling force.
   * *Peak Suction Coefficient:* $C_L = -6.0$
3. **Separation Phase (Stabilization):** As the noses pass the opposing tails, the localized pressure fields normalize, resembling the aerodynamic signature of a lone train.

**Real-World Scaling Impact:**
When scaled to a real-world scenario (25m trains, $100\text{ m/s}$ relative velocity), the lateral load transitions from $+78\text{ kN}$ of repulsion to $-110\text{ kN}$ of suction. This represents a net force swing of **$188\text{ kN}$ within just $0.23\text{ seconds}$**, underscoring the extreme fatigue accumulation risks on modern rail infrastructure.

---

## Conclusions & Limitations
The Localized ALE framework with periodic remeshing successfully tracks the extreme transient loads of passing vehicles without succumbing to element crushing. 

However, because this is a **2D simulation**, it inherently forces all displaced fluid strictly through the lateral gap between the trains. In a 3D reality, fluid can escape vertically over the roof and under the bogies. Consequently, the 2D constraint mathematically exaggerates the velocity gradients and over-predicts the resulting suction forces, serving as a worst-case upper bound.

---

## Future Work
* **3D Domain Extension:** Expanding the solver to 3D to allow vertical flow relief, generating highly accurate, comparable physical coefficients.
* **Overset (Chimera) Grids:** Migrating from periodic remeshing to true overlapping overset grids to allow continuous sliding interfaces, eliminating interpolation-induced pressure spikes entirely.
* **Triple Decomposition:** Analyzing shear, strain, and rotation fields to isolate and verify the presence of the trailing "tail wave."

---
*Created by Adam Carlström for DD2365 - Advanced Computation in Fluid Mechanics.*
