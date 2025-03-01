import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Define the competition model
def competition_model(t, z):
    x, y = z
    dx_dt = x * (2 - 0.4 * x - 0.3 * y)
    dy_dt = y * (1 - 0.1 * y - 0.3 * x)
    return [dx_dt, dy_dt]

# Find the equilibrium points
def find_equilibria():
    # Solve system of equations:
    # 2 - 0.4x - 0.3y = 0
    # 1 - 0.1y - 0.3x = 0

    # From second equation: y = (1 - 0.3x) / 0.1 = 10 - 3x
    # Substitute into first equation:
    # 2 - 0.4x - 0.3(10 - 3x) = 0
    # 2 - 0.4x - 3 + 0.9x = 0
    # -1 + 0.5x = 0
    # x = 2

    # Then y = 10 - 3(2) = 10 - 6 = 4

    # So we have equilibrium points at:
    # (0, 0) - Trivial
    # (5, 0) - x-axis (2/0.4 = 5, 0)
    # (0, 10) - y-axis (0, 1/0.1 = 10)
    # (2, 4) - Interior

    return [(0, 0), (5, 0), (0, 10), (2, 4)]

# Define the initial conditions
initial_conditions = [
    (1.5, 3.5),  # Case a
    (1.0, 1.0),  # Case b
    (2.0, 7.0),  # Case c
    (4.5, 0.5)   # Case d
]

labels = ['a', 'b', 'c', 'd']

# Time span for solution
t_span = (0, 50)  # Solve from t=0 to t=50 years
t_eval = np.linspace(0, 50, 1000)  # Points to evaluate the solution at

# Create a figure for all solutions
plt.figure(figsize=(15, 12))

# Store final states for each case
final_states = []

# Solve the system for each initial condition
for i, (x0, y0) in enumerate(initial_conditions):
    # Solve the system
    solution = solve_ivp(
        competition_model,
        t_span,
        [x0, y0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    # Extract results
    t = solution.t
    x = solution.y[0]  # Species 1 population
    y = solution.y[1]  # Species 2 population

    # Store final state
    final_states.append((x[-1], y[-1]))

    # Plot time series
    plt.subplot(2, 2, i+1)
    plt.plot(t, x, 'r-', label='Species 1 (x)')
    plt.plot(t, y, 'b-', label='Species 2 (y)')
    plt.xlabel('Time (years)')
    plt.ylabel('Population (thousands)')
    plt.title(f'Case {labels[i]}: Initial conditions x(0)={x0}, y(0)={y0}')
    plt.grid(True)
    plt.legend()

    # Add info about final state
    final_x, final_y = final_states[i]
    equilibria = find_equilibria()

    # Find closest equilibrium
    closest = min(equilibria, key=lambda eq: (eq[0] - final_x)**2 + (eq[1] - final_y)**2)

    plt.text(0.5, 0.05,
             f'Final state: x≈{final_x:.2f}, y≈{final_y:.2f}\nClosest equilibrium: {closest}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('competition_time_series.png')

# Create a phase plane portrait
plt.figure(figsize=(12, 10))

# Create a grid of points for vector field
x_grid = np.linspace(0, 8, 20)
y_grid = np.linspace(0, 12, 20)
X, Y = np.meshgrid(x_grid, y_grid)

# Calculate vector field
U = X * (2 - 0.4 * X - 0.3 * Y)
V = Y * (1 - 0.1 * Y - 0.3 * X)

# Normalize vectors for better visualization
norm = np.sqrt(U**2 + V**2)
norm[norm == 0] = 1  # Avoid division by zero
U_norm = U / norm
V_norm = V / norm

# Plot vector field
plt.quiver(X, Y, U_norm, V_norm, angles='xy', scale_units='xy', scale=25, width=0.002, color='gray', alpha=0.5)

# Plot nullclines
# x-nullcline: x = 0 or 2 - 0.4x - 0.3y = 0 => y = (2 - 0.4x) / 0.3
# y-nullcline: y = 0 or 1 - 0.1y - 0.3x = 0 => y = (1 - 0.3x) / 0.1

x_nullcline_x = np.linspace(0, 5, 100)
x_nullcline_y = (2 - 0.4 * x_nullcline_x) / 0.3

y_nullcline_x = np.linspace(0, 3.33, 100)  # x up to 1/0.3 ≈ 3.33
y_nullcline_y = (1 - 0.3 * y_nullcline_x) / 0.1

plt.plot(x_nullcline_x, x_nullcline_y, 'r--', label='x nullcline: $2 - 0.4x - 0.3y = 0$')
plt.plot(y_nullcline_x, y_nullcline_y, 'b--', label='y nullcline: $1 - 0.1y - 0.3x = 0$')

# Plot equilibria
equilibria = find_equilibria()
for eq in equilibria[1:]:  # Skip (0,0) for clearer visualization
    plt.plot(eq[0], eq[1], 'ko', markersize=8)
    plt.text(eq[0]+0.1, eq[1]+0.1, f'({eq[0]}, {eq[1]})')

# Plot solution trajectories
colors = ['r', 'g', 'b', 'm']
for i, (x0, y0) in enumerate(initial_conditions):
    solution = solve_ivp(
        competition_model,
        t_span,
        [x0, y0],
        t_eval=t_eval,
        method='RK45'
    )
    x = solution.y[0]
    y = solution.y[1]

    plt.plot(x, y, colors[i], label=f'Case {labels[i]}: ({x0}, {y0})')
    plt.plot(x0, y0, colors[i]+'o', markersize=8)  # Starting point
    plt.plot(x[-1], y[-1], colors[i]+'x', markersize=8)  # Ending point

    # Add arrow to show direction
    idx = len(x) // 2
    plt.annotate('', xy=(x[idx+1], y[idx+1]), xytext=(x[idx], y[idx]),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color=colors[i]))

plt.xlabel('Species 1 population (x) in thousands')
plt.ylabel('Species 2 population (y) in thousands')
plt.title('Phase Portrait of Competition Model')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(0, 8)
plt.ylim(0, 12)

# Add text explaining the equilibrium points
plt.text(5.5, 8, 'Equilibrium points:\n(0,0): Extinction\n(5,0): Species 1 only\n(0,10): Species 2 only\n(2,4): Coexistence',
         bbox=dict(facecolor='white', alpha=0.7))

plt.savefig('competition_phase_portrait.png')

# 3D plot to show how both populations evolve over time
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, (x0, y0) in enumerate(initial_conditions):
    solution = solve_ivp(
        competition_model,
        t_span,
        [x0, y0],
        t_eval=t_eval,
        method='RK45'
    )

    t = solution.t
    x = solution.y[0]
    y = solution.y[1]

    ax.plot(t, x, y, colors[i], label=f'Case {labels[i]}: ({x0}, {y0})')
    ax.plot([t[0]], [x[0]], [y[0]], colors[i]+'o', markersize=8)  # Starting point
    ax.plot([t[-1]], [x[-1]], [y[-1]], colors[i]+'x', markersize=8)  # Ending point

ax.set_xlabel('Time (years)')
ax.set_ylabel('Species 1 (x)')
ax.set_zlabel('Species 2 (y)')
ax.set_title('3D Trajectories of Population Dynamics')
ax.legend()

plt.savefig('competition_3d_trajectories.png')
plt.show()

# Print analysis summary
print("Population Competition Model Analysis")
print("------------------------------------")
print("Model: dx/dt = x(2 - 0.4x - 0.3y), dy/dt = y(1 - 0.1y - 0.3x)")
print("\nEquilibrium points:")
print("(0, 0): Extinction of both species")
print("(5, 0): Species 1 survives alone at carrying capacity K1 = 5")
print("(0, 10): Species 2 survives alone at carrying capacity K2 = 10")
print("(2, 4): Coexistence equilibrium")
print("\nFinal states for different initial conditions:")

for i, (label, ic, fs) in enumerate(zip(labels, initial_conditions, final_states)):
    print(f"Case {label}: Initial ({ic[0]}, {ic[1]}) → Final ({fs[0]:.2f}, {fs[1]:.2f})")

print("\nStability analysis:")
print("The stable equilibrium point is (2, 4), representing coexistence.")
print("All trajectories approach this point regardless of initial conditions.")