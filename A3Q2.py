import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Define the Lotka-Volterra system
def lotka_volterra(t, z):
    x, y = z
    dx_dt = -0.1 * x + 0.02 * x * y
    dy_dt = 0.2 * y - 0.025 * x * y
    return [dx_dt, dy_dt]

# Initial conditions
x0 = 6  # Initial predator population (thousands)
y0 = 6  # Initial prey population (thousands)
initial_conditions = [x0, y0]

# Time span for solution
t_span = (0, 100)  # Solve from t=0 to t=100
t_eval = np.linspace(0, 100, 1000)  # Points to evaluate the solution at

# Solve the system
solution = solve_ivp(
    lotka_volterra,
    t_span,
    initial_conditions,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-9
)

# Extract results
t = solution.t
x = solution.y[0]  # Predator population
y = solution.y[1]  # Prey population

# Find the first time when populations are equal
# First, find where the difference changes sign
diff = x - y
sign_changes = np.where(np.diff(np.signbit(diff)))[0]

# If there are sign changes, get the first one
if len(sign_changes) > 0:
    idx = sign_changes[0]
    # Estimate the exact time using linear interpolation
    t1, t2 = t[idx], t[idx + 1]
    x1, x2 = x[idx], x[idx + 1]
    y1, y2 = y[idx], y[idx + 1]

    # Linear interpolation to find exact crossing time
    t_equal = t1 + (t2 - t1) * (x1 - y1) / ((x1 - y1) - (x2 - y2))
    population_equal = x1 + (x2 - x1) * (t_equal - t1) / (t2 - t1)

    print(f"The populations are first equal at t ≈ {t_equal:.4f}")
    print(f"Equal population value ≈ {population_equal:.4f} thousands")
else:
    print("The populations do not become equal within the simulation time.")

# Create plots
plt.figure(figsize=(12, 10))

# Plot both populations vs time
plt.subplot(2, 2, 1)
plt.plot(t, x, 'r-', label='Predators (x)')
plt.plot(t, y, 'b-', label='Prey (y)')
if len(sign_changes) > 0:
    plt.axvline(x=t_equal, color='g', linestyle='--', label=f'Equal at t={t_equal:.4f}')
    plt.plot(t_equal, population_equal, 'go', markersize=8)
plt.xlabel('Time')
plt.ylabel('Population (thousands)')
plt.title('Predator and Prey Populations vs Time')
plt.grid(True)
plt.legend()

# Plot phase space
plt.subplot(2, 2, 2)
plt.plot(x, y, 'k-')
plt.plot(x0, y0, 'ko', markersize=8, label='Initial state')
if len(sign_changes) > 0:
    plt.plot(population_equal, population_equal, 'go', markersize=8, label=f'Equal at t={t_equal:.4f}')
plt.xlabel('Predators (x)')
plt.ylabel('Prey (y)')
plt.title('Phase Space Portrait')
plt.grid(True)
plt.legend()

# Plot the difference between populations
plt.subplot(2, 2, 3)
plt.plot(t, diff, 'k-')
plt.axhline(y=0, color='g', linestyle='--')
if len(sign_changes) > 0:
    plt.plot(t_equal, 0, 'go', markersize=8)
plt.xlabel('Time')
plt.ylabel('x - y')
plt.title('Difference Between Populations')
plt.grid(True)

# Plot a zoomed-in view around the first crossing point
if len(sign_changes) > 0:
    plt.subplot(2, 2, 4)
    zoom_range = 5  # Show 5 time units before and after crossing
    zoom_start = max(0, t_equal - zoom_range)
    zoom_end = min(t[-1], t_equal + zoom_range)

    # Find indices for zoomed view
    zoom_indices = (t >= zoom_start) & (t <= zoom_end)

    plt.plot(t[zoom_indices], x[zoom_indices], 'r-', label='Predators (x)')
    plt.plot(t[zoom_indices], y[zoom_indices], 'b-', label='Prey (y)')
    plt.axvline(x=t_equal, color='g', linestyle='--')
    plt.plot(t_equal, population_equal, 'go', markersize=8)
    plt.xlabel('Time')
    plt.ylabel('Population (thousands)')
    plt.title('Zoomed View Around Equal Point')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# Calculate and plot the first few cycles of the system
plt.figure(figsize=(12, 5))

# Find local maxima of x to identify cycles
from scipy.signal import find_peaks
peaks, _ = find_peaks(x, height=0)

if len(peaks) >= 3:
    cycle_end_idx = peaks[2]  # Use the third peak to show two full cycles

    plt.subplot(1, 2, 1)
    plt.plot(t[:cycle_end_idx], x[:cycle_end_idx], 'r-', label='Predators (x)')
    plt.plot(t[:cycle_end_idx], y[:cycle_end_idx], 'b-', label='Prey (y)')
    if len(sign_changes) > 0 and t_equal <= t[cycle_end_idx]:
        plt.axvline(x=t_equal, color='g', linestyle='--', label=f'Equal at t={t_equal:.4f}')
    plt.xlabel('Time')
    plt.ylabel('Population (thousands)')
    plt.title('First Two Cycles of the System')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x[:cycle_end_idx], y[:cycle_end_idx], 'k-')
    plt.plot(x0, y0, 'ko', markersize=8, label='Initial state')
    if len(sign_changes) > 0:
        plt.plot(population_equal, population_equal, 'go', markersize=8, label=f'Equal population')
    plt.xlabel('Predators (x)')
    plt.ylabel('Prey (y)')
    plt.title('Phase Space (First Two Cycles)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()