import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

# Define the exact solution for comparison
def exact_solution(x):
    return np.exp(-10*x)

# Define the differential equation y'' = 100y
def f(x, y):
    # y[0] is y, y[1] is y'
    return np.array([y[1], 100*y[0]])

# Linear Shooting Algorithm implementation
def linear_shooting(h):
    # Define the interval
    a, b = 0, 1
    # Define boundary conditions
    alpha, beta = 1, np.exp(-10)
    # Calculate number of steps
    n = int((b - a) / h)
    # Create the mesh
    x = np.linspace(a, b, n+1)

    # Solve the first initial value problem: y'' = 100y, y(0) = 1, y'(0) = 0
    def ivp1(t, y):
        return [y[1], 100*y[0]]

    sol1 = solve_ivp(ivp1, [a, b], [1, 0], method='RK45', t_eval=x)
    y1 = sol1.y[0]

    # Solve the second initial value problem: y'' = 100y, y(0) = 0, y'(0) = 1
    def ivp2(t, y):
        return [y[1], 100*y[0]]

    sol2 = solve_ivp(ivp2, [a, b], [0, 1], method='RK45', t_eval=x)
    y2 = sol2.y[0]

    # Calculate the linear combination that satisfies the boundary conditions
    # We need to find c such that: y1(b) + c*y2(b) = beta
    c = (beta - y1[-1]) / y2[-1]

    # Calculate the final approximation
    y_approx = y1 + c * y2

    # Calculate the error compared to the exact solution
    y_exact = exact_solution(x)
    error = np.max(np.abs(y_approx - y_exact))

    return x, y_approx, y_exact, error

# Solve using the Linear Shooting Algorithm with h = 0.1
x_h1, y_approx_h1, y_exact_h1, error_h1 = linear_shooting(0.1)
print(f"Maximum error with h = 0.1: {error_h1:.8e}")

# Solve using the Linear Shooting Algorithm with h = 0.05
x_h2, y_approx_h2, y_exact_h2, error_h2 = linear_shooting(0.05)
print(f"Maximum error with h = 0.05: {error_h2:.8e}")

# Solve using solve_bvp
def bvp_func(x, y):
    # y[0] is the function, y[1] is its derivative
    return np.vstack((y[1], 100*y[0]))

def bc(ya, yb):
    return np.array([ya[0] - 1, yb[0] - np.exp(-10)])

# Create initial mesh and guess
x_bvp = np.linspace(0, 1, 20)
y_guess = np.zeros((2, x_bvp.size))
y_guess[0] = np.exp(-10*x_bvp)  # Initial guess based on the known solution
y_guess[1] = -10*np.exp(-10*x_bvp)  # Derivative of the initial guess

# Solve the BVP
sol_bvp = solve_bvp(bvp_func, bc, x_bvp, y_guess)

# Calculate error for solve_bvp
x_fine = np.linspace(0, 1, 100)
y_bvp = sol_bvp.sol(x_fine)[0]
y_exact_fine = exact_solution(x_fine)
error_bvp = np.max(np.abs(y_bvp - y_exact_fine))
print(f"Maximum error with solve_bvp: {error_bvp:.8e}")

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x_h1, y_approx_h1, 'bo-', label='Linear Shooting (h=0.1)')
plt.plot(x_h1, y_exact_h1, 'r-', label='Exact Solution')
plt.title('Solution with h = 0.1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x_h2, y_approx_h2, 'go-', label='Linear Shooting (h=0.05)')
plt.plot(x_h2, y_exact_h2, 'r-', label='Exact Solution')
plt.title('Solution with h = 0.05')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x_fine, y_bvp, 'mo-', label='solve_bvp')
plt.plot(x_fine, y_exact_fine, 'r-', label='Exact Solution')
plt.title('Solution with solve_bvp')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.semilogy(x_h1, np.abs(y_approx_h1 - y_exact_h1), 'b-', label='Error (h=0.1)')
plt.semilogy(x_h2, np.abs(y_approx_h2 - y_exact_h2), 'g-', label='Error (h=0.05)')
plt.semilogy(x_fine, np.abs(y_bvp - y_exact_fine), 'm-', label='Error (solve_bvp)')
plt.title('Error Comparison (log scale)')
plt.xlabel('x')
plt.ylabel('|Error|')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print summary
print("\nSummary of Results:")
print("-" * 50)
print(f"Method              | Maximum Error")
print("-" * 50)
print(f"Linear Shooting (h=0.1)  | {error_h1:.8e}")
print(f"Linear Shooting (h=0.05) | {error_h2:.8e}")
print(f"solve_bvp               | {error_bvp:.8e}")
print("-" * 50)