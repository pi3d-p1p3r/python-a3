import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system_of_odes(t, y):
    """
    Define the system of first-order ODEs.
    y[0] = x_1
    y[1] = x_1'
    y[2] = x_1''
    y[3] = x_2
    y[4] = x_2'
    y[5] = x_2''
    """
    # From the equations:
    # x_1'' = -2*x_2^2 + x_2
    # x_2''' = -x_1^3 + x_2^2 + x_1 + sin(t)

    # Define the derivatives for our system
    dydt = [
        y[1],                                  # x_1' = y_1' = y_2
        y[2],                                  # x_1'' = y_2' = y_3
        -2 * y[3]**2 + y[3],                   # x_1''' = y_3' = -2*x_2^2 + x_2
        y[4],                                  # x_2' = y_4' = y_5
        y[5],                                  # x_2'' = y_5' = y_6
        -y[0]**3 + y[3]**2 + y[0] + np.sin(t)  # x_2''' = y_6' = -x_1^3 + x_2^2 + x_1 + sin(t)
    ]

    return dydt

# Time span for integration
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

# Initial conditions (all zeros for demonstration)
# [x_1(0), x_1'(0), x_1''(0), x_2(0), x_2'(0), x_2''(0)]
y0 = [0, 0, 0, 0, 0, 0]

# Solve the system
solution = solve_ivp(system_of_odes, t_span, y0, method='RK45', t_eval=t_eval)

# Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(solution.t, solution.y[0])
plt.title('x₁(t)')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(solution.t, solution.y[1])
plt.title('x₁\'(t)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(solution.t, solution.y[3])
plt.title('x₂(t)')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(solution.t, solution.y[4])
plt.title('x₂\'(t)')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(solution.t, solution.y[5])
plt.title('x₂\'\'(t)')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(solution.y[0], solution.y[3])
plt.title('Phase Portrait: x₁ vs x₂')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.grid(True)

plt.tight_layout()
plt.show()