import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tkinter import Tk, Scale, HORIZONTAL, Button

# Memory kernel function
def memory_kernel(t, alpha):
    return np.exp(-alpha * t)

# Function to simulate the damped oscillator with memory kernel
def damped_oscillator_with_memory(m, k, beta, alpha, x0, v0, t_max, dt):
    t_values = np.arange(0, t_max, dt)
    x_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)
    
    # Initial conditions
    x_values[0] = x0
    v_values[0] = v0
    
    # Time-stepping solution using a simple Euler method
    for i in range(1, len(t_values)):
        t = t_values[i]
        x = x_values[i - 1]
        v = v_values[i - 1]
        
        # Compute the damping force with memory kernel (integral approximated discretely)
        damping_force = 0
        for j in range(i):
            u = t_values[j]
            damping_force += memory_kernel(t - u, alpha) * v_values[j] * dt
        
        # Update velocity and displacement using Newton's laws
        a = (-k * x - beta * damping_force) / m
        v_new = v + a * dt
        x_new = x + v * dt
        
        # Store the new values
        x_values[i] = x_new
        v_values[i] = v_new
    
    return t_values, x_values

# Initialize the figure and axis
fig, ax = plt.subplots()

# Function to animate the motion of the oscillator
def animate_oscillator(m, k, beta, alpha, x0, v0, t_max=50, dt=0.05):
    t_values, x_values = damped_oscillator_with_memory(m, k, beta, alpha, x0, v0, t_max, dt)

    # Clear the previous plot and reset the axis
    ax.clear()
    ax.set_xlim(0, t_max)
    ax.set_ylim(np.min(x_values) * 1.1, np.max(x_values) * 1.1)
    line, = ax.plot([], [], lw=2)
    point, = ax.plot([], [], 'ro')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        # Update the line and point with x and y data from the oscillator
        line.set_data(t_values[:frame], x_values[:frame])  # Line gets sequence of points
        point.set_data([t_values[frame]], [x_values[frame]])  # Point gets single x and y value as lists
        return line, point

    ani = FuncAnimation(fig, update, frames=len(t_values), init_func=init, interval=20, blit=True)
    
    # Ensure plot is updated
    plt.pause(0.001)

# Tkinter GUI for sliders
def run_simulation():
    m = mass_slider.get()
    k = spring_slider.get()
    beta = damping_slider.get()
    alpha = memory_slider.get()
    x0 = x0_slider.get()
    v0 = v0_slider.get()
    
    # Update the existing plot with new parameters
    animate_oscillator(m=m, k=k, beta=beta, alpha=alpha, x0=x0, v0=v0, t_max=50, dt=0.05)

# Create the Tkinter window
root = Tk()
root.title("Damped Oscillator Simulation")

# Create sliders for parameters
mass_slider = Scale(root, from_=0.1, to=10, resolution=0.1, orient=HORIZONTAL, label="Mass")
mass_slider.set(1.0)
mass_slider.pack()

spring_slider = Scale(root, from_=0.1, to=10, resolution=0.1, orient=HORIZONTAL, label="Spring Constant")
spring_slider.set(1.0)
spring_slider.pack()

damping_slider = Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, label="Damping Coefficient")
damping_slider.set(0.1)
damping_slider.pack()

memory_slider = Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, label="Memory Decay")
memory_slider.set(0.1)
memory_slider.pack()

x0_slider = Scale(root, from_=-10, to=10, resolution=0.1, orient=HORIZONTAL, label="Initial Displacement")
x0_slider.set(1.0)
x0_slider.pack()

v0_slider = Scale(root, from_=-10, to=10, resolution=0.1, orient=HORIZONTAL, label="Initial Velocity")
v0_slider.set(0.0)
v0_slider.pack()

# Create button to start the simulation
start_button = Button(root, text="Run Simulation", command=run_simulation)
start_button.pack()

# Start the Tkinter event loop
root.mainloop()

# Show the plot continuously
plt.show()
