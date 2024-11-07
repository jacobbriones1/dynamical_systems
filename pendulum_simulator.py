import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define the PendulumSimulator class
class PendulumSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Driven Damped Pendulum Simulator with Phase Space Plot")

        # Default parameters
        self.length = 1.0        # Length of the pendulum (m)
        self.mass = 1.0          # Mass of the pendulum bob (kg)
        self.damping = 0.05      # Damping coefficient (kg/s)
        self.driving_amp = 1.2   # Driving amplitude (N·m)
        self.driving_freq = 2/3  # Driving frequency (rad/s)
        self.gravity = 9.81      # Acceleration due to gravity (m/s^2)

        # Time settings
        self.dt = 0.05           # Time step (s)
        self.t = 0.0             # Current time

        # Initial conditions: [theta, omega]
        self.init_theta = 0.2    # Initial angle (radians)
        self.init_omega = 0.0    # Initial angular velocity (rad/s)
        self.state = [self.init_theta, self.init_omega]

        # Data for phase space plot
        self.theta_vals = []
        self.omega_vals = []

        # Set up the figure and axes
        self.fig = plt.Figure(figsize=(10, 5))  # Increased width for two plots

        # Create subplots
        self.ax_pendulum = self.fig.add_subplot(1, 2, 1)
        self.ax_phase = self.fig.add_subplot(1, 2, 2)

        # Configure pendulum axis
        self.update_axes_limits()
        self.ax_pendulum.set_aspect('equal')
        self.ax_pendulum.axis('off')

        # Configure phase space axis
        self.ax_phase.set_xlabel('Angle (rad)')
        self.ax_phase.set_ylabel('Angular Velocity (rad/s)')
        self.ax_phase.set_xlim(-np.pi, np.pi)
        self.ax_phase.set_ylim(-10, 10)
        self.phase_line, = self.ax_phase.plot([], [], lw=1)

        # Pendulum line and bob
        self.line, = self.ax_pendulum.plot([], [], lw=2)
        self.bob, = self.ax_pendulum.plot([], [], 'o', markersize=20 * self.mass)

        # Set up the canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create sliders for parameters
        self.create_sliders()

        # Start the animation
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=self.dt*1000,
            blit=True,
            init_func=self.init_animation,
            cache_frame_data=False  # Suppress the warning about frame data caching
        )

    def create_sliders(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create frames for left and right columns
        left_frame = tk.Frame(control_frame)
        right_frame = tk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Left column sliders
        tk.Label(left_frame, text="Length (m):").pack(anchor='w')
        self.length_slider = tk.Scale(
            left_frame, from_=0.5, to=5.0, resolution=0.1,
            orient=tk.HORIZONTAL, length=200
        )
        self.length_slider.set(self.length)
        self.length_slider.pack()

        tk.Label(left_frame, text="Mass (kg):").pack(anchor='w')
        self.mass_slider = tk.Scale(
            left_frame, from_=0.1, to=5.0, resolution=0.1,
            orient=tk.HORIZONTAL, length=200
        )
        self.mass_slider.set(self.mass)
        self.mass_slider.pack()

        tk.Label(left_frame, text="Damping:").pack(anchor='w')
        self.damping_slider = tk.Scale(
            left_frame, from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, length=200
        )
        self.damping_slider.set(self.damping)
        self.damping_slider.pack()

        # Right column sliders
        tk.Label(right_frame, text="Driving Amplitude:").pack(anchor='w')
        self.driving_amp_slider = tk.Scale(
            right_frame, from_=0.0, to=2.0, resolution=0.05,
            orient=tk.HORIZONTAL, length=200
        )
        self.driving_amp_slider.set(self.driving_amp)
        self.driving_amp_slider.pack()

        tk.Label(right_frame, text="Driving Frequency:").pack(anchor='w')
        self.driving_freq_slider = tk.Scale(
            right_frame, from_=0.0, to=5.0, resolution=0.05,
            orient=tk.HORIZONTAL, length=200
        )
        self.driving_freq_slider.set(self.driving_freq)
        self.driving_freq_slider.pack()

        tk.Label(right_frame, text="Initial Angle (rad):").pack(anchor='w')
        self.init_theta_slider = tk.Scale(
            right_frame, from_=-np.pi, to=np.pi, resolution=0.05,
            orient=tk.HORIZONTAL, length=200
        )
        self.init_theta_slider.set(self.init_theta)
        self.init_theta_slider.pack()

        tk.Label(right_frame, text="Initial Angular Velocity (rad/s):").pack(anchor='w')
        self.init_omega_slider = tk.Scale(
            right_frame, from_=-5.0, to=5.0, resolution=0.1,
            orient=tk.HORIZONTAL, length=200
        )
        self.init_omega_slider.set(self.init_omega)
        self.init_omega_slider.pack()

        # Reset Button
        self.reset_button = tk.Button(control_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.BOTTOM, pady=10)


    def update_axes_limits(self):
        limit = 1.5 * self.length
        self.ax_pendulum.set_xlim(-limit, limit)
        self.ax_pendulum.set_ylim(-limit, limit)
        self.ax_pendulum.figure.canvas.draw()

    def reset_simulation(self):
        # Reset time
        self.t = 0.0

        # Read initial conditions from sliders
        self.init_theta = float(self.init_theta_slider.get())
        self.init_omega = float(self.init_omega_slider.get())
        self.state = [self.init_theta, self.init_omega]

        # Clear phase space data
        self.theta_vals = []
        self.omega_vals = []
        self.phase_line.set_data([], [])

        # Reset pendulum position
        x = self.length * np.sin(self.init_theta)
        y = -self.length * np.cos(self.init_theta)
        self.line.set_data([0, x], [0, y])
        self.bob.set_data([x], [y])

        # Redraw the canvas
        self.canvas.draw()
    def init_animation(self):
        self.line.set_data([], [])
        self.bob.set_data([], [])
        self.phase_line.set_data([], [])
        return self.line, self.bob, self.phase_line

    def animate(self, frame):
        # Read current parameters from sliders
        self.length = float(self.length_slider.get())
        self.mass = float(self.mass_slider.get())
        self.damping = float(self.damping_slider.get())
        self.driving_amp = float(self.driving_amp_slider.get())
        self.driving_freq = float(self.driving_freq_slider.get())
        self.init_theta = float(self.init_theta_slider.get())
        self.init_omega = float(self.init_omega_slider.get())

        # Update bob size based on mass
        self.bob.set_markersize(20 * self.mass)

        # Update axes limits if length has changed
        self.update_axes_limits()

        # Time span for this step
        t_span = [self.t, self.t + self.dt]

        # Integrate ODE for this small time step
        sol = odeint(self.derivatives, self.state, t_span)
        theta = sol[1, 0]
        omega = sol[1, 1]
        self.state = [theta, omega]
        self.t += self.dt

        # Store data for phase space plot
        self.theta_vals.append(theta)
        self.omega_vals.append(omega)

        # Limit the number of points to prevent memory issues
        max_points = 1000
        if len(self.theta_vals) > max_points:
            self.theta_vals = self.theta_vals[-max_points:]
            self.omega_vals = self.omega_vals[-max_points:]

        # Update phase space plot
        self.phase_line.set_data(self.theta_vals, self.omega_vals)

        # Calculate pendulum position
        x = self.length * np.sin(theta)
        y = -self.length * np.cos(theta)

        # Update the line and bob
        self.line.set_data([0, x], [0, y])
        self.bob.set_data([x], [y])

        return self.line, self.bob, self.phase_line

    def derivatives(self, state, t):
        theta, omega = state
        # Ensure theta stays within [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        dtheta_dt = omega
        domega_dt = (-self.damping / self.mass) * omega - (self.gravity / self.length) * np.sin(theta) + \
                    (self.driving_amp / (self.mass * self.length**2)) * np.cos(self.driving_freq * t)
        return [dtheta_dt, domega_dt]

# Run the simulator
if __name__ == "__main__":
    root = tk.Tk()
    simulator = PendulumSimulator(root)
    root.mainloop()
