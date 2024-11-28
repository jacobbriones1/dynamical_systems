import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Gravitational acceleration (m/s²)

def hamiltonian(theta, p_theta, omega, m, R):
    """
    Compute the Hamiltonian of the system.
    
    Parameters:
    - theta: Angle of the bead on the hoop (rad)
    - p_theta: Momentum conjugate to theta
    - omega: Rotation rate of the hoop (rad/s)
    - m: Mass of the bead (kg)
    - R: Radius of the hoop (m)
    
    Returns:
    - H: Hamiltonian value
    """
    kinetic = p_theta**2 / (2 * m * R**2)
    rotational = 0.5 * m * R**2 * omega**2 * np.sin(theta)**2
    potential = m * g * R * (1 - np.cos(theta))
    return kinetic + rotational + potential

def equations_of_motion(t, y, omega, m, R):
    """
    Define the equations of motion for the bead on the rotating hoop.
    
    Parameters:
    - t: Time variable
    - y: List containing [theta, p_theta]
    - omega: Rotation rate of the hoop (rad/s)
    - m: Mass of the bead (kg)
    - R: Radius of the hoop (m)
    
    Returns:
    - dydt: Derivatives [d(theta)/dt, d(p_theta)/dt]
    """
    theta, p_theta = y
    dtheta_dt = p_theta / (m * R**2)
    dp_theta_dt = -m * g * R * np.sin(theta) + m * R**2 * omega**2 * np.sin(theta) * np.cos(theta)
    return [dtheta_dt, dp_theta_dt]

class BeadHoopSimulation:
    def __init__(self):
        # Initialize simulation parameters
        self.omega = 2.0          # Rotation rate (rad/s)
        self.theta0 = 0.1         # Initial angle (rad)
        self.p_theta0 = 0.0       # Initial momentum (kg·m²/s)
        self.m = 1.0              # Mass of the bead (kg)
        self.R = 1.0              # Radius of the hoop (m)

        # Initialize the figure and subplots
        self.fig = plt.figure(figsize=(16, 8))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        plt.subplots_adjust(left=0.1, bottom=0.35)

        # Initialize colorbar reference
        self.cbar = None

        # Solve initial trajectory
        self.solve_trajectory()

        # Plot initial phase portrait
        self.plot_phase_portrait()

        # Setup 3D plot elements
        self.setup_3d_plot()

        # Setup sliders
        self.setup_sliders()

        # Initialize animation
        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.sol.t),
            interval=20,  # 50 fps
            blit=False
        )

    def solve_trajectory(self):
        """
        Solve the equations of motion for the bead on the rotating hoop.
        """
        self.sol = solve_ivp(
            equations_of_motion,
            [0, 20],
            [self.theta0, self.p_theta0],
            args=(self.omega, self.m, self.R),
            t_eval=np.linspace(0, 20, 300)  # Reduced frames for performance
        )

    def plot_phase_portrait(self):
        """
        Plot the phase portrait with Hamiltonian contours and bead trajectory.
        """
        self.ax1.cla()  # Clear previous plot

        # Define grid for phase portrait
        theta_vals = np.linspace(-np.pi, np.pi, 200)
        p_theta_vals = np.linspace(-10, 10, 200)
        Theta, P_theta = np.meshgrid(theta_vals, p_theta_vals)

        # Compute Hamiltonian
        H = hamiltonian(Theta, P_theta, self.omega, self.m, self.R)

        # Plot filled contour
        contour = self.ax1.contourf(Theta, P_theta, H, levels=25, cmap='viridis')

        # Add or update colorbar
        if self.cbar is None:
            self.cbar = self.fig.colorbar(contour, ax=self.ax1, label='Hamiltonian (H)')
        else:
            self.cbar.update_normal(contour)

        # Set plot titles and labels
        self.ax1.set_title(f"Phase Portrait\nω={self.omega:.2f} rad/s, m={self.m:.2f} kg, R={self.R:.2f} m")
        self.ax1.set_xlabel("θ (rad)")
        self.ax1.set_ylabel("p_θ")
        self.ax1.set_ylim(-10, 10)
        self.ax1.axvline(0, color='white', linestyle='--')
        self.ax1.axvline(np.pi, color='white', linestyle='--')
        self.ax1.axvline(-np.pi, color='white', linestyle='--')

        # Plot bead trajectory on phase portrait
        if hasattr(self, 'traj_line'):
            self.traj_line.remove()
        if hasattr(self, 'start_point'):
            self.start_point.remove()
        self.traj_line, = self.ax1.plot(self.sol.y[0], self.sol.y[1], color='red', linewidth=2, label='Trajectory')
        self.start_point, = self.ax1.plot([self.sol.y[0][0]], [self.sol.y[1][0]], 'bo', label='Start Point')
        self.ax1.legend()
        self.ax1.grid(True)

    def setup_3d_plot(self):
        """
        Setup the 3D plot with the rotating hoop, bead, and trace.
        """
        self.ax2.cla()  # Clear previous plot

        # Set plot limits and aspect
        self.ax2.set_xlim(-self.R - 0.2, self.R + 0.2)
        self.ax2.set_ylim(-self.R - 0.2, self.R + 0.2)
        self.ax2.set_zlim(-self.R - 0.2, self.R + 0.2)
        self.ax2.set_box_aspect([1,1,1])  # Equal aspect ratio
        self.ax2.view_init(elev=-20, azim=30)  # Flipped viewing angle to have equilibrium at bottom

        # Define initial hoop coordinates (hoop in y-z plane initially)
        theta_hoop = np.linspace(0, 2*np.pi, 50)  # Reduced points for performance
        self.x_hoop_initial = np.zeros_like(theta_hoop)  # Initially in y-z plane (x=0)
        self.y_hoop_initial = self.R * np.sin(theta_hoop)
        self.z_hoop_initial = -self.R * np.cos(theta_hoop)  # Negative for equilibrium at bottom

        # Plot initial hoop (stationary, will be rotated in animation)
        self.hoop_line, = self.ax2.plot(
            self.x_hoop_initial, self.y_hoop_initial, self.z_hoop_initial, 'k-', lw=2
        )

        # Initialize bead and trace
        self.bead, = self.ax2.plot([], [], [], 'ro', markersize=8)
        self.trace, = self.ax2.plot([], [], [], 'r-', lw=1, alpha=0.6)  # Trace path
        self.x_trace, self.y_trace, self.z_trace = [], [], []

    def animate(self, i):
        """
        Animation function to update the hoop rotation, bead position, and trace.
        """
        # Current time and rotation angle
        t = self.sol.t[i]
        phi = self.omega * t  # Rotation angle around z-axis (rad)

        # Precompute cosine and sine of rotation angle
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Rotate the hoop
        x_rot = self.x_hoop_initial * cos_phi - self.y_hoop_initial * sin_phi
        y_rot = self.x_hoop_initial * sin_phi + self.y_hoop_initial * cos_phi
        z_rot = self.z_hoop_initial
        self.hoop_line.set_data(x_rot, y_rot)
        self.hoop_line.set_3d_properties(z_rot)

        # Bead position on the rotating hoop
        theta = self.sol.y[0][i]
        # Bead's position in the hoop's local frame (y-z plane)
        x_bead_local = 0  # Always on the hoop, which lies in y-z plane before rotation
        y_bead_local = self.R * np.sin(theta)
        z_bead_local = -self.R * np.cos(theta)  # Negative for equilibrium at bottom

        # Rotate bead's position with the hoop
        x_bead = x_bead_local * cos_phi - y_bead_local * sin_phi
        y_bead = x_bead_local * sin_phi + y_bead_local * cos_phi
        z_bead = z_bead_local

        # Update bead position
        self.bead.set_data([x_bead], [y_bead])
        self.bead.set_3d_properties([z_bead])

        # Update trace
        self.x_trace.append(x_bead)
        self.y_trace.append(y_bead)
        self.z_trace.append(z_bead)
        self.trace.set_data(self.x_trace, self.y_trace)
        self.trace.set_3d_properties(self.z_trace)

        return self.hoop_line, self.bead, self.trace

    def setup_sliders(self):
        """
        Setup interactive sliders for adjusting simulation parameters.
        """
        # Define slider positions
        ax_omega = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        ax_theta0 = plt.axes([0.1, 0.20, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        ax_p_theta0 = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        ax_mass = plt.axes([0.1, 0.10, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        ax_radius = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')

        # Create sliders
        self.omega_slider = Slider(ax_omega, 'ω (Rotation Rate rad/s)', 0.1, 5.0, valinit=self.omega, valstep=0.1)
        self.theta0_slider = Slider(ax_theta0, 'θ₀ (Initial Angle rad)', -np.pi, np.pi, valinit=self.theta0, valstep=0.1)
        self.p_theta0_slider = Slider(ax_p_theta0, 'p₀ (Initial Momentum kg·m²/s)', -10, 10, valinit=self.p_theta0, valstep=0.1)
        self.mass_slider = Slider(ax_mass, 'm (Mass kg)', 0.1, 5.0, valinit=self.m, valstep=0.1)
        self.radius_slider = Slider(ax_radius, 'R (Radius m)', 0.1, 5.0, valinit=self.R, valstep=0.1)

        # Connect sliders to the update function
        self.omega_slider.on_changed(self.update_parameters)
        self.theta0_slider.on_changed(self.update_parameters)
        self.p_theta0_slider.on_changed(self.update_parameters)
        self.mass_slider.on_changed(self.update_parameters)
        self.radius_slider.on_changed(self.update_parameters)

    def update_parameters(self, val):
        """
        Update simulation parameters based on slider values and refresh plots.
        """
        # Update parameters from sliders
        self.omega = self.omega_slider.val
        self.theta0 = self.theta0_slider.val
        self.p_theta0 = self.p_theta0_slider.val
        self.m = self.mass_slider.val
        self.R = self.radius_slider.val

        # Solve new trajectory
        self.solve_trajectory()

        # Update phase portrait
        self.plot_phase_portrait()

        # Reset 3D plot with new radius
        self.setup_3d_plot()

        # Clear existing trace
        self.x_trace, self.y_trace, self.z_trace = [], [], []
        self.trace.set_data([], [])
        self.trace.set_3d_properties([])

        # Restart animation
        if self.anim is not None:
            self.anim.event_source.stop()
        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.sol.t),
            interval=20,  # Increased frame rate (50 fps)
            blit=False
        )

        plt.draw()

# Instantiate and run the simulation
simulation = BeadHoopSimulation()
plt.show()
