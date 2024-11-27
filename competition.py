# Improved Competition Model with Irreversible Extinction

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

# Define the competition model without extinction conditions
def competition_model(t, y, r1, r2, b1, b2):
    N1, N2 = y
    
    # Temporarily disable extinction conditions to debug
    dN1dt = r1 * N1 - b1 * N1 * N2
    dN2dt = r2 * N2 - b2 * N1 * N2

    return [dN1dt, dN2dt]

# Initial parameters
r1_init = 1.0      # Growth rate of species 1
r2_init = 1.0      # Growth rate of species 2
b1_init = 0.1      # Competition effect of species 2 on species 1
b2_init = 0.1      # Competition effect of species 1 on species 2

N1_init = 50.0     # Initial population of species 1
N2_init = 30.0     # Initial population of species 2

# Time span
t_max = 200
t_span = [0, t_max]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Set up the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)
line1, = ax.plot([], [], lw=2, label='Trajectory in Phase Plane')
ax.set_xlabel('Species 1 Population')
ax.set_ylabel('Species 2 Population')
ax.set_title('Phase Plane of the Improved Competition Model')
ax.grid(True)
ax.legend()

# Add text annotation for real-time population display
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top',
               fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Define sliders for parameters and initial conditions
axcolor = 'lightgoldenrodyellow'
ax_r1 = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_r2 = plt.axes([0.25, 0.26, 0.65, 0.03], facecolor=axcolor)
ax_b1 = plt.axes([0.25, 0.22, 0.65, 0.03], facecolor=axcolor)
ax_b2 = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
ax_N1 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_N2 = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)

s_r1 = Slider(ax_r1, 'r1 (Growth Rate of Species 1)', 0.0, 3.0, valinit=r1_init, valstep=0.01)
s_r2 = Slider(ax_r2, 'r2 (Growth Rate of Species 2)', 0.0, 3.0, valinit=r2_init, valstep=0.01)
s_b1 = Slider(ax_b1, 'b1 (Competition on Species 1)', 0.0, 1.0, valinit=b1_init, valstep=0.01)
s_b2 = Slider(ax_b2, 'b2 (Competition on Species 2)', 0.0, 1.0, valinit=b2_init, valstep=0.01)
s_N1 = Slider(ax_N1, 'Initial Species 1 (N1)', 0.0, 200.0, valinit=N1_init, valstep=1)
s_N2 = Slider(ax_N2, 'Initial Species 2 (N2)', 0.0, 200.0, valinit=N2_init, valstep=1)

# Variables to control animation state
is_paused = False
frame_number = [0]  # List to make frame_number mutable in nested functions

# Function to create and start the animation
def start_animation():
    global ani
    ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), interval=25, blit=False)
    plt.draw()

# Animation function
def animate(i):
    if is_paused:
        return line1, text
    frame = frame_number[0]
    line1.set_data(N1[:frame+1], N2[:frame+1])  # Note: Removed brackets around data
    current_N1 = max(N1[frame], 0)
    current_N2 = max(N2[frame], 0)
    text.set_text(f'Species 1 (N1): {int(np.round(current_N1))}\nSpecies 2 (N2): {int(np.round(current_N2))}')
    frame_number[0] = frame + 1
    if frame_number[0] >= len(t_eval):
        frame_number[0] = len(t_eval) - 1  # Stop at the last frame
    return line1, text

# Update function for sliders
def update(val):
    global N1, N2, ani, frame_number, is_paused
    # Stop the animation
    if 'ani' in globals():
        ani.event_source.stop()
    is_paused = False
    button_start_stop.label.set_text('Stop')
    frame_number[0] = 0  # Reset frame number

    # Get new parameters
    r1 = s_r1.val
    r2 = s_r2.val
    b1 = s_b1.val
    b2 = s_b2.val
    N1_0 = s_N1.val
    N2_0 = s_N2.val
    y0 = [N1_0, N2_0]
    params = (r1, r2, b1, b2)

    # Recompute the solution
    solution = solve_ivp(
        competition_model, t_span, y0, t_eval=t_eval, args=params,
        method='RK45', rtol=1e-8, atol=1e-10
    )
    N1 = solution.y[0]
    N2 = solution.y[1]

    # Print diagnostic information to help debug
    print(f'Initial populations: Species 1 (N1): {N1_0}, Species 2 (N2): {N2_0}')
    print(f'Parameter values: r1: {r1}, r2: {r2}, b1: {b1}, b2: {b2}')
    print(f'First few values of N1: {N1[:10]}')
    print(f'First few values of N2: {N2[:10]}')

    # Clear the line data and text
    line1.set_data([], [])
    text.set_text('')

    # Update plot limits
    ax.set_xlim(0, max(max(N1), 1)*1.1)
    ax.set_ylim(0, max(max(N2), 1)*1.1)

    # Restart the animation
    start_animation()

# Connect the sliders to the update function
s_r1.on_changed(update)
s_r2.on_changed(update)
s_b1.on_changed(update)
s_b2.on_changed(update)
s_N1.on_changed(update)
s_N2.on_changed(update)

# Reset button
resetax = plt.axes([0.7, 0.92, 0.08, 0.04])
button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_r1.reset()
    s_r2.reset()
    s_b1.reset()
    s_b2.reset()
    s_N1.reset()
    s_N2.reset()
button_reset.on_clicked(reset)

# Start/Stop button
start_stop_ax = plt.axes([0.81, 0.92, 0.08, 0.04])
button_start_stop = Button(start_stop_ax, 'Stop', color=axcolor, hovercolor='0.975')

def start_stop(event):
    global is_paused
    if is_paused:
        is_paused = False
        button_start_stop.label.set_text('Stop')
    else:
        is_paused = True
        button_start_stop.label.set_text('Start')
button_start_stop.on_clicked(start_stop)

# Initial computation and animation
update(None)
plt.show()
