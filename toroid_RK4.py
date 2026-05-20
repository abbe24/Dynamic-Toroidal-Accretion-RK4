# Truman Abbe | Utah State University | truman.abbe23@gmail.com | 05/20/2026

# This is a dynamic model for a growing and spinning toroid. It is assumed that 
# the tangent velocity and volume accretion rate are constant. RK4 is used for 
# integration. An animation for debugging is included.

# Variables:
# f         |  m**2/s  |  volume accretion rate
# R         |  m       |  centerline radius
# v         |  m/s     |  surface velocity
# ro        |  m       |  outer radius
# ss        |  m       |  arc length
# th        |  rad     |  rotation angle
# om        |  rad/s   |  angular velocity
# omdot     |  rad/s**2|  angular acceleration
# rodot     |  m/s     |  radial velocity
# rodotdot  |  m/s**2  |  radial acceleration
# sa        |  m**2    |  surface area

import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import os

def rhs(t, x, params):
    ro = x[0]
    f = params["f"]
    R = params["R"]

    rodot = f / (4*np.pi**2*R*(ro-R))

    return np.array([rodot])

def compute_outputs(t, x, params):
    f = params["f"]
    R = params["R"]
    v = params["v"]

    ss0 = params["ss0"]
    ss          = ss0 + v * t
    ssdot       = v
    ssdotdot    = 0.0
    ro          = x[0]
    rodot       = f / (4*np.pi**2*R*(ro-R))
    rodotdot    = -rodot**2 / (ro-R)
    th          = ss / ro
    thdot       = (ro*ssdot-ss*rodot) / ro**2
    om          = thdot
    omdot       = (ro**2*ssdotdot-ro*ss*rodotdot-2*ro*rodot*ssdot+2*ss*rodot**2) / ro**3
    sa          = 4*np.pi**2*R*(ro-R)

    outputs = np.array([ro, ss, th, om, omdot, rodot, rodotdot, sa])

    return outputs

def rk4_step(rhs, t, x, dt, params):
    k1 = rhs(t, x, params)
    k2 = rhs(t + 0.5*dt, x + 0.5*dt*k1, params)
    k3 = rhs(t + 0.5*dt, x + 0.5*dt*k2, params)
    k4 = rhs(t + dt, x + dt*k3, params)

    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(rhs, x0, t0, tf, dt, params):
    times = np.arange(t0, tf + dt, dt)
    states = np.zeros((len(times), len(x0)))
    outputs_history = np.zeros((len(times), 8))

    states[0, :] = x0
    outputs_history[0, :] = compute_outputs(times[0], states[0, :], params)

    total_steps = len(times) - 1
    print("Starting integration...")
    start_time = time.perf_counter()
    last_print_time = start_time

    for k in range(total_steps):
        states[k + 1, :] = rk4_step(rhs, times[k], states[k, :], dt, params)
        outputs_history[k + 1, :] = compute_outputs(times[k + 1], states[k + 1, :], params)

        current_time = time.perf_counter()
        if current_time - last_print_time >= 1.0:  # Has 1 second passed?
            elapsed = current_time - start_time
            percent = ((k + 1) / total_steps) * 100.0
            print(f"Status: {elapsed:>6.1f}s elapsed | {percent:>5.1f}% complete")
            last_print_time = current_time  # Reset the 1-second timer

    total_time = time.perf_counter() - start_time
    print(f"Integration complete. Total wall time: {total_time:.4f}s\n")

    return times, states, outputs_history

# -----------------------------
# Initial conditions and run
# -----------------------------

f = 9.0e-5      
R = 0.35        
v = 0.01       

ro0 = 0.4       
th0 = 0.0       
ss0 = th0 * ro0

t0 = 0.0
tf = 60 * 10    
dt = 0.05       

x0 = np.array([ro0])

params = {
    "f": f,
    "R": R,
    "v": v,
    "ss0": ss0,
}

times, states, outputs_history = integrate(rhs, x0, t0, tf, dt, params)

ro = outputs_history[:, 0]
ss = outputs_history[:, 1]
th = outputs_history[:, 2]
om = outputs_history[:, 3]
omdot = outputs_history[:, 4]
rodot = outputs_history[:, 5]
rodotdot = outputs_history[:, 6]
sa = outputs_history[:, 7]

print(f"{'time':<10} {'ro':<14} {'ss':<14} {'theta':<14} {'omega':<14} {'omegadot':<14} {'rodot':<14} {'rodotdot':<14} {'sa':<14}")

for k in range(len(times)):
    if times[k] % 1.0 < dt: 
        print(f"{times[k]:<10.4g} {ro[k]:<14.4g} {ss[k]:<14.4g} {th[k]:<14.4g} {om[k]:<14.4g} {omdot[k]:<14.4g} {rodot[k]:<14.4g} {rodotdot[k]:<14.4g} {sa[k]:<14.4g}")

plt.figure()
plt.plot(times, ro, marker="o")
plt.xlabel("Time [s]")
plt.ylabel("Outer radius ro [m]")
plt.grid(True)

# -----------------------------
# 3D Animation & Video Export
# -----------------------------

print("Generating 3D animation... this may take a minute.")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, 2 * np.pi, 30)
U, V = np.meshgrid(u, v)

total_frames = len(times)
skip = max(1, total_frames // 60)

def update(frame_idx):
    print(f"Rendering frame at t = {times[frame_idx]:.1f} s")
    ax.clear()  
    
    lim = 0.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim/2, lim/2)
    ax.set_box_aspect((1, 1, 0.5)) 
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f"Toroidal Accretion Simulation")

    current_th = th[frame_idx]
    cos_th = np.cos(current_th)
    sin_th = np.sin(current_th)

    ax.quiver(0, 0, 0, 0.5 * cos_th, 0.5 * sin_th, 0, color='r', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, -0.5 * sin_th, 0.5 * cos_th, 0, color='g', linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 0.3, color='b', linewidth=2, arrow_length_ratio=0.1)

    current_ro = ro[frame_idx]
    r_a = current_ro - R  

    X_base = (R + r_a * np.cos(V)) * np.cos(U)
    Y_base = (R + r_a * np.cos(V)) * np.sin(U)
    Z = r_a * np.sin(V)

    X = X_base * cos_th - Y_base * sin_th
    Y = X_base * sin_th + Y_base * cos_th

    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.6, edgecolor='black', linewidth=0.1)

    stats = (
        f"Time    : {times[frame_idx]:.1f} s\n"
        f"ro      : {current_ro:.5f} m\n"
        f"ss      : {ss[frame_idx]:.5f} m\n"
        f"theta   : {th[frame_idx]:.4f} rad\n"
        f"omega   : {om[frame_idx]:.4g} rad/s\n"
        f"omegadot: {omdot[frame_idx]:.4g} rad/s^2\n"
        f"rodot   : {rodot[frame_idx]:.4g} m/s\n"
        f"rodotdot: {rodotdot[frame_idx]:.4g} m/s^2\n"
        f"SA      : {sa[frame_idx]:.4f} m^2"
    )
    
    ax.text2D(0.02, 0.98, stats, transform=ax.transAxes,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

frames_to_render = np.arange(0, total_frames, skip)

ani = animation.FuncAnimation(fig, update, frames=frames_to_render, interval=50)

save_name = "animation.gif"
save_path = os.path.join(os.getcwd(), save_name)

print(f"Saving video to {save_path}...")
ani.save(save_path, writer="pillow", fps=20)
print(f"Saved successfully! Check your project folder for '{save_name}'.")

plt.show()
