import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Processor
from tqdm import tqdm
import imageio.v2 as imageio

# Constants from the original code
Offset = Processor.OFFSETS
Pos = Processor.ANTENNA_POS
PAD_R = Processor.PAD_R
PAD_D = Processor.PAD_D

def load_data_info(data_file):
    """Load data and compute resolution parameters"""
    data, f0, B, Ms, Mc, Ts, Tc = Processor.load_file(data_file)
    N_frame = data.shape[0]
    
    # Get sample RDM to determine dimensions
    sample_rdm = Processor.compute_RDM(data_file, 0)[0]
    N_v, N_r = sample_rdm.shape
    
    # Calculate resolution parameters
    delta_r = 3e8 / (2 * (B * PAD_R))
    delta_v = (3e8 / (2 * f0)) * (1.0 / (Mc * Tc) / PAD_D)
    
    return data, f0, B, Ms, Mc, Ts, Tc, N_frame, N_v, N_r, delta_r, delta_v

def targets_to_physical_coords(targets, delta_r, delta_v, N_v, channel_idx=0):
    """Convert target indices to physical coordinates"""
    points = []
    for (d_idx, r_idx), _ in targets:
        v = (d_idx - N_v // 2) * delta_v
        r = r_idx * delta_r - Offset[channel_idx]
        points.append((r, v))
    return np.array(points) if points else np.empty((0, 2))

def plot_basic_rdms(data_file, anim=False, save_path=None):
    """Plot or animate basic RDMs for all 4 channels"""
    data, f0, B, Ms, Mc, Ts, Tc, N_frame, N_v, N_r, delta_r, delta_v = load_data_info(data_file)
    
    if not anim:
        # Static plot of first frame
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        frame_idx = 0
        RDMs = Processor.compute_RDM(data_file, frame_idx)
        
        for ch in range(4):
            rdm = RDMs[ch]
            doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
            ranges = np.arange(rdm.shape[1]) * delta_r - Offset[ch]
            velocities = doppler_bins * delta_v
            
            axs[ch].imshow(
                rdm,
                cmap='jet',
                origin='lower',
                extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
                aspect='auto'
            )
            axs[ch].set_title(f"Canal {ch}")
            axs[ch].set_xlabel("Distance (m)")
            axs[ch].set_ylabel("Vitesse (m/s)")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    else:
        # Animation
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        # Initialize plots
        images = []
        titles = []
        for ch in range(4):
            rdm = Processor.compute_RDM(data_file, 0)[ch]
            doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
            ranges = np.arange(rdm.shape[1]) * delta_r - Offset[ch]
            velocities = doppler_bins * delta_v
            
            img = axs[ch].imshow(
                rdm,
                cmap='jet',
                origin='lower',
                extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
                aspect='auto'
            )
            images.append(img)
            title = axs[ch].set_title(f"Canal {ch} - Frame 1/{N_frame}")
            titles.append(title)
            axs[ch].set_xlabel("Distance (m)")
            axs[ch].set_ylabel("Vitesse (m/s)")
        
        def update(frame_idx):
            RDMs = Processor.compute_RDM(data_file, frame_idx)
            for ch in range(4):
                rdm = RDMs[ch]
                images[ch].set_data(rdm)
                images[ch].set_clim(rdm.min(), rdm.max())
                titles[ch].set_text(f"Canal {ch} - Frame {frame_idx+1}/{N_frame}")
            return images + titles
        
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(N_frame),
            interval=200,  # 200ms between frames
            blit=True,
            repeat=True
        )
        
        plt.tight_layout()
        if save_path:
            ani.save(save_path, writer='pillow', fps=5)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

def plot_multi_target_rdms(data_file, psf, anim=False, save_path=None):
    """Plot or animate RDMs with multiple targets"""
    data, f0, B, Ms, Mc, Ts, Tc, N_frame, N_v, N_r, delta_r, delta_v = load_data_info(data_file)
    
    if not anim:
        # Static plot of first frame with targets
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        frame_idx = 0
        RDMs = Processor.compute_RDM(data_file, frame_idx)
        
        for ch in range(4):
            rdm = RDMs[ch]
            raw_targets, _, _ = Processor.clean_rdm(rdm, psf, use_binary_mask=True)
            pts = targets_to_physical_coords(raw_targets, delta_r, delta_v, rdm.shape[0], ch)
            
            doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
            ranges = np.arange(rdm.shape[1]) * delta_r - Offset[ch]
            velocities = doppler_bins * delta_v
            
            axs[ch].imshow(
                rdm,
                cmap='jet',
                origin='lower',
                extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
                aspect='auto'
            )
            axs[ch].set_title(f"Canal {ch} - Avec cibles")
            axs[ch].set_xlabel("Distance (m)")
            axs[ch].set_ylabel("Vitesse (m/s)")
            
            if len(pts) > 0:
                axs[ch].scatter(pts[:, 0], pts[:, 1], edgecolors='white', facecolors='none', s=50)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    else:
        # Animation with targets
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        # Initialize plots
        images = []
        titles = []
        scatters = []
        
        for ch in range(4):
            rdm = Processor.compute_RDM(data_file, 0)[ch]
            doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
            ranges = np.arange(rdm.shape[1]) * delta_r - Offset[ch]
            velocities = doppler_bins * delta_v
            
            img = axs[ch].imshow(
                rdm,
                cmap='jet',
                origin='lower',
                extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
                aspect='auto'
            )
            images.append(img)
            
            title = axs[ch].set_title(f"Canal {ch} - Frame 1/{N_frame}")
            titles.append(title)
            
            scatter = axs[ch].scatter([], [], edgecolors='white', facecolors='none', s=50)
            scatters.append(scatter)
            
            axs[ch].set_xlabel("Distance (m)")
            axs[ch].set_ylabel("Vitesse (m/s)")
        
        def update(frame_idx):
            RDMs = Processor.compute_RDM(data_file, frame_idx)
            for ch in range(4):
                rdm = RDMs[ch]
                raw_targets, _, _ = Processor.clean_rdm(rdm, psf, use_binary_mask=True)
                pts = targets_to_physical_coords(raw_targets, delta_r, delta_v, rdm.shape[0], ch)
                
                images[ch].set_data(rdm)
                images[ch].set_clim(rdm.min(), rdm.max())
                titles[ch].set_text(f"Canal {ch} - Frame {frame_idx+1}/{N_frame}")
                
                if len(pts) > 0:
                    scatters[ch].set_offsets(pts)
                else:
                    scatters[ch].set_offsets(np.empty((0, 2)))
            
            return images + titles + scatters
        
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(N_frame),
            interval=200,
            blit=True,
            repeat=True
        )
        
        plt.tight_layout()
        if save_path:
            ani.save(save_path, writer='pillow', fps=5)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

def plot_clean_iterations(data_file, psf, frame_idx=0, channel_idx=0, save_path=None):
    """Plot the iterations of the CLEAN algorithm for a single RDM"""
    data, f0, B, Ms, Mc, Ts, Tc, N_frame, N_v, N_r, delta_r, delta_v = load_data_info(data_file)
    
    # Get RDM and clean it
    rdm = Processor.compute_RDM(data_file, frame_idx)[channel_idx]
    raw_targets, rdm_final, rdm_steps = Processor.clean_rdm(rdm, psf, use_binary_mask=True)
    
    # Add original RDM to steps
    rdm_steps_full = [rdm] + rdm_steps
    titles = ["RDM brute"] + [f"Itération {i+1}" for i in range(len(rdm_steps))]
    
    # Prepare for plotting
    n_steps = len(rdm_steps_full)
    fig, axs = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4), sharex=True, sharey=True)
    
    # If only one step, make axs iterable
    if n_steps == 1:
        axs = [axs]
    
    # Calculate axes for the plot
    doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
    ranges = np.arange(rdm.shape[1]) * delta_r - Offset[channel_idx]
    velocities = doppler_bins * delta_v
    
    # Find global min/max for consistent colorbars
    global_min = min(np.min(r) for r in rdm_steps_full)
    global_max = max(np.max(r) for r in rdm_steps_full)
    
    # Plot each step
    for i, rdm_step in enumerate(rdm_steps_full):
        im = axs[i].imshow(
            rdm_step,
            cmap='jet',
            origin='lower',
            extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
            aspect='auto',
            vmin=global_min,
            vmax=global_max
        )
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Distance (m)")
        if i == 0:  # Only set y-label for first plot to avoid redundancy
            axs[i].set_ylabel("Vitesse (m/s)")
    
    # Add a common colorbar
    cbar = fig.colorbar(im, ax=axs, shrink=0.9, label="Puissance")
    
    # Show targets in physical coordinates
    target_coords = targets_to_physical_coords(
        Processor.fuse_targets(raw_targets), 
        delta_r, 
        delta_v, 
        rdm.shape[0],
        channel_idx
    )
    
    if len(target_coords) > 0:
        print("Targets detected:", target_coords)
    else:
        print("No targets detected")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_target_trajectory_with_kalman(
    data_file,
    kalman_params=None,
    save_path=None,
):
    """
    Anime la position et la vitesse de la cible dans le plan (x, y) ; la trace
    rouge affiche **toute** la trajectoire prédit / lissée par le filtre Kalman.
    """
    (
        _,
        f0,
        B,
        Ms,
        Mc,
        Ts,
        Tc,
        N_frame,
        _,
        _,
        delta_r,
        delta_v,
    ) = load_data_info(data_file)

    # ────────────────────── paramètres Kalman par défaut ──────────────────────
    if kalman_params is None:
        init_pos = np.array(Processor.compute_position(data_file, 0))
        kalman_params = {
            "kalman_x": np.array([init_pos[0], init_pos[1], 0.0, 0.0]),
            "kalman_p": np.eye(4) * 0.1,
            "outlier_radius": 1.0,
        }

    dt = 0.128  # intervalle entre deux frames

    # ───────────────────────────── figure & objets ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 8))

    # antennes
    ant_pos = Processor.ANTENNA_POS
    ax.scatter(ant_pos[:, 0], ant_pos[:, 1], marker="^", c="k", label="Antennes")

    # mesures et Kalman – points + vecteurs vitesse
    first_pos = np.array(Processor.compute_position(data_file, 0))
    first_vel_vec, _ = Processor.compute_speed(data_file, 0)
    vx0, vy0 = first_vel_vec * dt

    meas_point = ax.scatter(*first_pos, c="b", s=50, label="Mesure")
    kal_point = ax.scatter(
        kalman_params["kalman_x"][0],
        kalman_params["kalman_x"][1],
        c="r",
        s=50,
        label="Kalman",
    )

    meas_quiv = ax.quiver(
        first_pos[0],
        first_pos[1],
        vx0,
        vy0,
        angles="xy",
        pivot="tail",
        scale_units="xy",
        scale=1,
        color="b",
    )

    kal_quiv = ax.quiver(
        kalman_params["kalman_x"][0],
        kalman_params["kalman_x"][1],
        kalman_params["kalman_x"][2] * dt,
        kalman_params["kalman_x"][3] * dt,
        angles="xy",
        pivot="tail",
        scale_units="xy",
        scale=1,
        color="r",
    )

    # ───────────── ajout d'une ligne pour tracer la trajectoire Kalman ─────────
    traj_x, traj_y = [kalman_params["kalman_x"][0]], [kalman_params["kalman_x"][1]]
    (traj_line,) = ax.plot(traj_x, traj_y, "r-", linewidth=1.5, label="Trajectoire Kalman")

    # Axes
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-1, 26)
    ax.legend()

    title = ax.set_title(f"Position de la cible – Frame 1/{N_frame}")

    # état Kalman vivant entre les frames
    kal_state = {
        "x": kalman_params["kalman_x"].copy(),
        "p": kalman_params["kalman_p"].copy(),
    }

    # ──────────────────────────── fonction update ────────────────────────────
    def update(frame_idx):
        # 1) mesure brute
        meas_pos = np.array(Processor.compute_position(data_file, frame_idx))
        meas_vel_vec, _ = Processor.compute_speed(data_file, frame_idx)
        vx, vy = meas_vel_vec * dt

        meas_point.set_offsets(meas_pos)
        meas_quiv.set_offsets(meas_pos)
        meas_quiv.set_UVC(vx, vy)

        # 2) Kalman
        kal_x, kal_p = Processor.kalman_filter_monocible(
            data_file,
            frame_idx,
            kal_state["x"],
            kal_state["p"],
            kalman_params["outlier_radius"],
        )
        kal_state["x"], kal_state["p"] = kal_x, kal_p

        kal_pos = kal_x[:2]
        kal_vel = kal_x[2:] * dt

        kal_point.set_offsets(kal_pos)
        kal_quiv.set_offsets(kal_pos)
        kal_quiv.set_UVC(kal_vel[0], kal_vel[1])

        # 3) mise à jour de la trajectoire
        traj_x.append(kal_pos[0])
        traj_y.append(kal_pos[1])
        traj_line.set_data(traj_x, traj_y)

        # 4) titre
        title.set_text(f"Position de la cible – Frame {frame_idx+1}/{N_frame}")

        return (
            meas_point,
            meas_quiv,
            kal_point,
            kal_quiv,
            traj_line,
            title,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(N_frame),
        interval=50,
        blit=True,
        repeat=True,
    )

    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer="pillow", fps=5)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

def main():
    # File path - update with your data file
    data_file = "data/30-04/marche 2-15m.npz"
    
    # PSF definition for CLEAN algorithm
    psf = Processor.psf_th  # or Processor.psf_empirique_centered
    
    # Visualization mode selection
    # Choose one of: "basic_rdm", "multi_target", "clean_iterations", "trajectory_kalman"
    visualization_mode = "trajectory_kalman"
    
    # Animation flag - if True, creates animation, otherwise static plot
    anim = True
    
    # Save path - set to None to display plot instead of saving
    save_path = None  # or "output.gif" for animations, "output.png" for static plots
    
    # Optional Kalman parameters
    kalman_params = {
        'kalman_x': np.array([0, 2, 0, 0.5]),  # Initial state [x, y, vx, vy]
        'kalman_p': np.eye(4) * 1,          # Initial covariance
        'outlier_radius':100.0                  # Outlier radius
    }
    
    # Choose visualization based on mode
    if visualization_mode == "basic_rdm":
        plot_basic_rdms(data_file, anim, save_path)
    
    elif visualization_mode == "multi_target":
        plot_multi_target_rdms(data_file, psf, anim, save_path)
    
    elif visualization_mode == "clean_iterations":
        # Always static plot for clean iterations
        frame_idx = 0  # Frame to analyze
        channel_idx = 0  # Channel to analyze (0-3)
        plot_clean_iterations(data_file, psf, frame_idx, channel_idx, save_path)
    
    elif visualization_mode == "trajectory_kalman":
        # Always animation for trajectory
        plot_target_trajectory_with_kalman(data_file, kalman_params, save_path)
    
    else:
        print(f"Unknown visualization mode: {visualization_mode}")
        print("Valid options: basic_rdm, multi_target, clean_iterations, trajectory_kalman")

if __name__ == "__main__":
    main()