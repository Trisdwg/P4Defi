import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Processor

def targets_to_physical_coords(targets, delta_r, delta_v, N_v):
    points = []
    for (d_idx, r_idx), _ in targets:
        v = (d_idx - N_v // 2) * delta_v
        r = r_idx * delta_r
        points.append((r, v))
    return np.array(points)

def main():
    data_file = "data/18-04/doublemarche4-17memevit.npz"
    data, f0, B, Ms, Mc, Ts, Tc = Processor.load_file(data_file)
    N_frame = data.shape[0]

    PAD_R = Processor.PAD_R
    PAD_D = Processor.PAD_D
    sample_rdm = Processor.compute_RDM(data_file, 0)[1]
    N_v, N_r = sample_rdm.shape
    delta_r = 3e8 / (2 * (B * PAD_R))
    vel_res = (3e8 / (2 * f0)) * (1.0 / (Mc * Tc) / PAD_D)

    ranges = np.arange(N_r) * delta_r
    doppler_bins = np.arange(N_v) - N_v // 2
    velocities = doppler_bins * vel_res

    fig, (ax_rdm, ax_map) = plt.subplots(1, 2, figsize=(12, 5))
    dt = 1

    im = ax_rdm.imshow(
        sample_rdm,
        cmap='jet',
        origin='lower',
        extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
        aspect='auto'
    )
    cbar = fig.colorbar(im, ax=ax_rdm)
    cbar.set_label('Puissance')
    ax_rdm.set_xlabel('Distance (m)')
    ax_rdm.set_ylabel('Vitesse (m/s)')
    title_obj = ax_rdm.set_title(f"RDM - frame 1/{N_frame}")

    # Cibles sur la RDM
    target_dots = ax_rdm.scatter([], [], edgecolors='white', facecolors='none', s=50, label='Cibles')
    ax_rdm.legend()

    ant_pos = Processor.ANTENNA_POS
    ax_map.scatter(ant_pos[:,0], ant_pos[:,1], marker='^', c='k', label='Antennes')
    ax_map.set_xlabel('X (m)')
    ax_map.set_ylabel('Y (m)')
    ax_map.legend()
    ax_map.set_xlim(-15, 15)
    ax_map.set_ylim(-1, 26)

    init_pos = np.array(Processor.compute_position(data_file, 0))
    vel_vec, _ = Processor.compute_speed(data_file, 0)
    vx, vy = vel_vec * dt
    point = ax_map.scatter(init_pos[0], init_pos[1], c='r', label='Cible')
    quiv = ax_map.quiver(
        init_pos[0], init_pos[1], vx, vy,
        angles='xy', pivot='tail', scale_units='xy', scale=1, color='r'
    )

    def update(frame_idx):
        rdm = Processor.compute_RDM(data_file, frame_idx)[1]
        im.set_data(rdm)
        im.set_clim(rdm.min(), rdm.max())
        cbar.update_normal(im)
        title_obj.set_text(f"RDM - frame {frame_idx+1}/{N_frame}")

        pos = np.array(Processor.compute_position(data_file, frame_idx))
        vel_vec, _ = Processor.compute_speed(data_file, frame_idx)
        vx, vy = vel_vec * dt
        point.set_offsets([pos])
        quiv.set_offsets([pos])
        quiv.set_UVC(vx, vy)

        # Détection de cibles multiples sur la RDM
        RDMs = Processor.compute_RDM(data_file, frame_idx)
        rdm_channel = RDMs[1]  # Channel 0 utilisé

        psf = Processor.psf_empirique_centered  # PSF expérimentale centrée
        raw_targets, _ = Processor.clean_rdm(rdm_channel, psf)
        grouped = Processor.fuse_targets(raw_targets)

        pts = targets_to_physical_coords(grouped, delta_r, vel_res, N_v)
        if len(pts) > 0:
            target_dots.set_offsets(pts)
        else:
            target_dots.set_offsets([])

        return im, point, quiv, title_obj, target_dots

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(N_frame),
        interval=40,
        blit=True,
        repeat=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()