import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Processor 

Offset = Processor.OFFSETS
Pos = Processor.ANTENNA_POS
idx = 1
data_file = "data/30-04/calibration 1.npz"
data, f0, B, Ms, Mc, Ts, Tc = Processor.load_file(data_file)
N_frame = data.shape[0]

PAD_R = Processor.PAD_R
PAD_D = Processor.PAD_D
sample_rdm = Processor.compute_RDM(data_file, 0)[idx]
N_v, N_r = sample_rdm.shape
delta_r = 3e8 / (2 * (B * PAD_R))
delta_v = (3e8 / (2 * f0)) * (1.0 / (Mc * Tc) / PAD_D)

def targets_to_physical_coords(targets, delta_r, delta_v, N_v):
    points = []
    for (d_idx, r_idx), _ in targets:
        v = (d_idx - N_v // 2) * delta_v
        r = r_idx * delta_r - Offset[idx]  # Offset pour le canal 0
        points.append((r, v))
    return np.array(points)


def plot_rdm_modes(data_file, frame_idx, psf, mode="multi_rdm"):
    RDMs = Processor.compute_RDM(data_file, frame_idx)
    fig, axs = None, None

    if mode == "multi_rdm":
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        for ch in range(4):
            rdm = RDMs[ch]
            raw_targets, _, _ = Processor.clean_rdm(rdm, psf, use_binary_mask=True)
            pts = targets_to_physical_coords(raw_targets, delta_r, delta_v, rdm.shape[0])
            doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
            ranges = np.arange(rdm.shape[1]) * delta_r - Processor.OFFSETS[ch]
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

            if len(pts) > 0:
                axs[ch].scatter(pts[:, 0], pts[:, 1], edgecolors='white', facecolors='none', s=50)

    elif mode == "clean_steps":
        ch = 0  # choisir le canal à afficher
        rdm = RDMs[ch]
        raw_targets, rdm_final, rdm_steps = Processor.clean_rdm(rdm, psf, use_binary_mask=True)
        doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
        ranges = np.arange(rdm.shape[1]) * delta_r - Processor.OFFSETS[ch]
        velocities = doppler_bins * delta_v

        # On ajoute la RDM brute au début
        rdm_steps_full = [rdm] + rdm_steps
        titles = ["RDM brute"] + [f"Itération {i+1}" for i in range(len(rdm_steps))]

        n_steps = len(rdm_steps_full)
        fig, axs = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4), sharex=True, sharey=True)

        # Trouver les bornes communes pour la colorbar
        global_min = min(np.min(r) for r in rdm_steps_full)
        global_max = max(np.max(r) for r in rdm_steps_full)

        im_list = []
        for i, rdm_step in enumerate(rdm_steps_full):
            im = axs[i].imshow(
                rdm_step,
                cmap='jet',
                origin='lower',
                extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
                aspect='auto',
                #vmin=global_min,
                vmax=global_max
            )
            axs[i].set_title(titles[i])
            axs[i].set_xlabel("Distance (m)")
            axs[i].set_ylabel("Vitesse (m/s)")
            im_list.append(im)

        # Colorbar commune
        #cbar = fig.colorbar(im_list[0], ax=axs.ravel().tolist(), shrink=0.9, label="Puissance")

        plt.tight_layout()
        plt.show()
        print("targets : ", targets_to_physical_coords(Processor.fuse_targets(raw_targets), delta_r, delta_v, rdm.shape[0]))

    else:
        raise ValueError(f"Mode inconnu : {mode}")

    plt.tight_layout()
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm  # Optional: progress bar

def generate_rdm_gif(data_file, psf, mode="multi_rdm", out_path="rdm_animation.gif"):
    data, *_ = Processor.load_file(data_file)
    N_frames = data.shape[0]

    temp_dir = "temp_rdm_frames"
    os.makedirs(temp_dir, exist_ok=True)
    filenames = []

    for frame_idx in tqdm(range(N_frames), desc="Generating frames"):
        fig = plt.figure(figsize=(12, 10))
        RDMs = Processor.compute_RDM(data_file, frame_idx)
        axs = fig.subplots(2, 2).flatten()

        for ch in range(4):
            rdm = RDMs[ch]
            raw_targets, _, _ = Processor.clean_rdm(rdm, psf, use_binary_mask=True)
            pts = targets_to_physical_coords(raw_targets, delta_r, delta_v, rdm.shape[0])
            doppler_bins = np.arange(rdm.shape[0]) - rdm.shape[0] // 2
            ranges = np.arange(rdm.shape[1]) * delta_r - Processor.OFFSETS[ch]
            velocities = doppler_bins * delta_v

            axs[ch].imshow(
                rdm,
                cmap='jet',
                origin='lower',
                extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]],
                aspect='auto'
            )
            axs[ch].set_title(f"Canal {ch} - Frame {frame_idx}")
            axs[ch].set_xlabel("Distance (m)")
            axs[ch].set_ylabel("Vitesse (m/s)")

            if len(pts) > 0:
                axs[ch].scatter(pts[:, 0], pts[:, 1], edgecolors='white', facecolors='none', s=50)

        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:03d}.png")
        filenames.append(frame_path)
        plt.savefig(frame_path)
        plt.close()

    # Generate GIF
    with imageio.get_writer(out_path, mode='I', duration=0.2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF saved to {out_path}")

    # Optional: cleanup
    for file in filenames:
        os.remove(file)
    os.rmdir(temp_dir)


def main() :
    data_file = "data/Mesures_18-04/doublemarche4-17memevit.npz"
    frame_to_plot = 80
    psf = Processor.psf_th

    plt.imshow(psf, cmap='jet', origin='lower', aspect='auto')
    plt.colorbar(label='PSF')
    plt.title("PSF théorique centrée")
    plt.show()

    # Mode 2 : étapes CLEAN sur une RDM
    plot_rdm_modes(data_file, frame_to_plot, psf, mode="clean_steps")

    # Mode 1 : 4 RDMs avec cibles
    # plot_rdm_modes(data_file, frame_to_plot, psf, mode="multi_rdm")
    """generate_rdm_gif(
    data_file="data/Mesures_30-04/démo 3 cibles.npz",
    psf=Processor.psf_empirique_centered,
    mode="multi_rdm",
    out_path="multi_rdm_targets.gif"
    )"""


if __name__ == '__main__':
    main()   


"""def main():
    data_file = "data/Mesures_30-04/démo 3 cibles.npz"
    data, f0, B, Ms, Mc, Ts, Tc = Processor.load_file(data_file)
    print(f"f0: {f0}, B: {B}, Ms: {Ms}, Mc: {Mc}, Ts: {Ts}, Tc: {Tc}")
    N_frame = data.shape[0]

    PAD_R = Processor.PAD_R
    PAD_D = Processor.PAD_D
    sample_rdm = Processor.compute_RDM(data_file, 0)[idx]
    N_v, N_r = sample_rdm.shape
    delta_r = 3e8 / (2 * (B * PAD_R))
    vel_res = (3e8 / (2 * f0)) * (1.0 / (Mc * Tc) / PAD_D)

    ranges = np.arange(N_r) * delta_r - Offset[idx]  # Offset pour le canal 1
    doppler_bins = np.arange(N_v) - N_v // 2 
    velocities = doppler_bins * vel_res

    fig, (ax_rdm, ax_map) = plt.subplots(1, 2, figsize=(12, 5))
    dt = 0.128 # Intervalle de temps entre les frames

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
        rdm = Processor.compute_RDM(data_file, frame_idx)[idx]
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
        rdm_channel = RDMs[idx]  # Channel 0 utilisé

        psf = Processor.psf_empirique_centered  # PSF expérimentale centrée
        raw_targets, _ = Processor.clean_rdm(rdm_channel, psf, use_binary_mask=True)
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
    main()"""