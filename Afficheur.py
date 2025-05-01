import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Processor 
import imageio.v2 as imageio
from tqdm import tqdm  # Optional: progress bar

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
    data_file = "data/30-04/marche 2-15m.npz"
    frame_to_plot = 0
    psf = Processor.psf_th
    sol = Processor.compute_position(data_file,frame_to_plot)
    print("Position : ", sol)


if __name__ == '__main__':
    main()   
