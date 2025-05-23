import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Processor
from tqdm import tqdm
import imageio.v2 as imageio
from itertools import cycle
import argparse

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
        r = r_idx * delta_r 
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
            ranges = np.arange(rdm.shape[1]) * delta_r
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

            # axs[ch].set_xticks(np.arange(rdm.shape[1]))
            # axs[ch].set_yticks(np.arange(rdm.shape[0]))
            # axs[ch].set_xticks(np.arange(-0.5, rdm.shape[1], 1), minor=True)
            # axs[ch].set_yticks(np.arange(-0.5, rdm.shape[0], 1), minor=True)
            # axs[ch].grid(which='minor', color='w', linestyle='-', linewidth=1)
            # axs[ch].tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
        
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
            ranges = np.arange(rdm.shape[1]) * delta_r
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

# ────────────────────── affichage multicible ─────────────────────

def plot_multi_target_rdms(data_file, save_path = None, 
                           anim = False, frame_idx = 0):
    
    """Visualise les RDM (4 canaux) et les cibles CFAR.

    Si *anim* est **True**, crée un ``FuncAnimation`` sur toutes les frames.
    Sinon, affiche/sauvegarde la frame ``frame_idx``.
    """

    # ─── infos fichier & résolutions ─────────────────────────────────────────
    data, f0, B, Ms, Mc, Ts, Tc, N_frame, N_v, N_r, delta_r, delta_v = load_data_info(data_file)
    n_doppler, n_range = Processor.compute_RDM(data_file, 0)[0].shape

    # ─── helper interne pour un affichage d'une frame ───────────────────────
    def _draw(axs, frame: int):
        print(frame)
        RDMs = Processor.compute_RDM(data_file, frame)
        for ch, ax in enumerate(axs):
            rdm = RDMs[ch]
            targets, mask, _ = Processor.cfar_2d_adaptive(
                rdm
            )
            phys_pts = targets_to_physical_coords(targets, delta_r, delta_v, n_doppler, ch)

            doppler_bins = np.arange(n_doppler) - n_doppler // 2
            ranges_m = np.arange(n_range) * delta_r 
            vels_mps = doppler_bins * delta_v

            im = ax.images[0]
            im.set_data(rdm)
            im.set_clim(vmin=rdm.min(), vmax=rdm.max())
            ax.set_title(f"Canal {ch} – {len(phys_pts)} cible(s) – Frame {frame+1}/{N_frame}")

            scat = ax.collections[0]
            if phys_pts.size:
                scat.set_offsets(phys_pts)
            else:
                scat.set_offsets(np.empty((0, 2)))

    # ─── création figure de base ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ch, ax in enumerate(axes):
        # placeholder, sera mis à jour par _draw
        dummy = np.zeros((n_doppler, n_range))
        doppler_bins = np.arange(n_doppler) - n_doppler // 2
        ranges_m = np.arange(n_range) * delta_r 
        vels_mps = doppler_bins * delta_v
        im = ax.imshow(
            dummy,
            cmap="jet",
            origin="lower",
            extent=[ranges_m[0], ranges_m[-1], vels_mps[0], vels_mps[-1]],
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
        scat = ax.scatter([], [], edgecolors="white", facecolors="none", s=60, linewidths=1.5)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Vitesse (m/s)")

    # ─── mode animation ou image fixe ───────────────────────────────────────
    if anim:
        import matplotlib.animation as animation

        def _update(frame):
            _draw(axes, frame)
            return [child for ax in axes for child in (ax.images + ax.collections)]

        ani = animation.FuncAnimation(
            fig, _update, frames=range(N_frame), interval=200, blit=True, repeat=True
        )
        plt.tight_layout()
        if save_path:
            ani.save(save_path, writer="pillow", fps=5)
            print(f"Animation sauvegardée → {save_path}")
        else:
            plt.show()
    else:
        frame_idx = np.clip(frame_idx, 0, N_frame - 1)
        _draw(axes, frame_idx)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Figure sauvegardée → {save_path}")
        else:
            plt.show()

def plot_multi_target_rdmsv2(data_file, save_path = None, 
                           anim = False, frame_idx = 0):
    
    """Visualise les RDM (4 canaux) et les cibles CFAR.

    Si *anim* est **True**, crée un ``FuncAnimation`` sur toutes les frames.
    Sinon, affiche/sauvegarde la frame ``frame_idx``.
    """

    # ─── infos fichier & résolutions ─────────────────────────────────────────
    data, f0, B, Ms, Mc, Ts, Tc, N_frame, N_v, N_r, delta_r, delta_v = load_data_info(data_file)
    n_doppler, n_range = Processor.compute_RDM(data_file, 0)[0].shape

    # ─── helper interne pour un affichage d'une frame ───────────────────────
    def _draw(axs, frame: int):
        # print(frame)
        RDMs = Processor.compute_RDM(data_file, frame)
        for ch, ax in enumerate(axs):
            rdm = RDMs[ch]
            targets, mask, _ = Processor.cfar_2d(
                rdm
            )
            phys_pts = targets_to_physical_coords(targets, delta_r, delta_v, n_doppler, ch)

            doppler_bins = np.arange(n_doppler) - n_doppler // 2
            ranges_m = np.arange(n_range) * delta_r
            vels_mps = doppler_bins * delta_v

            im = ax.images[0]
            im.set_data(rdm)
            im.set_clim(vmin=rdm.min(), vmax=rdm.max())
            ax.set_title(f"Canal {ch} – {len(phys_pts)} cible(s) – Frame {frame+1}/{N_frame}")

            scat = ax.collections[0]
            if phys_pts.size:
                scat.set_offsets(phys_pts)
            else:
                scat.set_offsets(np.empty((0, 2)))
            

    # ─── création figure de base ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ch, ax in enumerate(axes):
        # placeholder, sera mis à jour par _draw
        dummy = np.zeros((n_doppler, n_range))
        doppler_bins = np.arange(n_doppler) - n_doppler // 2
        ranges_m = np.arange(n_range) * delta_r 
        vels_mps = doppler_bins * delta_v
        im = ax.imshow(
            dummy,
            cmap="jet",
            origin="lower",
            extent=[ranges_m[0], ranges_m[-1], vels_mps[0], vels_mps[-1]],
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
        scat = ax.scatter([], [], edgecolors="white", facecolors="none", s=60, linewidths=1.5)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Vitesse (m/s)")

    # ─── mode animation ou image fixe ───────────────────────────────────────
    if anim:
        import matplotlib.animation as animation

        def _update(frame):
            _draw(axes, frame)
            return [child for ax in axes for child in (ax.images + ax.collections)]

        ani = animation.FuncAnimation(
            fig, _update, frames=range(N_frame), interval=200, blit=True, repeat=True
        )
        plt.tight_layout()
        if save_path:
            ani.save(save_path, writer="pillow", fps=5)
            print(f"Animation sauvegardée → {save_path}")
        else:
            plt.show()
    else:
        frame_idx = np.clip(frame_idx, 0, N_frame - 1)
        _draw(axes, frame_idx)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Figure sauvegardée → {save_path}")
        else:
            plt.show()

def transform_coordinates(x, y, data_file):
    """
    Transforme les coordonnées (x,y) en inversant la composante x si le fichier
    est dans le dossier data/30-04. Cette transformation s'applique aussi aux positions
    des antennes.
    """
    if "data/30-04" in data_file:
        return -x, y
    return x, y

def get_antenna_positions(data_file):
    """
    Retourne les positions des antennes, en appliquant la transformation de coordonnées
    si nécessaire (pour data/30-04).
    """
    if "data/30-04" in data_file:
        # Inverser les positions x des antennes
        ant_pos = Processor.ANTENNA_POS.copy()
        ant_pos[:, 0] = -ant_pos[:, 0]
        return ant_pos
    return Processor.ANTENNA_POS

def plot_target_trajectory_with_kalman(
    data_file,
    kalman_params=None,
    save_path=None,
):
    """
    Anime la position et la vitesse de la cible dans le plan (x, y) ;
    - Les états mesurés sont affichés en gris semi-transparent
    - La trace rouge affiche **toute** la trajectoire prédit / lissée par le filtre Kalman avec 
      des mini-flèches indiquant le sens du parcours
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
    plt.grid(True)

    # antennes
    ant_pos = get_antenna_positions(data_file)
    ax.scatter(ant_pos[:, 0], ant_pos[:, 1], marker="^", c="k", label="Antennes")

    # mesures et Kalman – points + vecteurs vitesse
    first_pos = np.array(Processor.compute_position(data_file, 0))
    first_vel_vec, _ = Processor.compute_speed(data_file, 0)
    vx0, vy0 = first_vel_vec * dt

    # Transformer les coordonnées
    first_pos_x, first_pos_y = transform_coordinates(first_pos[0], first_pos[1], data_file)
    kalman_x, kalman_y = transform_coordinates(kalman_params["kalman_x"][0], kalman_params["kalman_x"][1], data_file)

    meas_point = ax.scatter(first_pos_x, first_pos_y, c="b", s=50, label="Mesure")
    kal_point = ax.scatter(
        kalman_x,
        kalman_y,
        c="r",
        s=50,
        label="Kalman",
    )

    meas_quiv = ax.quiver(
        first_pos_x,
        first_pos_y,
        vx0,
        vy0,
        angles="xy",
        pivot="tail",
        scale_units="xy",
        scale=1,
        color="b",
    )

    kal_quiv = ax.quiver(
        kalman_x,
        kalman_y,
        kalman_params["kalman_x"][2] * dt,
        kalman_params["kalman_x"][3] * dt,
        angles="xy",
        pivot="tail",
        scale_units="xy",
        scale=1,
        color="r",
    )

    # ───────────── ajout d'une ligne pour tracer la trajectoire Kalman et les mesures ─────────
    traj_x, traj_y = [kalman_x], [kalman_y]
    (traj_line,) = ax.plot(traj_x, traj_y, "r-", linewidth=1.5, label="Trajectoire Kalman")
    
    # Nouvelle liste pour stocker toutes les mesures
    meas_traj_x, meas_traj_y = [first_pos_x], [first_pos_y]
    meas_traj_scatter = ax.scatter(meas_traj_x, meas_traj_y, c='gray', alpha=0.3, s=20, label="Historique mesures")
    
    # Collection de flèches pour indiquer le sens du parcours
    direction_arrows = []

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

        # Transformer les coordonnées
        meas_pos_x, meas_pos_y = transform_coordinates(meas_pos[0], meas_pos[1], data_file)

        meas_point.set_offsets([meas_pos_x, meas_pos_y])
        meas_quiv.set_offsets([meas_pos_x, meas_pos_y])
        meas_quiv.set_UVC(vx, vy)
        
        # Ajouter la mesure à l'historique
        meas_traj_x.append(meas_pos_x)
        meas_traj_y.append(meas_pos_y)
        meas_traj_scatter.set_offsets(np.column_stack((meas_traj_x, meas_traj_y)))

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

        # Transformer les coordonnées Kalman
        kal_pos_x, kal_pos_y = transform_coordinates(kal_pos[0], kal_pos[1], data_file)

        kal_point.set_offsets([kal_pos_x, kal_pos_y])
        kal_quiv.set_offsets([kal_pos_x, kal_pos_y])
        kal_quiv.set_UVC(kal_vel[0], kal_vel[1])

        # 3) mise à jour de la trajectoire Kalman
        if len(traj_x) > 0:
            # Ajouter une mini-flèche pour montrer la direction entre le dernier point et le nouveau
            last_x, last_y = traj_x[-1], traj_y[-1]
            
            # Calculer le milieu du segment pour placer la flèche
            mid_x = (last_x + kal_pos_x) / 2
            mid_y = (last_y + kal_pos_y) / 2
            
            # Calculer le vecteur direction
            dx = kal_pos_x - last_x
            dy = kal_pos_y - last_y
            
            # Normaliser et réduire la taille de la flèche
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:  # Éviter division par zéro
                dx = dx / norm * 0.1  # Facteur d'échelle pour la taille de la flèche
                dy = dy / norm * 0.1
                
                # Créer une petite flèche de direction
                direction_arrow = ax.quiver(
                    mid_x, mid_y, dx, dy,
                    angles='xy', pivot='middle', scale_units='xy',
                    scale=0.3, color='r', width=0.003, headwidth=5, headlength=6
                )
                direction_arrows.append(direction_arrow)
        
        traj_x.append(kal_pos_x)
        traj_y.append(kal_pos_y)
        traj_line.set_data(traj_x, traj_y)

        # 4) titre
        title.set_text(f"Position de la cible – Frame {frame_idx+1}/{N_frame}")

        return_objects = [
            meas_point,
            meas_quiv,
            kal_point,
            kal_quiv,
            traj_line,
            meas_traj_scatter,
            title,
        ]
        
        # Ajouter les flèches de direction à la liste des objets à retourner
        if direction_arrows:
            return_objects.extend(direction_arrows)
            
        return tuple(return_objects)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(N_frame),
        interval=50,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer="pillow", fps=5)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

def run_full_tracking_and_plot(data_file, save_plot_path=None, dist_threshold=2.0, angle_threshold=15.0, overlap_threshold=0.6):
    """
    • Lance le suivi multicible sur TOUTES les frames du fichier.
    • À la fin, affiche (ou enregistre) les trajectoires (x,y) des trackers
      passés en liste `retired`, chaque cible de couleur différente.
    • Ajoute des petites flèches entre chaque point pour indiquer la direction.
    """
    Processor.NEXT_ID = 0  # reset ID counter for new run

    # 0) ————————————————————————————————————————————————————————————————
    #    petit nettoyage si on enchaîne plusieurs runs dans la même session
    Processor.non_official.clear()
    Processor.official.clear()
    Processor.retired.clear()

    # 1) ————————————————————————— initialisation ——————————————————————————
    data, *_ = Processor.load_file(data_file)
    N_frame   = data.shape[0]

    Processor.tracking_init(data_file)          # remplit Processor.non_official
    print(f"Initialisation des trackers : {len(Processor.non_official)} trackers non officiels créés.")
    for trk in Processor.non_official:
        print(trk)
    print(f"Initialisation des trackers : {len(Processor.official)} trackers officiels créés.")
    for trk in Processor.official:
        print(trk)

    # 2) ———————————————————————— boucle sur les frames ————————————————————
    for k in range(1, N_frame):
        print(f"Frame {k}/{N_frame}")
        Processor.tracking_update(
            Processor.non_official,
            k,
            data_file,
            Processor.official
        )
        print(f"number of nonoff trackers after frame {k} = {len(Processor.non_official)}")
        for trk in Processor.non_official:
            print(trk)
        print(f"number of off trackers after frame {k} = {len(Processor.official)}")
        print(f"number of retired trackers after frame {k} = {len(Processor.retired)}")

    # 3) ———————————————————————— finalisation ————————————————————————
    Processor.tracking_finalize(Processor.official)   # pousse tout dans retired
    
    # 3.5) ——————————————————————— clusterisation ———————————————————————
    print("Clusterisation des trajectoires...")
    n_before = len(Processor.retired)
    Processor.cluster_retired_trackers(distance_threshold=dist_threshold, angle_threshold=angle_threshold, time_overlap_threshold=overlap_threshold)
    n_after = len(Processor.retired)
    print(f"Clusterisation terminée : {n_before} trajectoires -> {n_after} trajectoires")

    # 4) ———————————————————————— tracé des trajectoires ————————————————————
    retired_trackers = Processor.retired
    if not retired_trackers:
        print("Aucun tracker retiré : rien à tracer.")
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    color_cycle = cycle(plt.cm.tab10.colors)

    for trk, col in zip(retired_trackers, color_cycle):
        hist = trk.history
        traj = [state[0] for state in hist]     # positions (x,y)
        # Transformer les coordonnées
        traj = [(transform_coordinates(x, y, data_file)) for x, y in traj]
        xs, ys = zip(*traj)
        
        # Tracer la trajectoire
        ax.plot(xs, ys, marker='o', ms=3, lw=1.3,
                color=col, label=f"Track {trk.id}")
        
        # Calculer et tracer les vecteurs de vitesse entre chaque point
        for i in range(len(traj)-1):
            # Calculer le vecteur de vitesse entre deux points consécutifs
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            
            # Normaliser le vecteur pour une longueur constante
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 0:  # Éviter la division par zéro
                dx = dx / norm * 0.1  # 0.1 est la longueur de la flèche
                dy = dy / norm * 0.1
            
            # Tracer la flèche
            ax.quiver(xs[i], ys[i], dx, dy,
                     color=col, scale=1, scale_units='inches',
                     width=0.005, headwidth=2, headlength=2, headaxislength=2)

    # (optionnel) affichage des antennes
    ant = get_antenna_positions(data_file)
    ax.scatter(ant[:,0], ant[:,1], marker='^', c='k', s=60, label="Antennes")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect('auto')
    ax.set_title("Trajectoires des cibles (trackers retirés)")
    ax.legend()
    ax.grid(True)

    if save_plot_path:
        fig.savefig(save_plot_path, dpi=150)
        print(f"Figure sauvegardée → {save_plot_path}")
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_cfar(data_file, frame_idx=0, channel=0, save_path=None):
    """
    Affiche côte-à-côte :
     - à gauche : seuils CFAR (cmap='viridis')
     - à droite : RDM brute (cmap='jet') + contour des zones rdm>seuil
    Pour une frame et un canal donnés.
    """
    # 1) Récupération RDM + CFAR
    rdm = Processor.compute_RDM(data_file, frame_idx)[channel]
    mask, thresholds = Processor.ca_cfar_convolve(rdm)
    # mask est un booléen où True = dépassement du seuil

    # 2) Coordonnées physiques
    n_dop, n_rng = rdm.shape
    dop_bins = np.arange(n_dop) - n_dop//2
    ranges = np.arange(n_rng)*Processor.delta_r
    vels   = dop_bins * Processor.delta_v

    # 3) Création de la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    # --- 4) Sous-plot 1 : seuil CFAR ---
    pcm1 = ax1.pcolormesh(
        ranges, vels, thresholds,
        cmap='jet', shading='auto'
    )
    fig.colorbar(pcm1, ax=ax1, shrink=0.8, label="Seuil CFAR")
    ax1.set_title(f"Seuil CFAR\ncanal {channel}, frame {frame_idx}")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Vitesse (m/s)")
    ax1.grid(False)

    # --- 5) Sous-plot 2 : RDM + contours du mask ---
    im2 = ax2.imshow(
        rdm,
        origin='lower',
        extent=[ranges[0], ranges[-1], vels[0], vels[-1]],
        aspect='auto',
        cmap='jet'
    )
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Puissance RDM")
    # tracer le contour de mask (rdm>seuil)
    cs = ax2.contour(
        ranges, vels, mask.astype(int),
        levels=[0.5],          # la frontière True/False
        colors='white',
        linewidths=1.4,
        linestyles='-'
    )
    ax2.set_title("RDM brute + zones > seuil CFAR")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Vitesse (m/s)")
    ax2.grid(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure sauvegardée → {save_path}")
    else:
        plt.show()

def animate_tracker_evolution(data_file, save_path=None, dist_threshold=2.0, angle_threshold=15.0, overlap_threshold=0.5):
    """
    Effectue le tracking complet et la clusterisation, puis anime l'évolution frame 
    par frame des trackers pour montrer comment ils se déplacent dans le temps.
    
    Args:
        data_file: Chemin vers le fichier de données
        save_path: Chemin pour sauvegarder l'animation
        dist_threshold: Seuil de distance pour clustering
        angle_threshold: Seuil d'angle pour clustering
        overlap_threshold: Seuil de chevauchement pour clustering
    """
    # 0) ————————————————————————————————————————————————————————————————
    #    Nettoyage et initialisation
    Processor.NEXT_ID = 0
    Processor.non_official.clear()
    Processor.official.clear()
    Processor.retired.clear()

    # 1) ————————————————————————— Exécution du tracking complet ——————————————————————————
    data, *_ = Processor.load_file(data_file)
    N_frame = data.shape[0]
    
    print(f"Exécution du tracking sur {N_frame} frames...")
    
    # Initialiser le tracking
    Processor.tracking_init(data_file)
    print(f"Initialisation: {len(Processor.non_official)} trackers non officiels, {len(Processor.official)} trackers officiels")
    
    # Suivre sur toutes les frames
    for k in range(1, N_frame):
        Processor.tracking_update(
            Processor.non_official,
            k,
            data_file,
            Processor.official
        )
    
    # Finaliser le tracking
    Processor.tracking_finalize(Processor.official)
    
    # Clusteriser les trackers
    print("Clusterisation des trajectoires...")
    n_before = len(Processor.retired)
    Processor.cluster_retired_trackers(
        distance_threshold=dist_threshold, 
        angle_threshold=angle_threshold,
        time_overlap_threshold=overlap_threshold
    )
    n_after = len(Processor.retired)
    print(f"Clusterisation terminée: {n_before} → {n_after} trajectoires")
    
    # Si aucun tracker, rien à animer
    if not Processor.retired:
        print("Aucun tracker à animer!")
        return
    
    # 2) ————————————————————————— Créer l'animation ——————————————————————————
    retired_trackers = Processor.retired
    
    # Déterminer la plage de frames couverte par tous les trackers
    min_frame = min(trk.frame_start for trk in retired_trackers)
    max_frame = max(trk.frame_start + len(trk.history) - 1 for trk in retired_trackers)
    animation_length = max_frame - min_frame + 1
    
    print(f"Animation sur {animation_length} frames (de {min_frame} à {max_frame})")
    
    # Créer la figure et les axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Afficher les antennes
    ant = get_antenna_positions(data_file)
    ax.scatter(ant[:,0], ant[:,1], marker='^', c='k', s=60, label="Antennes")
    
    # Préparer les couleurs pour chaque tracker
    color_cycle = cycle(plt.cm.tab10.colors)
    colors = {trk.id: next(color_cycle) for trk in retired_trackers}
    
    # Préparer des collections vides pour chaque tracker
    trajectories = {}  # Stocke toutes les lignes de trajectoire
    positions = {}     # Stocke les marqueurs de position actuelle
    endpoints = {}     # Stocke les marqueurs de fin de trajectoire (croix)
    
    for trk in retired_trackers:
        # Trajectoire complète (sera mise à jour pendant l'animation)
        line, = ax.plot([], [], '-', lw=1.5, color=colors[trk.id], alpha=0.7)
        trajectories[trk.id] = line
        
        # Position actuelle (cercle)
        scatter = ax.scatter([], [], marker='o', s=80, color=colors[trk.id], 
                          edgecolor='w', linewidth=1.5, zorder=10, label=f"Tracker {trk.id}")
        positions[trk.id] = scatter
        
        # Position finale (croix) - initialement invisible
        endmarker = ax.scatter([], [], marker='x', s=100, color=colors[trk.id],
                            linewidth=2, zorder=11)
        endmarker.set_alpha(0)  # Initialement invisible
        endpoints[trk.id] = endmarker
    
    # Légende (affichée une seule fois avec tous les trackers)
    ax.legend(loc='upper right')
    
    # Titre avec numéro de frame (sera mis à jour)
    title = ax.set_title(f"Frame: {min_frame}/{max_frame}")
    
    # Configuration des axes
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(-15, 15)  # Ajuster selon vos données
    ax.set_ylim(-5, 26)   # Ajuster selon vos données
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Fonction d'animation
    def update(frame_idx):
        # Convertir l'indice d'animation en numéro de frame réel
        real_frame = min_frame + frame_idx
        
        # Mettre à jour le titre
        title.set_text(f"Frame: {real_frame}/{max_frame}")
        
        # Liste des éléments à retourner pour le blitting
        artists = [title]
        
        # Mettre à jour chaque tracker
        for trk in retired_trackers:
            trk_start = trk.frame_start
            trk_end = trk_start + len(trk.history) - 1
            
            # Si le tracker est déjà apparu à cette frame ou avant
            if real_frame >= trk_start:
                # Si le tracker est actif à cette frame
                if real_frame <= trk_end:
                    # Indice dans l'historique du tracker
                    hist_idx = real_frame - trk_start
                
                    # Extraire toutes les positions jusqu'à la frame actuelle
                    positions_up_to_now = [state[0] for state in trk.history[:hist_idx+1]]
                    # Transformer les coordonnées
                    positions_up_to_now = [(transform_coordinates(x, y, data_file)) for x, y in positions_up_to_now]
                    xs, ys = zip(*positions_up_to_now) if positions_up_to_now else ([], [])
                
                    # Mettre à jour la trajectoire
                    trajectories[trk.id].set_data(xs, ys)
                    artists.append(trajectories[trk.id])
                
                    # Mettre à jour la position actuelle (cercle)
                    if positions_up_to_now:
                        curr_pos = np.array([positions_up_to_now[-1]])
                        positions[trk.id].set_offsets(curr_pos)
                        positions[trk.id].set_alpha(1.0)  # Visible
                        artists.append(positions[trk.id])
                    
                    # S'assurer que le marqueur de fin est invisible
                    endpoints[trk.id].set_alpha(0)
                    artists.append(endpoints[trk.id])
                        
                # Si le tracker n'est plus actif mais a existé avant
                else:
                    # On garde la trajectoire complète
                    full_trajectory = [state[0] for state in trk.history]
                    # Transformer les coordonnées
                    full_trajectory = [(transform_coordinates(x, y, data_file)) for x, y in full_trajectory]
                    xs, ys = zip(*full_trajectory) if full_trajectory else ([], [])
                    trajectories[trk.id].set_data(xs, ys)
                    artists.append(trajectories[trk.id])
                    
                    # On garde la dernière position connue, mais avec une croix
                    if full_trajectory:
                        last_pos = np.array([full_trajectory[-1]])
                        
                        # Rendre invisible le cercle
                        positions[trk.id].set_alpha(0)
                        artists.append(positions[trk.id])
                        
                        # Afficher la croix à la position finale
                        endpoints[trk.id].set_offsets(last_pos)
                        endpoints[trk.id].set_alpha(1.0)  # Visible
                        artists.append(endpoints[trk.id])
            
            # Si le tracker n'existe pas encore à cette frame
            else:
                # Masquer la trajectoire et tous les marqueurs
                trajectories[trk.id].set_data([], [])
                positions[trk.id].set_alpha(0.0)  # Invisible
                endpoints[trk.id].set_alpha(0.0)  # Invisible
                artists.append(trajectories[trk.id])
                artists.append(positions[trk.id])
                artists.append(endpoints[trk.id])
        
        return artists
    
    # Créer l'animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=animation_length,
        interval=200,  # 200ms par frame
        blit=True, 
        repeat=True
    )
    
    # Sauvegarder ou afficher
    plt.tight_layout()
    if save_path:
        print(f"Sauvegarde de l'animation vers {save_path}...")
        ani.save(save_path, writer='pillow', fps=5)
        print(f"Animation sauvegardée!")
    else:
        plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualisation des données radar')
    
    # Required arguments
    parser.add_argument('data_file', type=str, nargs='?', 
                      default="data/30-04/démo1 monocible.npz",
                      help='Chemin vers le fichier de données (.npz)')
    
    # Optional arguments
    parser.add_argument('--mode', type=str, default='basic_rdm',
                      choices=['basic_rdm', 'multi_target', 'multi_targetv2', 
                              'trajectory_kalman', 'full_tracking', 'plot_cfar',
                              'animate_trackers'],
                      help='Mode de visualisation')
    parser.add_argument('--anim', action='store_true',
                      help='Activer l\'animation (si disponible pour le mode)')
    parser.add_argument('--save', type=str, default=None,
                      help='Chemin pour sauvegarder la visualisation')
    parser.add_argument('--frame', type=int, default=0,
                      help='Index de la frame à afficher (pour les modes non-animés)')
    parser.add_argument('--channel', type=int, default=0,
                      help='Canal à afficher (pour plot_cfar)')
    
    # Clustering parameters
    parser.add_argument('--dist-threshold', type=float, default=0.07,
                      help='Seuil de distance (m) pour le clustering de trajectoires')
    parser.add_argument('--angle-threshold', type=float, default=5.0,
                      help='Seuil d\'angle (degrés) pour le clustering de trajectoires')
    parser.add_argument('--overlap-threshold', type=float, default=0.6,
                      help='Seuil de chevauchement temporel (0-1) pour le clustering de trajectoires')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\nConfiguration:")
    print(f"Fichier de données : {args.data_file}")
    print(f"Mode de visualisation : {args.mode}")
    print(f"Animation : {'Activée' if args.anim else 'Désactivée'}")
    if args.save:
        print(f"Sauvegarde : {args.save}")
    if args.mode in ['multi_targetv2', 'plot_cfar']:
        print(f"Frame : {args.frame}")
    if args.mode == 'plot_cfar':
        print(f"Canal : {args.channel}")
    if args.mode in ['full_tracking', 'animate_trackers']:
        print(f"Paramètres de clustering:")
        print(f"  - Seuil de distance : {args.dist_threshold} m")
        print(f"  - Seuil d'angle : {args.angle_threshold}°")
        print(f"  - Seuil de chevauchement : {args.overlap_threshold}")
    print()
    
    # Set global variables based on data file path
    global Offset, Pos
    if "data/18-04" in args.data_file:
        Offset = Processor.OFFSETS_2
        Pos = Processor.ANTENNA_POS_2
    else:
        Offset = Processor.OFFSETS
        Pos = Processor.ANTENNA_POS
    
    # Kalman parameters (if needed)
    kalman_params = {
        'kalman_x': np.array([0, 2, 0, 0.5]),  # Initial state [x, y, vx, vy]
        'kalman_p': np.eye(4) * 1,          # Initial covariance
        'outlier_radius': 100.0             # Outlier radius
    }
    
    # Choose visualization based on mode
    if args.mode == "basic_rdm":
        plot_basic_rdms(args.data_file, args.anim, args.save)
    
    elif args.mode == "multi_target":
        plot_multi_target_rdms(args.data_file, anim=args.anim, save_path=args.save)

    elif args.mode == "multi_targetv2":
        plot_multi_target_rdmsv2(args.data_file, anim=args.anim, 
                                save_path=args.save, frame_idx=args.frame)
    
    elif args.mode == "trajectory_kalman":
        plot_target_trajectory_with_kalman(args.data_file, kalman_params, args.save)
    
    elif args.mode == "full_tracking":
        run_full_tracking_and_plot(args.data_file, save_plot_path=args.save,
                                  dist_threshold=args.dist_threshold,
                                  angle_threshold=args.angle_threshold,
                                  overlap_threshold=args.overlap_threshold)
    
    elif args.mode == "plot_cfar":
        plot_cfar(args.data_file, frame_idx=args.frame, 
                 channel=args.channel, save_path=args.save)
                 
    elif args.mode == "animate_trackers":
        animate_tracker_evolution(args.data_file, save_path=args.save,
                                 dist_threshold=args.dist_threshold,
                                 angle_threshold=args.angle_threshold,
                                 overlap_threshold=args.overlap_threshold)
    
    else:
        print(f"Mode de visualisation inconnu: {args.mode}")
        print("Options valides: basic_rdm, multi_target, multi_targetv2, trajectory_kalman, full_tracking, plot_cfar, animate_trackers")

if __name__ == "__main__":
    main()