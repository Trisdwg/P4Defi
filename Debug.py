import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import Processor
import Afficheur
from pathlib import Path
import sys
import logging
from itertools import cycle

def setup_logger(data_file):
    """Configure le logger pour écrire dans un fichier avec horodatage."""
    # Créer un nom de fichier basé sur le nom du fichier de données et l'horodatage
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.basename(data_file).split('.')[0]
    log_dir = "logs"
    
    # Créer le répertoire logs s'il n'existe pas
    Path(log_dir).mkdir(exist_ok=True)
    
    log_filename = f"{log_dir}/{base_filename}_{timestamp}.log"
    
    # Configurer le logger
    logger = logging.getLogger('tracking_debug')
    logger.setLevel(logging.INFO)
    
    # Gestionnaire de fichiers
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Formateur
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Ajouter le gestionnaire au logger
    logger.addHandler(file_handler)
    
    return logger, log_filename

class StdoutRedirector:
    """Classe pour rediriger stdout vers un logger."""
    def __init__(self, logger):
        self.logger = logger
        self.terminal = sys.stdout
        
    def write(self, message):
        if message.strip():  # Éviter d'enregistrer les lignes vides
            self.logger.info(message.strip())
        self.terminal.write(message)  # Conserver l'affichage dans le terminal
        
    def flush(self):
        # Nécessaire pour être un substitut complet pour sys.stdout
        self.terminal.flush()

def plot_trackers(fig, ax, non_official, official, retired, frame_idx):
    """Affiche les trajectoires des trackers sur une figure."""
    # À implémenter: affichage des trajectoires des trackers
    color_cycle = cycle(plt.cm.tab10.colors)

    for trk, col in zip(official, color_cycle):
        hist = trk.history
        traj = [state[0] for state in hist]     # positions (x,y)
        xs, ys = zip(*traj)
        ax.plot(xs, ys, marker='o', ms=3, lw=1.3,
                color=col, label=f"Track {trk.id}")

    # (optionnel) affichage des antennes
    ant = Processor.ANTENNA_POS
    ax.scatter(ant[:,0], ant[:,1], marker='^', c='k', s=60, label="Antennes")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect('equal')
    ax.set_title("Trajectoires des cibles (trackers retirés), Frame " + str(frame_idx))
    ax.grid(True)
    plt.tight_layout()

    plt.show()

def display_rdms(data_file, frame_idx):
    """Affiche les RDMs avec les cibles détectées par CFAR."""
    # À implémenter: affichage des 4 RDMs avec les cibles détectées
    Afficheur.plot_multi_target_rdmsv2(data_file, frame_idx=frame_idx)

def extract_measures(data_file, frame_idx):
    """Extrait et retourne les mesures possibles à cette frame."""
    # À implémenter: extraction des mesures via la combinatoire
    RDM     = Processor.compute_RDM(data_file, frame_idx)
    tracks  = Processor.make_intraframe_tracks(Processor.extract_all_targets(RDM))
    return tracks

def debug_full_tracking_and_plot(data_file):
    """
    Lance le suivi multicible de manière interactive, frame par frame.

    Commandes disponibles:
    - next: passe à la prochaine frame
    - lists: affiche les listes de trackers
    - print tracker <ID>: affiche les détails d'un tracker
    - show rdm: affiche les RDMs de la frame actuelle
    - measures: affiche les mesures possibles via la combinatoire
    - goto FRAME_IDX: saute à la frame spécifiée
    - run: continue l'exécution sans pause
    """
    # Redirection de stdout
    original_stdout = sys.stdout

    try:
        # Réinitialiser les compteurs et listes
        Processor.NEXT_ID = 0
        Processor.non_official.clear()
        Processor.official.clear()
        Processor.retired.clear()

        # Charger les données
        data = Processor.load_file(data_file)[0]
        N_frame = data.shape[0]

        print(f"Démarrage du tracking debug sur {data_file}")
        print(f"Fichier contient {N_frame} frames")

        # Initialiser les trackers sur la frame 0
        Processor.tracking_init(data_file)
        print(f"Initialisation des trackers : {len(Processor.non_official)} trackers non officiels créés.")

        # Afficher les RDMs initiaux
        display_rdms(data_file, 0)
        print("Trackers initialisés. Entrez 'run' pour continuer.")

        # Configuration de la figure pour le tracking
        plt.ion()  # Mode interactif
        fig, ax = plt.subplots(figsize=(6, 8))
        plot_trackers(fig, ax, Processor.non_official, Processor.official, Processor.retired, 0)
        # plt.show(block=False)

        # Boucle interactive
        frame_idx = 0
        run_mode = False
        # plt.close(fig)  # Ferme la figure précédente proprement
        # fig, ax = plt.subplots(figsize=(6, 8))

        while frame_idx < N_frame - 1:
            if not run_mode:
                cmd = input("Commande (next, lists, print tracker ID, show rdm, measures, goto FRAME_IDX, run): ")

                if cmd.strip() == "next":
                    frame_idx += 1
                    print(f"Passage à la frame {frame_idx}")

                elif cmd.strip() == "lists":
                    print("Liste des trackers:")
                    print(f"Non-officiels: {[t.id for t in Processor.non_official]}")
                    print(f"Officiels: {[t.id for t in Processor.official]}")
                    print(f"Retirés: {[t.id for t in Processor.retired]}")
                    continue

                elif cmd.strip().startswith("print tracker"):
                    try:
                        tracker_id = int(cmd.split("print tracker")[1].strip())
                        found = False
                        for trk_list in [Processor.non_official, Processor.official, Processor.retired]:
                            for trk in trk_list:
                                if trk.id == tracker_id:
                                    print(f"Détails du tracker {tracker_id}:")
                                    print(str(trk))
                                    found = True
                                    break
                        if not found:
                            print(f"Tracker ID {tracker_id} non trouvé.")
                    except ValueError:
                        print("ID de tracker invalide. Utilisation: print tracker <ID>")
                    continue

                elif cmd.strip() == "show rdm":
                    display_rdms(data_file, frame_idx)
                    continue

                elif cmd.strip() == "measures":
                    measures = extract_measures(data_file, frame_idx)
                    print(f"Mesures disponibles à la frame {frame_idx}:")
                    print(str(measures))
                    continue

                elif cmd.strip().startswith("goto"):
                    try:
                        target_frame = int(cmd.split("goto")[1].strip())
                        if 0 <= target_frame < N_frame:
                            while frame_idx < target_frame:
                                frame_idx += 1
                                print(f"Passage à la frame {frame_idx}")
                                Processor.tracking_update(
                                    Processor.non_official,
                                    frame_idx,
                                    data_file,
                                    Processor.official
                                )
                            print(f"Arrivé à la frame {frame_idx}")
                        else:
                            print(f"Index de frame invalide. Doit être entre 0 et {N_frame-1}")
                            continue
                    except ValueError:
                        print("Index de frame invalide. Utilisation: goto FRAME_IDX")
                        continue

                elif cmd.strip() == "run":
                    run_mode = True
                    print("Mode exécution continue activé.")

                else:
                    print("Commande non reconnue.")
                    continue

            # Exécuter le tracking pour la frame actuelle
            if frame_idx < N_frame - 1:
                Processor.tracking_update(
                    Processor.non_official,
                    frame_idx,
                    data_file,
                    Processor.official
                )

                plot_trackers(fig, ax, Processor.non_official, Processor.official, Processor.retired, frame_idx)
                # plt.draw()
                # plt.pause(0.1)

                if run_mode:
                    frame_idx += 1
                    plt.pause(0.2)

        print("Tracking terminé. Finalisation...")
        Processor.tracking_finalize(Processor.official)

        plot_trackers(fig, ax, [], [], Processor.retired, frame_idx)
        plt.show()

        print(f"Trackers retirés: {len(Processor.retired)}")
        print("Fin de la session de débogage")

    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    data_file = "data/30-04/marche 2-15m.npz"  # Exemple de fichier
    debug_full_tracking_and_plot(data_file)