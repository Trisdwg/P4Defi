import matplotlib.pyplot as plt
import numpy as np
import csv
import Processor

# === PARAMÈTRES ===
data_file = "data/18-04/doublemarche4-17memevit.npz"
output_csv = "annotations.csv"
channel = 1

# === Chargement des données ===
data, f0, B, Ms, Mc, Ts, Tc = Processor.load_file(data_file)
N_frame = data.shape[0]

# === Sauvegarde interactive ===
all_annotations = []

for frame_idx in range(N_frame):
    rdm = Processor.compute_RDM(data_file, frame_idx)[channel]

    fig, ax = plt.subplots()
    ax.set_title(f"Cliquez sur les cibles - Frame {frame_idx+1}/{N_frame}")
    im = ax.imshow(rdm, cmap='jet', origin='lower', aspect='auto')
    plt.xlabel("Range bins")
    plt.ylabel("Doppler bins")
    plt.colorbar(im, label="Amplitude")

    print(f"\n[Frame {frame_idx+1}] Cliquez sur les cibles (appuyez sur Entrée quand fini)")
    clicks = plt.ginput(n=-1, timeout=0)
    plt.close()

    # Convertit les clics en indices entiers (col=x=range, row=y=doppler)
    frame_annotations = [int(len(clicks))]
    for x, y in clicks:
        frame_annotations.append(int(round(y)))  # doppler_idx
        frame_annotations.append(int(round(x)))  # range_idx

    all_annotations.append(frame_annotations)

# === Sauvegarde dans un fichier CSV ===
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_annotations)

print(f"\nAnnotations sauvegardées dans {output_csv} ✅")
