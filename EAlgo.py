import numpy as np
import csv
import Processor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

psf_empirique_centered = Processor.psf_empirique_centered
psf_theorique_centered = Processor.psf_theorique_centered

# --- GROUND TRUTH ---
def load_ground_truth_csv(path):
    all_gt = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for frame_idx, row in enumerate(reader):
            if not row:
                all_gt.append([])  # ligne vide = aucune cible
                continue
            try:
                n = int(row[0])
                if len(row) != 1 + 2 * n:
                    print(f"⚠️ Ligne {frame_idx+1}: attendu {2*n} valeurs mais reçu {len(row)-1}. Ignorée.")
                    all_gt.append([])
                    continue
                gt = [(int(row[1 + 2 * i]), int(row[1 + 2 * i + 1])) for i in range(n)]
                all_gt.append(gt)
            except Exception as e:
                print(f"❌ Erreur parsing ligne {frame_idx+1}: {e}")
                all_gt.append([])
    return all_gt

def compute_f1_score(detections, ground_truth, max_distance=15, soft=True):
    if len(ground_truth) == 0 and len(detections) == 0:
        return 1.0, 1.0, 1.0
    if len(ground_truth) == 0 or len(detections) == 0:
        return 0.0, 0.0, 0.0

    TP = 0.0
    matched_gt = set()

    for d in detections:
        best_score = 0.0
        best_idx = None
        for i, g in enumerate(ground_truth):
            if i in matched_gt:
                continue
            dist = np.linalg.norm(np.array(d) - np.array(g))
            if dist <= max_distance:
                score = 1.0 if not soft else 1.0
                if score > best_score:
                    best_score = score
                    best_idx = i
        if best_idx is not None:
            matched_gt.add(best_idx)
            TP += best_score

    FP = len(detections) - TP
    FN = len(ground_truth) - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return f1, precision, recall

# --- EVALUATION ---
def evaluate_individual(indiv, data_file, frames, ground_truths):
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    psf = indiv['alpha']*psf_empirique_centered + (1-indiv['alpha'])*psf_theorique_centered 

    for frame_idx in frames:
        rdm = Processor.compute_RDM(data_file, frame_idx)[1]
        gt = ground_truths[frame_idx]

        raw_targets, _ = Processor.clean_rdm(
            rdm,
            psf_full=psf,
            threshold=indiv['threshold'],
            max_iter=int(indiv['max_iter']),
            radius_doppler=int(indiv['size_doppler']),
            radius_range=int(indiv['size_range']),
            use_binary_mask=True,
        )
        grouped = Processor.fuse_targets(
            raw_targets,
            max_dist_doppler=indiv['eps_doppler'],
            max_dist_range=indiv['eps_range']
        )
        detected = [g[0] for g in grouped]

        f1, precision, recall = compute_f1_score(detected, gt)
        total_f1 += f1
        total_precision += precision
        total_recall += recall

    n = len(frames)
    # Normalisation des eps sur leur plage max possible
    max_eps_doppler = 100.0
    max_eps_range   = 60.0
    norm_eps = (indiv['eps_doppler']/max_eps_doppler + indiv['eps_range']/max_eps_range) / 2

    # Coefficient de pénalité à ajuster (entre 0 et 1)
    lambda_pen = 0.3

    fitness = (total_f1 / n) - lambda_pen * norm_eps
    #return fitness

    return total_f1 / n, total_precision / n, total_recall / n

# --- PARAMETRES ---
DATA_FILE = "data/Mesures_18-04/doublemarche4-17memevit.npz"
CSV_GT = "annotations.csv"
FRAME_INDICES = list(range(150))
POP_SIZE = 100
GENERATIONS = 10

# --- GENERATION & MUTATION ---
def random_individual():
    return {
        'threshold': np.random.uniform(0.009, 0.012),
        'size_doppler': np.random.uniform(42, 46),
        'size_range': np.random.uniform(10, 12),
        'eps_doppler': np.random.randint(10, 50),
        'eps_range': np.random.randint(5, 35),
        'max_iter': np.random.randint(4, 12),
        'alpha': np.random.uniform(0, 1)
    }

def mutate(ind):
    new = ind.copy()

    # --- real-valued genes ------------------------------------------------
    for key in ['threshold', 'alpha']:
        if np.random.rand() < 0.3:
            new[key] += np.random.normal(0, 0.5)

        # keep each parameter in its legal range
        if key == 'alpha':                 # ← 0 ≤ alpha ≤ 1
            new[key] = np.clip(new[key], 0.0, 1.0)
        else:                              # threshold ≥ 0.01
            new[key] = max(0.01, new[key])

    # --- integer-valued genes --------------------------------------------
    for key in ['eps_doppler', 'eps_range',
                'max_iter', 'size_doppler', 'size_range']:
        if np.random.rand() < 0.4:
            new[key] += np.random.randint(-2, 3)
        new[key] = max(1, new[key])

    return new

# --- EVOLUTION ---
def evolve_population(data_file, frames, ground_truths, psf):
    population = [random_individual() for _ in range(POP_SIZE)]
    for ind in population:
        ind['psf'] = psf

    best_score = -np.inf
    best_ind = None
    f1_per_generation = []
    plateau_count = 0

    for gen in range(GENERATIONS):
        print(f"\n=== Génération {gen+1} ===")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_individual, ind, data_file, frames, ground_truths) for ind in population]
            scores = [f.result() for f in tqdm(futures)]

        gen_best_idx = int(np.argmax(scores))
        gen_best_score = scores[gen_best_idx]
        f1_per_generation.append(gen_best_score)
        gen_best_ind = population[gen_best_idx] 


        print(f"  🥇 Meilleur F1 ajusté : {gen_best_score:.4f}")
        print("  🧬 Paramètres :")
        for k, v in gen_best_ind.items():
            if k != 'psf':
                print(f"     {k}: {v}")

        if gen_best_score > best_score:
            best_score = gen_best_score
            print(f"Meilleur F1-score de la génération {gen+1}: {best_score:.4f}")
            best_ind = population[gen_best_idx]
            plateau_count = 0
        else:
            plateau_count += 1

        if plateau_count >= 10:
            print("\n🔁 Arrêt anticipé : F1-score stagné pendant 4 générations.")
            break

        top_k = max(1, POP_SIZE // 4)
        top_inds = [population[i] for i in np.argsort(scores)[-top_k:]]

        new_population = [best_ind.copy()]  # elitism
        while len(new_population) < POP_SIZE:
            parent = top_inds[np.random.randint(len(top_inds))]
            child = mutate(parent)
            child['psf'] = psf
            new_population.append(child)

        population = new_population

    # Plot convergence
    plt.figure()
    plt.plot(f1_per_generation, marker='o')
    plt.title("Évolution du F1-score par génération")
    plt.xlabel("Génération")
    plt.ylabel("Meilleur F1-score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_ind, best_score

# --- MAIN ---
"""if __name__ == '__main__':
    print("Chargement de la PSF expérimentale...")
    psf = Processor.build_empirical_psf(DATA_FILE, frame_idx=5, ch=0)

    print("Chargement des annotations ground-truth...")
    ground_truths = load_ground_truth_csv(CSV_GT)

    print("Lancement de l'optimisation évolutionnaire...")
    best_ind, best_score = evolve_population(DATA_FILE, FRAME_INDICES, ground_truths, psf)

    print("\n=== Meilleurs paramètres trouvés ===")
    for k, v in best_ind.items():
        if k != 'psf':
            print(f"{k} = {v}")
    print(f"Score F1 moyen : {best_score:.4f}")"""

if __name__ == '__main__':

    indiv = {
        'threshold' : 0.010790605558155648,
        'size_doppler' : 45.059643672455984,
        'size_range' : 9.999565022794656,
        'eps_doppler' : 57,
        'eps_range' : 41,
        'max_iter' : 7,
        'alpha' : 0.0
    }
    print("Chargement de la PSF ...")
    psf = indiv['alpha']*psf_empirique_centered + (1-indiv['alpha'])*psf_theorique_centered

    print("evaluation de l'individu...")
    data_file = DATA_FILE
    frames = FRAME_INDICES
    ground_truths = load_ground_truth_csv(CSV_GT)
    f1, precision, recall= evaluate_individual(indiv, data_file, frames, ground_truths)
    print(f"F1-score: {f1:.4f}")
    print(f"Précision: {precision:.4f}")
    print(f"Rappel: {recall:.4f}")
    print("Fin de l'évaluation.")





