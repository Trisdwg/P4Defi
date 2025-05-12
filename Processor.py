import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.ndimage import shift as subpixel_shift
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
import time

# ===========================================================================
# Paramètres physiques 
# ===========================================================================

OFFSETS = np.asarray([ 7.33982036,  7.3548503 , 10.86700599,  3.64670659])/2
OFFSETS_2 = np.asarray([4.63640474, 8.4018184, 8.3141635, 11.66615912])/2
ANTENNA_POS = np.asarray([[-0.35,2.7],[1.8,0.5],[5.1,-2.3],[0.0,0.0]])
ANTENNA_POS_2 = np.asarray([[0.0,0.0],[-2.5,1.6],[2.05,0.0],[2.4,0.9]])

def load_file(file) :
    file = np.load(file)
    raw = file['data']      
    f0, B, Ms, Mc, Ts, Tc = file['chirp']
    Ms, Mc = int(Ms), int(Mc)
    N_frame, N_chan, M = raw.shape

    # On enlève les pauses
    M_pause = int(M / Mc - Ms)
    data = raw.reshape(N_frame, N_chan, Mc, Ms + M_pause)[..., :Ms]
    data = data.reshape(N_frame, N_chan, Mc * Ms)

    return data, float(f0), float(B), Ms, Mc, float(Ts), float(Tc)

f0, B, Ms, Mc, Ts, Tc = load_file("data/18-04/calibration6m.npz")[1:7]

PAD_R = 16
PAD_D = 16

delta_r = 3e8 / (2 * (B * PAD_R))
delta_v = 3e8 / (2 * (f0 * Mc * Tc * PAD_D))
# ===========================================================================
# Partie 1 : RDM
# ===========================================================================

def compute_RDM(file, frame_idx=0):
    data, f0, B, Ms, Mc, Ts, Tc = load_file(file)

    # Choix du jeu d'offsets
    # si le chemin ou le nom du fichier contient "data/18-04", on prend OFFSETS_2
    if "data/18-04" in file:
        offsets = OFFSETS_2
    else:
        offsets = OFFSETS

    RDM = []
    for ch in range(data.shape[1]):
        sig = data[frame_idx, ch].astype(float).reshape(Mc, Ms)
        # centrage en amplitude
        sig -= sig.mean(axis=0, keepdims=True)

        # FFT rang puis FFT Doppler
        R = np.fft.fft(sig, n=PAD_R * Ms, axis=1)
        D = np.fft.fftshift(
                np.fft.fft(R, n=PAD_D * Mc, axis=0),
                axes=0
            )
        rdm = np.abs(D)**2

        # Correction des distances par décalage subpixel
        # => on roule la carte RDM selon l'offset spécifique à ce canal
        shift_bins = int(np.round(-offsets[ch] / delta_r))
        rdm = np.roll(rdm, shift_bins, axis=1)

        # on ne garde que la moitié avant de l'axe distances
        rdm = rdm[:, : (PAD_R * Ms)//2]

        RDM.append(rdm)

    return RDM
# ===========================================================================
# Partie 2 : Tracking monocible multistatique
# ===========================================================================
 
def compute_position(file, frame_idx=0):
    # On récupère aussi B
    data, f0, B, Ms, Mc, Ts, Tc = load_file(file)
    RDM = compute_RDM(file, frame_idx)
    distances = []

    # Choix des positions d'antennes en fonction du chemin de fichier
    if "data/18-04" in file:
        antenna_pos = ANTENNA_POS_2
    else:
        antenna_pos = ANTENNA_POS

    for ch, rdm in enumerate(RDM):
        _, r_idx = np.unravel_index(np.argmax(rdm), rdm.shape)
        dist = max(2* r_idx * delta_r, 0.0)
        distances.append(dist)

    def resid(p):
        x, y = p
        res = []
        for q,dmes in enumerate(distances):
            d = (x**2+y**2)**(1/2) + ((x-antenna_pos[q][0])**2 + (y-antenna_pos[q][1])**2)**(1/2)
            res.append(dmes-d)
        return res


    p0 = np.mean(antenna_pos, axis=0)
    sol = least_squares(resid, p0, loss='cauchy')
    return tuple(sol.x)


def compute_speed(file, frame_idx=0):
    # 1) Estimer la position P
    x, y = compute_position(file, frame_idx)
    P = np.array([x, y])
    
    # Choix des positions d'antennes en fonction du chemin de fichier
    if "data/18-04" in file:
        antenna_pos = ANTENNA_POS_2
    else:
        antenna_pos = ANTENNA_POS
        
    T = antenna_pos[0]

    # 2) Charger le RDM et extraire les vitesses radiales u_q pour q=0..3
    RDM = compute_RDM(file, frame_idx)
    u = []
    N = PAD_D * Mc
    for ch in range(4):
        rdm = RDM[ch]
        # on récupère l'indice Doppler comme l'indice de la ligne (axis=0)
        v_idx, _ = np.unravel_index(np.argmax(rdm), rdm.shape)
        # centrage autour de zéro
        k = v_idx - N//2
        # fréquence Doppler (Hz)
        fd = k * (1.0 / (Mc * Tc) / PAD_D)
        # vitesse radiale (m/s)
        u.append((3e8 / (2 * f0)) * fd)
    u = np.array(u)   # shape (4,)

    # 3) Construire la matrice H (4 lignes, 2 colonnes)
    n_tx = (P - T) / np.linalg.norm(P - T)
    H = []
    for ch in range(4):
        Rq = antenna_pos[ch]
        n_rx = (P - Rq) / np.linalg.norm(P - Rq)
        hq = 0.5 * (n_tx + n_rx)
        H.append(hq)
    H = np.vstack(H)  # shape (4,2)
    A = (np.linalg.inv(H.T @ H)) @ (H.T)
    v = A @ np.array(u)
    return v, np.linalg.norm(v)

def kalman_filter_monocible(file, frame_idx, kalman_x, kalman_P, outlierRadius):
    print("Frame ", frame_idx)

    RDMs = compute_RDM(file, frame_idx)
    d_q = []
    v_q = []
    for q, rdm in enumerate(RDMs):
        vindex, rindex = np.unravel_index(np.argmax(rdm), rdm.shape)
        d = 2 * float(rindex)* delta_r
        v = float(vindex-(Mc*16)//2)*delta_v
        d_q.append(d)
        v_q.append(v)
    # print(v_q)
    d_q = np.array(d_q)
    v_q = np.array(v_q)

    # Choix des positions d'antennes en fonction du chemin de fichier
    if "data/18-04" in file:
        antenna_pos = ANTENNA_POS_2
    else:
        antenna_pos = ANTENNA_POS

    deltaTFrame = Mc * Tc
    kalman_F = np.array([[1,0,deltaTFrame,0],[0,1,0,deltaTFrame],[0,0,1,0],[0,0,0,1]])
    kalman_Q = np.array([[0,0,0,0],[0,0,0,0],[0,0,0.1,0],[0,0,0,0.1]])
    kalman_H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    kalman_R = np.array([[3,0,0,0],[0,3,0,0],[0,0,5,0],[0,0,0,5]])
    kalman_xp = kalman_F @ kalman_x
    kalman_Pp = kalman_F @ kalman_P @ kalman_F.T + kalman_Q

    kalman_K = kalman_Pp @ kalman_H.T @ np.linalg.inv(kalman_H @ kalman_P @ kalman_H.T + kalman_R)
    takeRDM = np.ones(4, dtype=int)
    z = np.zeros(4)
    # point_est = speed_est = np.zeros(2)
    for q in range(len(takeRDM)):
        dp = (kalman_xp[0]**2+kalman_xp[1]**2)**(1/2) + ((kalman_xp[0]-antenna_pos[q][0])**2 + (kalman_xp[1]-antenna_pos[q][1])**2)**(1/2)
        vp = 0.5 * ((np.array([kalman_xp[0]-antenna_pos[q][0], kalman_xp[1]-antenna_pos[q][1]]) @ kalman_xp[2:])*(1/np.sqrt((kalman_xp[0]-antenna_pos[q][0])**2+(kalman_xp[1]-antenna_pos[q][1])**2)) +
                    (kalman_xp[:2] @ kalman_xp[2:])*(1/np.sqrt((kalman_xp[0])**2+(kalman_xp[1])**2)))
        dist = np.sqrt((dp - d_q[q])**2+(vp - v_q[q])**2)
        if(dist >= outlierRadius):
            takeRDM[q] = 0
    if(np.sum(takeRDM) < 2):
        z = kalman_xp
    else:
        # print(d_q)
        # print(v_q)
        d_q = d_q[takeRDM==1]
        v_q = v_q[takeRDM==1]
        print(takeRDM)
        # print(d_q)
        # print(v_q)
        def diff(p):
            x,y = p
            res = []
            for q,dmes in enumerate(d_q):
                d = (x**2+y**2)**(1/2) + ((x-antenna_pos[q][0])**2 + (y-antenna_pos[q][1])**2)**(1/2)
                res.append(dmes-d)
            return res
        
        x0 = [0.0,0.0]
        point_est = least_squares(diff,x0,loss='cauchy').x
        # print(point_est)

        N = np.fromfunction(
            lambda q, i: 0.5 * (
                (point_est[i.astype(int)] - antenna_pos[q.astype(int), i.astype(int)]) *
                (1 / np.sqrt((point_est[0] - antenna_pos[q.astype(int), 0])**2 + (point_est[1] - antenna_pos[q.astype(int), 1])**2)) +
                point_est[i.astype(int)] *
                (1 / np.sqrt(point_est[0]**2 + point_est[1]**2))
            ),
            (np.sum(takeRDM), 2),
        )

        try:
            speed_est = np.linalg.inv(N.T @ N) @ N.T @ v_q
        except np.linalg.LinAlgError:
            # chute propre : on retourne une vitesse nulle ou None
            speed_est = np.array([None, None])
        # print(speed_est)
        z = np.concatenate((point_est,speed_est))

    kalman_x = kalman_xp + kalman_K @ (z - kalman_H @ kalman_xp)
    kalman_P = kalman_Pp - kalman_K @ kalman_H @ kalman_Pp

    return kalman_x, kalman_P

# ===========================================================================
# Partie 3 : Multicible Bistatique
# ===========================================================================

def ca_cfar_convolve(rdm, guard_size_doppler=10, guard_size_range=11,
                     window_size_doppler=45,window_size_range=15, alpha=10.0):
    n_doppler, n_range = rdm.shape

    # Total window size
    total_doppler = 2 * (window_size_doppler + guard_size_doppler) + 1
    total_range = 2 * (window_size_range + guard_size_range) + 1

    # Guard size
    guard_doppler = 2 * guard_size_doppler + 1
    guard_range = 2 * guard_size_range + 1

    # Create convolution kernels
    kernel_window = np.ones((total_doppler, total_range), dtype=np.float32)
    kernel_guard = np.zeros_like(kernel_window)

    # Fill guard region inside the total window with 1s
    center_d, center_r = total_doppler // 2, total_range // 2
    kernel_guard[
        center_d - guard_size_doppler:center_d + guard_size_doppler + 1,
        center_r - guard_size_range:center_r + guard_size_range + 1
    ] = 1

    # CFAR kernel = window - guard
    kernel_cfar = kernel_window - kernel_guard
    n_cfar_cells = np.sum(kernel_cfar)

    # Convolve to get sum of training cells at each position
    padded_rdm = np.pad(rdm, ((total_doppler // 2, total_doppler // 2),
                              (total_range // 2, total_range // 2)), mode='edge')
    training_sums = fftconvolve(padded_rdm, kernel_cfar, mode='valid')

    # Compute thresholds
    thresholds = alpha * (training_sums / n_cfar_cells)

    # Generate detection mask
    mask = rdm > thresholds

    return mask, thresholds

def extract_targets_from_cfar(rdm, mask,
                              min_distance_doppler=20,
                              min_distance_range=10,
                              min_points_per_target=60):
    """
    Extrait les cibles à partir du masque CFAR en fusionnant les détections proches.
    Écarte les clusters contenant moins de `min_points_per_target` points.

    Retourne:
        targets : liste de tuples ((d, r), amplitude)
                  où (d, r) est la cellule CFAR de plus forte amplitude du cluster.
    """
    # print(min_distance_doppler,min_distance_range,min_points_per_target)
    detected_points = np.argwhere(mask)
    targets = []

    if len(detected_points) == 0:
        return targets

    # Indices des détections triés par amplitude décroissante
    amplitudes = rdm[mask]
    sorted_order = amplitudes.argsort()[::-1]
    sorted_points = detected_points[sorted_order]

    processed_mask = np.zeros_like(mask, dtype=bool)

    for d, r in sorted_points:

        # point déjà absorbé par un cluster précédent ?
        if processed_mask[d, r]:
            continue

        # ---------------------------------------------------------------------------------
        # Construction du cluster autour du point (d, r)
        # ---------------------------------------------------------------------------------
        cluster_idx = []           # pour compter les points du cluster
        d_min = max(0, d - min_distance_doppler)
        d_max = min(rdm.shape[0], d + min_distance_doppler + 1)
        r_min = max(0, r - min_distance_range)
        r_max = min(rdm.shape[1], r + min_distance_range + 1)

        for dd in range(d_min, d_max):
            for rr in range(r_min, r_max):
                if mask[dd, rr] and not processed_mask[dd, rr]:
                    # Condition elliptique de voisinage
                    if ((dd - d) / min_distance_doppler) ** 2 + \
                       ((rr - r) / min_distance_range) ** 2 <= 1:
                        cluster_idx.append((dd, rr))
                        processed_mask[dd, rr] = True

        # ---------------------------------------------------------------------------------
        # Accepte ou rejette le cluster selon sa taille
        # ---------------------------------------------------------------------------------
        if len(cluster_idx) >= min_points_per_target:
            # (d, r) est forcément le point de plus forte amplitude du cluster
            amp = rdm[d, r]
            targets.append(((d, r), amp))
        # sinon : cluster trop petit → ignoré

    return targets

def cfar_2d(rdm):
    """
    CFAR 2D adaptatif qui combine CA-CFAR et OS-CFAR avec détection 
    d'asymétrie pour éliminer les lobes secondaires.
    
    Arguments:
    - rdm: Matrice Range-Doppler
    - guard_size: Tuple (doppler, range) pour taille de la zone de garde
    - window_size: Tuple (doppler, range) pour taille de la fenêtre
    - alpha: Facteur multiplicatif pour le seuil
    - use_os_cfar: Si True, utilise OS-CFAR sinon CA-CFAR
    - os_percentile: Percentile pour OS-CFAR [0-100]
    - min_distance: Distance minimale entre cibles pour fusion
    
    Retourne:
    - targets: Liste de tuples ((d, r), amplitude)
    - mask: Masque binaire des détections
    - thresholds: Matrice des seuils calculés
    """

    # start_time = time.time()
    mask, thresholds = ca_cfar_convolve(
        rdm, 
    )
    # end_time = time.time()
    # print("Execution time convolve ca_cfar:", end_time - start_time, "seconds")

    # Define a star-shaped footprint (this one is a 3x3 example)
    # footprint = create_star_shaped_footprint([3*x for x in window_size])
    # print("Footprint shape:\n", footprint)

    # Apply the local maximum filter using the star-shaped footprint
    # local_max = rdm == maximum_filter(rdm, footprint=footprint, mode='constant')

    # Combine the CFAR mask with the local maximum mask
    # mask = mask & local_max  # Only keep the local maxima that are CFAR detections
    # Extraction et fusion des cibles
    targets = extract_targets_from_cfar(
        rdm, 
        mask, 
    )
    
    return targets, mask, thresholds

def compute_track_position_and_speed(track, file=None):
    """
    Calcule la position et la vitesse basées sur une piste (track).
    Utilise le paramètre file pour déterminer quelle configuration d'antennes utiliser.
    """
    # Déterminer la configuration d'antennes à utiliser
    if file and "data/18-04" in file:
        antenna_pos = ANTENNA_POS_2
    else:
        antenna_pos = ANTENNA_POS
        
    d_q = [2*track[i][1] for i in range(len(track))] #*2 car r -> d
    v_q = [track[i][0] for i in range(len(track))]
    def diff(p):
        x,y = p
        res = []
        for q,dmes in enumerate(d_q):
            d = (x**2+y**2)**(1/2) + ((x-antenna_pos[q][0])**2 + (y-antenna_pos[q][1])**2)**(1/2)
            res.append(dmes-d)
        return res
    
    x0 = [0.0,0.0]
    point_est = least_squares(diff,x0,loss='linear').x

    speed_est = [None, None]
    if (point_est[0] != 0.0 and point_est[1] != 0.0):
        N = np.fromfunction(
            lambda q, i: 0.5 * (
                (point_est[i.astype(int)] - antenna_pos[q.astype(int), i.astype(int)]) *
                (1 / np.sqrt((point_est[0] - antenna_pos[q.astype(int), 0])**2 + (point_est[1] - antenna_pos[q.astype(int), 1])**2)) +
                point_est[i.astype(int)] *
                (1 / np.sqrt(point_est[0]**2 + point_est[1]**2))
            ),
            (len(track), 2),
            dtype=int
        )

        try:
            speed_est = np.linalg.inv(N.T @ N) @ N.T @ v_q
        except np.linalg.LinAlgError:
            # chute propre : on retourne une vitesse nulle ou None
            speed_est = np.array([None, None])

    return [(point_est[0],point_est[1]),(speed_est[0],speed_est[1])]


from itertools import product
deltaTFrame = Mc * Tc
kalman_params = {
    'F' : np.array([[1,0,deltaTFrame,0],[0,1,0,deltaTFrame],[0,0,1,0],[0,0,0,1]]),
    'Q' : np.array([[0,0,0,0],[0,0,0,0],[0,0,0.1,0],[0,0,0,0.1]]),
    'H' : np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
    'R' : np.array([[1,0,0,0],[0,1,0,0],[0,0,2.5,0],[0,0,0,2.5]])
}
non_official = []
official = []
retired = []
class tracker:
    def __init__(self, id, kalman_x, kalman_P, frame_start=0) :
        self.id = id
        self.kalman_xp = None
        self.kalman_Pp = None
        self.kalman_x = kalman_x
        self.kalman_P = kalman_P
        self.kalman_R = kalman_params['R']
        self.non_official_count = 0
        self.misses = 0
        self.official_count = 0
        self.frame_start = frame_start  # Nouveau: frame à laquelle le tracker a commencé
        self.history = [[(kalman_x[0], kalman_x[1]),(kalman_x[2], kalman_x[3])]] # [(pos),(vel)]
    
    def kalman_predict(self):
        self.kalman_xp = kalman_params['F'] @ self.kalman_x
        self.kalman_Pp = kalman_params['F'] @ self.kalman_P @ kalman_params['F'].T + kalman_params['Q']
        return self.kalman_xp, self.kalman_Pp
    
    def kalman_update(self, z, distance=-1):
        if distance == -1:
            self.kalman_R = 2*kalman_params['R']
        else:
            self.kalman_R = distance*kalman_params['R']
        kalman_K = self.kalman_Pp @ kalman_params['H'].T @ np.linalg.inv(kalman_params['H'] @ self.kalman_P @ kalman_params['H'].T + self.kalman_R)
        self.kalman_x = self.kalman_xp + kalman_K @ (z - kalman_params['H'] @ self.kalman_xp)
        self.kalman_P = self.kalman_Pp - kalman_K @ kalman_params['H'] @ self.kalman_Pp
        self.history.append([(self.kalman_x[0], self.kalman_x[1]),(self.kalman_x[2], self.kalman_x[3])])
        return self.kalman_x, self.kalman_P
    
    def __str__(self):
        return f"Tracker ID: {self.id},\n Kalman State: {self.kalman_x}, Kalman Covariance: {self.kalman_P},\n Kalman Predicted State: {self.kalman_xp}, Kalman Predicted Covariance: {self.kalman_Pp}, \n Kalman R: {self.kalman_R} \n History: {self.history}, \n Misses: {self.misses}, Non-official Count: {self.non_official_count}, Official Count: {self.official_count},\n Frame Start: {self.frame_start}\n\n"
    

def extract_all_targets(RDM_frame, file=None):
    """Retourne une liste à 4 entrées (une par canal) contenant
    soit la liste [(v, r), …] soit [None] si pas de cible."""
    all_t = [[] for _ in range(4)]
    for ch in range(4):
        mask, _ = ca_cfar_convolve(RDM_frame[ch])
        targets = extract_targets_from_cfar(RDM_frame[ch], mask)
        for (v_idx, r_idx) in (t[0] for t in targets):
            r = float(r_idx) * delta_r
            v = float(v_idx-(Mc*16)//2)*delta_v
            all_t[ch].append((v, r))
    # remplace les listes vides par [None] pour que product() les traite
    return [chan if chan else [None] for chan in all_t]

def make_intraframe_tracks(all_targets, file=None):
    """Associe entre eux les canaux d'une même frame.
       Une track = liste de (v, r) ayant >=2 canaux renseignés."""
    tracks = []
    for combo in product(*all_targets):          # 4-uplet (ou None)
        track = [combo[ch] for ch in range(4) if combo[ch] is not None]
        acceptableTrack = True
        track_pos_speed = compute_track_position_and_speed(track, file)
        if(track_pos_speed[0][0] == 0.0 and track_pos_speed[0][1] == 0.0 or track_pos_speed[1][0] == None or track_pos_speed[1][1] == None):
            acceptableTrack = False
        if 2 <= len(track) <= 4 and acceptableTrack:
            tracks.append(track_pos_speed)
    return tracks

    
def tracking_init(file) :
    global NEXT_ID
    RDM = compute_RDM(file,0)  # Utiliser frame 0 pour l'initialisation
    all_tragets = extract_all_targets(RDM, file)
    tracks = make_intraframe_tracks(all_tragets, file)
    for i, track in enumerate(tracks):
        non_official.append(tracker(NEXT_ID, np.array([track[0][0], track[0][1], track[1][0], track[1][1]]), np.eye(4), frame_start=0))
        NEXT_ID += 1
    return non_official


from scipy.spatial.distance import cdist
MAX_GATING_DIST_4D = 3.4                # seuil en 4-D

def tracking_update(non_official, frame_idx, file, official=None):
    global NEXT_ID
    # 1. ---------- prédiction ----------
    all_trk = non_official + (official or [])
    if not all_trk:
        return
    for trk in all_trk:
        trk.kalman_predict()

    # 2. ---------- extraction des mesures ----------
    RDM     = compute_RDM(file, frame_idx)
    tracks  = make_intraframe_tracks(extract_all_targets(RDM, file), file)   # [(pos),(vel)]
    # print(len(tracks), " tracks for frame", frame_idx, ":", tracks)

    # Initialiser assigned_cols comme un ensemble vide dans tous les cas
    assigned_cols = set()
    
    if not tracks:
        # Si pas de tracks, on met à jour avec la prédiction
        for trk in all_trk:
            trk.kalman_update(trk.kalman_xp)
    else:
        # 3. ---------- matrices 4-D ----------
        pred_state = np.vstack([trk.kalman_xp for trk in all_trk])     # (N,4)
        meas_state = np.vstack([[p[0], p[1], v[0], v[1]] for p, v in tracks])  # (M,4)

        D = cdist(pred_state, meas_state)                              # (N,M)

        # 4. ---------- association NN ----------
        # print(assigned_cols)
        for i, row in enumerate(D):
            j = np.argmin(row)
            if row[j] < MAX_GATING_DIST_4D:
                z = meas_state[j]
                # print(f"Tracker {i} assigned to measurement {j} with distance {row[j]}")
                assigned_cols.add(j)
                all_trk[i].kalman_update(z,row[j])
            else:
                z = pred_state[i]
                all_trk[i].misses += 1
                all_trk[i].kalman_update(z)

        # print(assigned_cols)
    
    # Mise à jour des compteurs et gestion des trackers
    for tracked in all_trk:
    # ---------- compteurs ----------
        if tracked in non_official:
            tracked.non_official_count += 1
            
            # Décision pour les trackers non_official uniquement
            if tracked.misses > 0.4 * tracked.non_official_count:
                # Trop de détections manquées pour un tracker non_official -> on le supprime
                non_official.remove(tracked)
            elif tracked.non_official_count > 20 and tracked.misses < 0.15 * tracked.non_official_count:
                # Assez de détections pour la promotion -> on le transfère vers official
                official.append(tracked)
                non_official.remove(tracked)
                
        else:  # Le tracker est dans official
            tracked.official_count += 1
            
            # Décision pour les trackers official uniquement
            if tracked.misses > 0.5 * (tracked.non_official_count + tracked.official_count):
                # Trop de détections manquées pour un tracker official -> on le retire
                retired.append(tracked)
                official.remove(tracked)
    
    # Gestion des nouvelles pistes uniquement si on a des tracks
    if tracks:
        remaining_cols = set(range(len(tracks))) - assigned_cols
        for j in remaining_cols:
            z = meas_state[j]

            # Vérifie la proximité
            dists_to_existing = np.linalg.norm(pred_state - z, axis=1)
            if np.any(dists_to_existing < 2):   # seuil à ajuster
                # print(f"--> Skip creating tracker for measurement {j}, too close to existing tracker")
                continue

            # Crée le tracker si aucun doublon
            new_trk = tracker(NEXT_ID, np.array([z[0], z[1], z[2], z[3]]), np.eye(4), frame_start=frame_idx)
            NEXT_ID += 1
            non_official.append(new_trk)
            # print(f"--> Created new tracker {new_trk.id} for measurement {j}")


def tracking_finalize(official):
    # transfère VRAIMENT tous les trackers d'« official » vers « retired »
    retired.extend(official)      # ou retired += official
    official.clear()

def cluster_retired_trackers(distance_threshold=2.0, angle_threshold=15.0, time_overlap_threshold=0.6):
    """
    Fusionne les trackers de la liste 'retired' qui sont trop proches et ont des directions similaires.
    Version améliorée avec une approche plus robuste et moins sensible au chevauchement temporel.
    
    Args:
        distance_threshold: Seuil de distance maximale entre les trajectoires (en mètres)
        angle_threshold: Seuil d'angle maximal entre les directions (en degrés)
        time_overlap_threshold: Proportion minimale de temps pendant lequel les trajectoires doivent se chevaucher
    
    Returns:
        Liste des trackers après clustering
    """
    print(f"DEBUG - Clustering parameters: dist={distance_threshold}, angle={angle_threshold}, overlap={time_overlap_threshold}")
    
    if len(retired) <= 1:
        return retired
        
    # Fonction pour calculer l'angle entre deux vecteurs
    def angle_between(v1, v2):
        # Normaliser les vecteurs
        v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
        # Calculer l'angle en degrés
        angle_rad = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
        return np.degrees(angle_rad)
    
    # Nouvelle fonction pour calculer la distance minimale entre deux trajectoires
    def min_distance(trk_i, trk_j):
        # Extraire toutes les positions de chaque trajectoire
        pos_i = np.array([state[0] for state in trk_i.history])
        pos_j = np.array([state[0] for state in trk_j.history])
        
        # Calculer la distance minimale entre n'importe quelle paire de points
        min_dist = float('inf')
        for p_i in pos_i:
            for p_j in pos_j:
                dist = np.linalg.norm(np.array(p_i) - np.array(p_j))
                if dist < min_dist:
                    min_dist = dist
        
        return min_dist
    
    # Fonction pour vérifier si deux trajectoires sont temporellement proches
    def are_temporally_close(trk_i, trk_j, overlap_threshold):
        # Calculer les périodes de temps des deux trajectoires
        start_i = trk_i.frame_start
        end_i = trk_i.frame_start + len(trk_i.history) - 1
        
        start_j = trk_j.frame_start
        end_j = trk_j.frame_start + len(trk_j.history) - 1
        
        # Calculer le chevauchement
        overlap_start = max(start_i, start_j)
        overlap_end = min(end_i, end_j)
        
        # Durée du chevauchement
        overlap_length = max(0, overlap_end - overlap_start + 1)
        
        # Durées totales
        duration_i = end_i - start_i + 1
        duration_j = end_j - start_j + 1
        
        # Calculer le ratio de chevauchement par rapport à la trajectoire la plus courte
        min_duration = min(duration_i, duration_j)
        overlap_ratio = overlap_length / min_duration if min_duration > 0 else 0
        
        print(f"DEBUG - Overlap check: trk {trk_i.id} vs {trk_j.id}, ratio={overlap_ratio}, threshold={overlap_threshold}")
        
        # Premier test: le ratio de chevauchement doit être >= au seuil
        if overlap_ratio >= overlap_threshold:
            return True
            
        # Si le seuil de chevauchement est 0, on peut utiliser le critère de proximité temporelle
        if overlap_threshold == 0 and min_duration > 0:
            # Si les tracks ne se chevauchent pas, vérifier la proximité temporelle
            if overlap_length == 0:
                gap = min(abs(end_i - start_j), abs(end_j - start_i))
                max_allowed_gap = 10  # Maximum 10 frames d'écart
                return gap <= max_allowed_gap
        
        # Par défaut, pas assez de chevauchement
        return False
    
    # Fonction pour vérifier si deux trajectoires peuvent être fusionnées
    def can_merge(trk_i, trk_j):
        # 1. Vérifier la distance spatiale
        dist = min_distance(trk_i, trk_j)
        if dist > distance_threshold:
            print(f"DEBUG - Distance check failed: trk {trk_i.id} vs {trk_j.id}, dist={dist}, threshold={distance_threshold}")
            return False
        
        # 2. Vérifier la proximité temporelle (chevauchement ou gap raisonnable)
        if not are_temporally_close(trk_i, trk_j, time_overlap_threshold):
            print(f"DEBUG - Temporal check failed: trk {trk_i.id} vs {trk_j.id}")
            return False
            
        # 3. Vérifier la similitude de direction
        # Calculer les vecteurs de direction moyenne
        if len(trk_i.history) < 2 or len(trk_j.history) < 2:
            # Si une des trajectoires est trop courte, ne pas vérifier l'angle
            return True
            
        # Calculer les directions moyennes (fin - début pour les trajectoires assez longues)
        dir_i = np.array(trk_i.history[-1][0]) - np.array(trk_i.history[0][0])
        dir_j = np.array(trk_j.history[-1][0]) - np.array(trk_j.history[0][0])
        
        # Si l'un des vecteurs est trop court, utiliser les vitesses Kalman
        if np.linalg.norm(dir_i) < 0.5:
            dir_i = np.array([trk_i.kalman_x[2], trk_i.kalman_x[3]])
        if np.linalg.norm(dir_j) < 0.5:
            dir_j = np.array([trk_j.kalman_x[2], trk_j.kalman_x[3]])
            
        # Si les vecteurs sont toujours trop courts, considérer que les directions sont similaires
        if np.linalg.norm(dir_i) < 0.1 or np.linalg.norm(dir_j) < 0.1:
            return True
            
        angle = angle_between(dir_i, dir_j)
        print(f"DEBUG - Angle check: trk {trk_i.id} vs {trk_j.id}, angle={angle}, threshold={angle_threshold}")
        return angle <= angle_threshold
    
    def simple_merge(candidates):
        # Sélectionner le meilleur tracker basé sur le ratio de miss le plus bas
        best_idx = 0
        best_ratio = float('inf')
        for idx, trk in enumerate(merge_candidates):
            total_frames = trk.non_official_count + trk.official_count
            ratio = trk.misses / total_frames if total_frames > 0 else 1.0
            if ratio < best_ratio or (ratio == best_ratio and len(trk.history) > len(merge_candidates[best_idx].history)):
                best_ratio = ratio
                best_idx = idx
        return best_idx
    
    def better_merge(candidates):
        min_start = min(trk.frame_start for trk in candidates)
        max_end = max(trk.frame_start + len(trk.history) - 1 for trk in candidates)
        ntracks = np.sum(1 for trk in candidates if trk.frame_start == min_start)
        prevstart = min_start
        overlaps = []
        for frame in range(min_start, max_end + 1):
            newntracks = np.sum(1 for trk in candidates if trk.frame_start == frame)
            if newntracks != ntracks:
                overlaps.append((prevstart, frame, ntracks))
                prevstart = frame
                ntracks = newntracks
    
    # Créer une copie de la liste des trackers
    trackers = retired.copy()
    
    # Réaliser plusieurs passes de fusion pour s'assurer que toutes les fusions possibles sont effectuées
    max_passes = 3
    for pass_num in range(max_passes):
        merges_done = False
        
        # Matrice d'adjacence pour les fusions potentielles
        n = len(trackers)
        merge_matrix = np.zeros((n, n), dtype=bool)
        
        # Construire la matrice d'adjacence
        for i in range(n):
            for j in range(i+1, n):
                if can_merge(trackers[i], trackers[j]):
                    merge_matrix[i, j] = merge_matrix[j, i] = True
                    print(f"DEBUG - Will merge: trk {trackers[i].id} and {trackers[j].id}")
        
        # Effectuer les fusions basées sur la matrice
        i = 0
        while i < len(trackers):
            # Trouver tous les indices à fusionner avec i
            to_merge = [j for j in range(len(trackers)) if j > i and merge_matrix[i, j]]
            
            if to_merge:
                merges_done = True
                # Obtenir tous les trackers à fusionner
                merge_candidates = [trackers[i]] + [trackers[j] for j in to_merge]
                print(f"NUMBER OF TRACKERS TO MERGE: {len(merge_candidates)}")

                best_idx = simple_merge(merge_candidates)
                
                # Conserver le meilleur tracker
                best_tracker = merge_candidates[best_idx]
                print(f"Fusion des trackers: {[t.id for t in merge_candidates]} -> {best_tracker.id}")
                
                # Supprimer tous les trackers fusionnés sauf le meilleur
                for idx in sorted(to_merge, reverse=True):
                    trackers.pop(idx)
                if best_idx != 0:  # Si le meilleur n'est pas le tracker i actuel
                    trackers[i] = best_tracker
                    
                # Ne pas incrémenter i, car nous avons modifié la liste
            else:
                i += 1
        
        # Si aucune fusion n'a été faite dans cette passe, arrêter
        if not merges_done:
            break
    
    # Mettre à jour la liste des trackers retirés
    retired.clear()
    retired.extend(trackers)
    
    return trackers
