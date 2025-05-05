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

ANTENNA_POS = np.asarray([
    [-0.35, 2.7],     # Channel 0 (common focus)
    [1.8, 0.5],   # Channel 1
    [5.1, -2.3],    # Channel 2
    [0.0, 0.0],     # Channel 3
])
OFFSETS = np.asarray([ 7.33982036,  7.3548503 , 10.86700599,  3.64670659])/2
ANTENNA_POS = np.asarray([[-0.35,2.7],[1.8,0.5],[5.1,-2.3],[0.0,0.0]])

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
    RDM = []
    for ch in range(data.shape[1]):
        sig = data[frame_idx, ch].astype(float).reshape(Mc, Ms)
        sig -= sig.mean(axis=0, keepdims=True)
        R = np.fft.fft(sig, n=PAD_R * Ms, axis=1)
        D = np.fft.fftshift(np.fft.fft(R, n=PAD_D * Mc, axis=0),axes=0)
        rdm = np.abs(D)**2
        #bins_5m = int(np.round(5.0 / delta_r))
        # on ne garde que la moitié de l'axe des distances
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

    for ch, rdm in enumerate(RDM):
        _, r_idx = np.unravel_index(np.argmax(rdm), rdm.shape)
        dist = max(2* r_idx * delta_r - OFFSETS[ch], 0.0)
        distances.append(dist)

    def resid(p):
        x, y = p
        res = []
        for q,dmes in enumerate(distances):
            d = (x**2+y**2)**(1/2) + ((x-ANTENNA_POS[q][0])**2 + (y-ANTENNA_POS[q][1])**2)**(1/2)
            res.append(dmes-d)
        return res


    p0 = np.mean(ANTENNA_POS, axis=0)
    sol = least_squares(resid, p0, loss='cauchy')
    return tuple(sol.x)


def compute_speed(file, frame_idx=0):
    # 1) Estimer la position P
    x, y = compute_position(file, frame_idx)
    P = np.array([x, y])
    T = ANTENNA_POS[0]

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
        Rq = ANTENNA_POS[ch]
        n_rx = (P - Rq) / np.linalg.norm(P - Rq)
        hq = 0.5 * (n_tx + n_rx)
        H.append(hq)
    H = np.vstack(H)  # shape (4,2)
    A = (np.linalg.inv(H.T @ H)) @ (H.T)
    v = A @ np.array(u)
    return v, np.linalg.norm(v)

def kalman_filter_monocible(file, frame_idx, kalman_x, kalman_P, outlierRadius):

    RDMs = compute_RDM(file, frame_idx)
    d_q = []
    v_q = []
    for q, rdm in enumerate(RDMs):
        vindex, rindex = np.unravel_index(np.argmax(rdm), rdm.shape)
        d = 2 * float(rindex)* delta_r - OFFSETS[q]
        v = float(vindex-(Mc*16)//2)*delta_v
        d_q.append(d)
        v_q.append(v)
    # print(v_q)
    d_q = np.array(d_q)
    v_q = np.array(v_q)

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
        dp = (kalman_xp[0]**2+kalman_xp[1]**2)**(1/2) + ((kalman_xp[0]-ANTENNA_POS[q][0])**2 + (kalman_xp[1]-ANTENNA_POS[q][1])**2)**(1/2)
        vp = 0.5 * ((np.array([kalman_xp[0]-ANTENNA_POS[q][0], kalman_xp[1]-ANTENNA_POS[q][1]]) @ kalman_xp[2:])*(1/np.sqrt((kalman_xp[0]-ANTENNA_POS[q][0])**2+(kalman_xp[1]-ANTENNA_POS[q][1])**2)) +
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
                d = (x**2+y**2)**(1/2) + ((x-ANTENNA_POS[q][0])**2 + (y-ANTENNA_POS[q][1])**2)**(1/2)
                res.append(dmes-d)
            return res
        
        x0 = [0.0,0.0]
        point_est = least_squares(diff,x0,loss='cauchy').x
        # print(point_est)

        N = np.fromfunction(
            lambda q, i: 0.5 * (
                (point_est[i.astype(int)] - ANTENNA_POS[q.astype(int), i.astype(int)]) *
                (1 / np.sqrt((point_est[0] - ANTENNA_POS[q.astype(int), 0])**2 + (point_est[1] - ANTENNA_POS[q.astype(int), 1])**2)) +
                point_est[i.astype(int)] *
                (1 / np.sqrt(point_est[0]**2 + point_est[1]**2))
            ),
            (np.sum(takeRDM), 2),
            dtype=int
        )

        speed_est = np.linalg.inv(N.T @ N) @ N.T @ v_q
        # print(speed_est)
        z = np.concatenate((point_est,speed_est))

    kalman_x = kalman_xp + kalman_K @ (z - kalman_H @ kalman_xp)
    kalman_P = kalman_Pp - kalman_K @ kalman_H @ kalman_Pp

    return kalman_x, kalman_P

# ===========================================================================
# Partie 3 : Multicible Bistatique
# ===========================================================================

def ca_cfar_convolve(rdm, guard_size_doppler=10, guard_size_range=11,
                     window_size_doppler=45, window_size_range=15, alpha=8.0):
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
                              (total_range // 2, total_range // 2)), mode='wrap')
    training_sums = fftconvolve(padded_rdm, kernel_cfar, mode='valid')

    # Compute thresholds
    thresholds = alpha * (training_sums / n_cfar_cells)

    # Generate detection mask
    mask = rdm > thresholds

    return mask, thresholds

def extract_targets_from_cfar(rdm, mask,
                              min_distance_doppler=20,
                              min_distance_range=10,
                              min_points_per_target=150):
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

    start_time = time.time()
    mask, thresholds = ca_cfar_convolve(
        rdm, 
    )
    end_time = time.time()
    print("Execution time convolve ca_cfar:", end_time - start_time, "seconds")

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

def compute_track_position_and_speed(track):
    d_q = [2*track[i][1] for i in range(len(track))] #*2 car r -> d
    v_q = [track[i][0] for i in range(len(track))]
    def diff(p):
        x,y = p
        res = []
        for q,dmes in enumerate(d_q):
            d = (x**2+y**2)**(1/2) + ((x-ANTENNA_POS[q][0])**2 + (y-ANTENNA_POS[q][1])**2)**(1/2)
            res.append(dmes-d)
        return res
    
    x0 = [0.0,0.0]
    point_est = least_squares(diff,x0,loss='linear').x

    speed_est = [None, None]
    if (point_est[0] != 0.0 and point_est[1] != 0.0):
        N = np.fromfunction(
            lambda q, i: 0.5 * (
                (point_est[i.astype(int)] - ANTENNA_POS[q.astype(int), i.astype(int)]) *
                (1 / np.sqrt((point_est[0] - ANTENNA_POS[q.astype(int), 0])**2 + (point_est[1] - ANTENNA_POS[q.astype(int), 1])**2)) +
                point_est[i.astype(int)] *
                (1 / np.sqrt(point_est[0]**2 + point_est[1]**2))
            ),
            (len(track), 2),
            dtype=int
        )

        speed_est = np.linalg.inv(N.T @ N) @ N.T @ v_q

    return [(point_est[0],point_est[1]),(speed_est[0],speed_est[1])]


from itertools import product
deltaTFrame = Mc * Tc
kalman_params = {
    'F' : np.array([[1,0,deltaTFrame,0],[0,1,0,deltaTFrame],[0,0,1,0],[0,0,0,1]]),
    'Q' : np.array([[0,0,0,0],[0,0,0,0],[0,0,0.1,0],[0,0,0,0.1]]),
    'H' : np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
    'R' : np.array([[3,0,0,0],[0,3,0,0],[0,0,5,0],[0,0,0,5]])
}

class tracker:
    def __init__(self, id, kalman_x, kalman_P, history) :
        self.id = id
        self.kalman_xp = None
        self.kalman_Pp = None
        self.kalman_x = kalman_x
        self.kalman_P = kalman_P
        self.non_official_count = 0
        self.misses = 0
        self.official_count = 0
        self.history = history 
    
    def kalman_predict(self):
        self.kalman_xp = kalman_params['F'] @ self.kalman_x
        self.kalman_Pp = kalman_params['F'] @ self.kalman_P @ kalman_params['F'].T + kalman_params['Q']
        return self.kalman_xp, self.kalman_Pp
    
    def kalman_update(self, z):
        kalman_K = self.kalman_Pp @ kalman_params['H'].T @ np.linalg.inv(kalman_params['H'] @ self.kalman_P @ kalman_params['H'].T + kalman_params['R'])
        self.kalman_x = self.kalman_xp + kalman_K @ (z - kalman_params['H'] @ self.kalman_xp)
        self.kalman_P = self.kalman_Pp - kalman_K @ kalman_params['H'] @ self.kalman_Pp
        return self.kalman_x, self.kalman_P
    
    def __str__(self):
        return f"Tracker ID: {self.id}, Kalman State: {self.kalman_x}, Kalman Covariance: {self.kalman_P}, Kalman Predicted State: {self.kalman_xp}, Kalman Predicted Covariance: {self.kalman_Pp}, History: {self.history}, Misses: {self.misses}, Non-official Count: {self.non_official_count}, Official Count: {self.official_count}"

def extract_all_targets(RDM_frame):
    """Retourne une liste à 4 entrées (une par canal) contenant
    soit la liste [(v, r), …] soit [None] si pas de cible."""
    all_t = [[] for _ in range(4)]
    for ch in range(4):
        mask, _ = ca_cfar_convolve(RDM_frame[ch])
        targets = extract_targets_from_cfar(RDM_frame[ch], mask)
        for (v_idx, r_idx) in (t[0] for t in targets):
            r = float(r_idx) * delta_r - OFFSETS[ch]
            v = float(v_idx-(Mc*16)//2)*delta_v
            all_t[ch].append((v, r))
    # remplace les listes vides par [None] pour que product() les traite
    return [chan if chan else [None] for chan in all_t]

def make_intraframe_tracks(all_targets):
    """Associe entre eux les canaux d’une même frame.
       Une track = liste de (v, r) ayant >=2 canaux renseignés."""
    tracks = []
    for combo in product(*all_targets):          # 4‑uplet (ou None)
        track = [combo[ch] for ch in range(4) if combo[ch] is not None]
        acceptableTrack = True
        track_pos_speed = compute_track_position_and_speed(track)
        if(track_pos_speed[0][0] == 0.0 and track_pos_speed[0][1] == 0.0 and track_pos_speed[1][0] == None and track_pos_speed[1][1] == None):
            acceptableTrack = False
        if 2 <= len(track) <= 4 and acceptableTrack:
            tracks.append(track_pos_speed)
    return tracks

    
def tracking_init(file) :
    RDM = compute_RDM(file, 0)
    all_tragets = extract_all_targets(RDM)
    tracks = make_intraframe_tracks(all_tragets)
    print("Nombre de tracks : ", len(tracks))
    print(tracks)
    non_official = []
    for i, track in enumerate(tracks):
        non_official.append(tracker(i, np.array([track[0][0], track[0][1], track[1][0], track[1][1]]), np.eye(4), [track]))
    return non_official


from scipy.spatial.distance import cdist

MAX_GATING_DIST_4D = 3.0                          # seuil en 4‑D

def nearest_neighbor(non_official, frame_idx, file, official=None):
    # 1. ---------- prédiction ----------
    all_trk = non_official + (official or [])
    for trk in all_trk:
        trk.kalman_predict()

    # 2. ---------- extraction des mesures ----------
    RDM     = compute_RDM(file, frame_idx)
    tracks  = make_intraframe_tracks(extract_all_targets(RDM))   # [(pos),(vel)]
    if not tracks:
        return

    # 3. ---------- matrices 4‑D ----------

    pred_state = np.vstack([trk.kalman_xp for trk in all_trk])     # (N,4)
    meas_state = np.vstack([[p[0], p[1], v[0], v[1]] for p, v in tracks])  # (M,4)

    D = cdist(pred_state, meas_state)                              # (N,M)

    # 4. ---------- association NN ----------
    assigned_cols = set()
    for i, row in enumerate(D):
        j = np.argmin(row)
        if j in assigned_cols:
            all_trk[i].misses += 1
            continue

        if row[j] < MAX_GATING_DIST_4D:
            z = meas_state[j]
            all_trk[i].kalman_update(z)
            all_trk[i].history.append(tracks[j])
            assigned_cols.add(j)
        else:
            all_trk[i].misses += 1

        # ---------- compteurs ----------
        if all_trk[i] in non_official:
            all_trk[i].non_official_count += 1
        else:                       # appartient à la liste official
            all_trk[i].official_count += 1


    
non_official = tracking_init("data/30-04/marche 2-15m.npz")
print("Initialisation des tracks : ", non_official[0])
print("Initialisation des tracks 2: ", non_official[1])
nearest_neighbor(non_official, 1, "data/30-04/marche 2-15m.npz")
print("nearest neighbor : ", non_official[0])
print("nearest neighbor 2 : ", non_official[1])