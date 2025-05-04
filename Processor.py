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

OFFSETS = np.asarray([7.33982036, 7.3548503, 10.86700599, 3.64670659])
ANTENNA_POS = np.asarray([
    [-0.35, 2.7],     # Channel 0 (common focus)
    [1.8, 0.5],   # Channel 1
    [5.1, -2.3],    # Channel 2
    [0.0, 0.0],     # Channel 3
])
OFFSETS = np.asarray([ 7.33982036,  7.3548503 , 10.86700599,  3.64670659])
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

def os_cfar(rdm, guard_size_doppler=3, guard_size_range=2, 
                          window_size_doppler=12, window_size_range=8, 
                          alpha=2.0, ordered_statistic_idx=0.75):
    """
    Implémentation d'un OS-CFAR 2D avec emphase sur les directions orthogonales
    pour mieux discriminer les cibles des lobes secondaires.
    
    Arguments:
    - rdm: Matrice Range-Doppler à traiter
    - guard_size_doppler/range: Taille de la zone de garde dans chaque direction
    - window_size_doppler/range: Taille de la fenêtre dans chaque direction
    - alpha: Facteur multiplicatif pour le seuil
    - ordered_statistic_idx: Indice normalisé [0,1] pour l'OS-CFAR
    
    Retourne:
    - mask: Masque binaire des détections
    - thresholds: Matrice des seuils calculés
    """
    n_doppler, n_range = rdm.shape
    mask = np.zeros_like(rdm, dtype=bool)
    thresholds = np.zeros_like(rdm, dtype=float)
    
    # Calcul des paramètres pour l'OS-CFAR
    total_cells_orthogonal = 2 * (window_size_doppler + window_size_range)
    os_idx = int(np.floor(total_cells_orthogonal * ordered_statistic_idx))
    
    # Parcours de tous les points de la matrice
    for d in range(n_doppler):
        for r in range(n_range):
            # Échantillons orthogonaux: colonnes et lignes exclusivement
            samples = []
            
            # Échantillons en Doppler (verticaux)
            d_start = max(0, d - window_size_doppler - guard_size_doppler)
            d_guard_start = max(0, d - guard_size_doppler)
            d_guard_end = min(n_doppler, d + guard_size_doppler + 1)
            d_end = min(n_doppler, d + window_size_doppler + guard_size_doppler + 1)
            
            # Partie supérieure (Doppler négatif)
            samples.extend(rdm[d_start:d_guard_start, r].tolist())
            # Partie inférieure (Doppler positif)
            samples.extend(rdm[d_guard_end:d_end, r].tolist())
            
            # Échantillons en Range (horizontaux)
            r_start = max(0, r - window_size_range - guard_size_range)
            r_guard_start = max(0, r - guard_size_range)
            r_guard_end = min(n_range, r + guard_size_range + 1)
            r_end = min(n_range, r + window_size_range + guard_size_range + 1)
            
            # Partie gauche
            samples.extend(rdm[d, r_start:r_guard_start].tolist())
            # Partie droite
            samples.extend(rdm[d, r_guard_end:r_end].tolist())
            
            # Si on n'a pas assez d'échantillons, ajuster l'indice OS
            actual_os_idx = min(os_idx, len(samples) - 1) if samples else 0
            
            # Calcul du seuil OS-CFAR
            if samples:
                samples.sort()
                threshold = alpha * samples[actual_os_idx]
                thresholds[d, r] = threshold
                
                # Détection
                if rdm[d, r] > threshold:
                    # Vérification supplémentaire pour rejeter les lobes secondaires
                    is_lobe = False
                    
                    # Vérifier l'asymétrie en Doppler (typique des lobes secondaires)
                    if d_guard_start > 0 and d_guard_end < n_doppler:
                        up_vals = rdm[d_guard_start-1:d_guard_start, r]
                        down_vals = rdm[d_guard_end:d_guard_end+1, r]
                        
                        # Si forte asymétrie (un côté beaucoup plus fort que l'autre)
                        if np.mean(up_vals) > 2 * rdm[d, r] or np.mean(down_vals) > 2 * rdm[d, r]:
                            is_lobe = True
                    
                    # Vérifier l'asymétrie en Range
                    if r_guard_start > 0 and r_guard_end < n_range:
                        left_vals = rdm[d, r_guard_start-1:r_guard_start]
                        right_vals = rdm[d, r_guard_end:r_guard_end+1]
                        
                        # Si forte asymétrie (un côté beaucoup plus fort que l'autre)
                        if np.mean(left_vals) > 2 * rdm[d, r] or np.mean(right_vals) > 2 * rdm[d, r]:
                            is_lobe = True
                    
                    # Accepter uniquement si ce n'est pas un lobe secondaire
                    mask[d, r] = not is_lobe
    
    return mask, thresholds

def ca_cfar(rdm, guard_size_doppler=3, guard_size_range=2, window_size_doppler=12, window_size_range=8,
                          alpha=2.0, ordered_statistic_idx=0.75):
    """
    Implémentation d'un OS-CFAR 2D avec emphase sur les directions orthogonales
    pour mieux discriminer les cibles des lobes secondaires.
    
    Arguments:
    - rdm: Matrice Range-Doppler à traiter
    - guard_size_doppler/range: Taille de la zone de garde dans chaque direction
    - window_size_doppler/range: Taille de la fenêtre dans chaque direction
    - alpha: Facteur multiplicatif pour le seuil
    - ordered_statistic_idx: Indice normalisé [0,1] pour l'OS-CFAR
    
    Retourne:
    - mask: Masque binaire des détections
    - thresholds: Matrice des seuils calculés
    """
    n_doppler, n_range = rdm.shape
    mask = np.zeros_like(rdm, dtype=bool)
    thresholds = np.zeros_like(rdm, dtype=float)
    data = np.copy(rdm)
    print(rdm.shape)
    print(guard_size_doppler, guard_size_range, window_size_doppler, window_size_range)

    data = np.pad(data, ((guard_size_doppler+window_size_doppler,guard_size_doppler+window_size_doppler),(guard_size_range+window_size_range,guard_size_range+window_size_range)), 'wrap')
    print(data.shape)
    area_covered = (2*(window_size_doppler+guard_size_doppler)+1)*(2*(window_size_range+guard_size_range)+1) - (2*guard_size_doppler+1)*(2*guard_size_range+1)

    for d in range(n_doppler):
        # window_mean = np.sum(data[d:d+2*(guard_size_doppler+window_size_doppler)+1,0:2*(guard_size_range+window_size_range)+1])/area_covered
        # guard_mean = np.sum(data[d+window_size_doppler:d+window_size_doppler+2*guard_size_doppler+1,window_size_range:window_size_range+2*guard_size_range+1])/area_covered
        for r in range(n_range):
            # mean = window_mean - guard_mean
            # threshold = alpha * mean
            # thresholds[d, r] = threshold

            actual_mean = (np.sum(data[d:d+2*(guard_size_doppler+window_size_doppler)+1,r:r+2*(guard_size_range+window_size_range)+1])
                           -np.sum(data[d+window_size_doppler:d+window_size_doppler+2*guard_size_doppler+1,r+window_size_range:r+window_size_range+2*guard_size_range+1]))/area_covered
            threshold = alpha * actual_mean
            thresholds[d, r] = threshold

            # if(mean - actual_mean > 0.1):
            #     print("mean: ", mean, "actual_mean: ", actual_mean)

            # Détection
            if rdm[d, r] > threshold:
                mask[d, r] = True

            # window_mean -= np.sum(data[d:d+2*(window_size_doppler+guard_size_doppler)+1,r:r+1])/area_covered
            # window_mean += np.sum(data[d:d+2*(guard_size_doppler+window_size_doppler)+1,r+2*(guard_size_range+window_size_range)+1:r+2*(guard_size_range+window_size_range)+2])/area_covered
            # guard_mean -= np.sum(data[d+window_size_doppler:d+window_size_doppler+2*guard_size_doppler+1,r+window_size_range:r+window_size_range+1])/area_covered
            # guard_mean += np.sum(data[d+window_size_doppler:d+window_size_doppler+2*guard_size_doppler+1,r+window_size_range+2*guard_size_range+1:r+window_size_range+2*guard_size_range+2])/area_covered

    # Parcours de tous les points de la matrice
    # for d in range(n_doppler):
        # for r in range(n_range):
            # Échantillons orthogonaux: colonnes et lignes exclusivement
            # samples = np.array([])
            # 
            # Échantillons en Doppler (verticaux)
            # d_start = max(0, d - window_size_doppler - guard_size_doppler)
            # d_guard_start = max(0, d - guard_size_doppler)
            # d_guard_end = min(n_doppler, d + guard_size_doppler + 1)
            # d_end = min(n_doppler, d + window_size_doppler + guard_size_doppler + 1)
            # 
            # 
            # Échantillons en Range (horizontaux)
            # r_start = max(0, r - window_size_range - guard_size_range)
            # r_guard_start = max(0, r - guard_size_range)
            # r_guard_end = min(n_range, r + guard_size_range + 1)
            # r_end = min(n_range, r + window_size_range + guard_size_range + 1)
            # 
            # samples = np.append(samples, rdm[d_start:d_guard_start, r_start:r_end].flatten())
            # samples = np.append(samples, rdm[d_guard_end:d_end, r_start:r_end].flatten())
            # samples = np.append(samples, rdm[d_guard_start:d_guard_end, r_start:r_guard_start].flatten())
            # samples = np.append(samples, rdm[d_guard_start:d_guard_end, r_guard_end:r_end].flatten())

            # Si on n'a pas assez d'échantillons, ajuster l'indice OS
            # os_idx = int(np.floor(len(samples) * ordered_statistic_idx))
            # actual_os_idx = min(os_idx, len(samples) - 1) if samples.size != 0 else 0
            # 
            # Calcul du seuil OS-CFAR
            # if samples.size != 0:
                # np.sort(samples, kind='mergesort')
                # threshold = alpha * samples[actual_os_idx]
                # thresholds[d, r] = threshold
                # threshold = alpha * np.mean(samples)
                # thresholds[d, r] = threshold
                # 
                # Détection
                # if rdm[d, r] > threshold:
                    # Vérification supplémentaire pour rejeter les lobes secondaires
                    # is_lobe = False
                    # 
                    # Vérifier l'asymétrie en Doppler (typique des lobes secondaires)
                    # if d_start > 0 and d_end < n_doppler:
                        # up_vals = rdm[d_start:d, r]
                        # down_vals = rdm[d+1:d_end, r]
                        # 
                        # Si forte asymétrie (un côté beaucoup plus fort que l'autre)
                        # if np.mean(up_vals) > 2 * rdm[d, r] or np.mean(down_vals) > 2 * rdm[d, r]:
                            # is_lobe = True
                    # 
                    # Vérifier l'asymétrie en Range
                    # if r_start > 0 and r_end < n_range:
                        # left_vals = rdm[d, r_start:r]
                        # right_vals = rdm[d, r+1:r_end]
                        # 
                        # Si forte asymétrie (un côté beaucoup plus fort que l'autre)
                        # if np.mean(left_vals) > 2 * rdm[d, r] or np.mean(right_vals) > 2 * rdm[d, r]:
                            # is_lobe = True
                    # 
                    # Accepter uniquement si ce n'est pas un lobe secondaire
                    # mask[d, r] = not is_lobe
    
    return mask, thresholds

def ca_cfar_convolve(rdm, guard_size_doppler=3, guard_size_range=2,
                     window_size_doppler=12, window_size_range=8, alpha=2.0):
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

def osca_cfar(rdm, guard_size_doppler=3, guard_size_range=2, window_size_doppler=12, window_size_range=8, 
                          alpha=2.0, ordered_statistic_idx=0.75):
    """
    Implémentation d'un OS-CFAR 2D avec emphase sur les directions orthogonales
    pour mieux discriminer les cibles des lobes secondaires.
    
    Arguments:
    - rdm: Matrice Range-Doppler à traiter
    - guard_size_doppler/range: Taille de la zone de garde dans chaque direction
    - window_size_doppler/range: Taille de la fenêtre dans chaque direction
    - alpha: Facteur multiplicatif pour le seuil
    - ordered_statistic_idx: Indice normalisé [0,1] pour l'OS-CFAR
    
    Retourne:
    - mask: Masque binaire des détections
    - thresholds: Matrice des seuils calculés
    """
    n_doppler, n_range = rdm.shape
    mask = np.zeros_like(rdm, dtype=bool)
    thresholds = np.zeros_like(rdm, dtype=float)
    #print(rdm.shape)
    
    
    # Parcours de tous les points de la matrice
    for d in range(n_doppler):
        for r in range(n_range):
            
            # Échantillons en Doppler (verticaux)
            d_start = max(0, d - window_size_doppler - guard_size_doppler)
            d_guard_start = max(0, d - guard_size_doppler)
            d_guard_end = min(n_doppler, d + guard_size_doppler + 1)
            d_end = min(n_doppler, d + window_size_doppler + guard_size_doppler + 1)
            
            
            # Échantillons en Range (horizontaux)
            r_start = max(0, r - window_size_range - guard_size_range)
            r_guard_start = max(0, r - guard_size_range)
            r_guard_end = min(n_range, r + guard_size_range + 1)
            r_end = min(n_range, r + window_size_range + guard_size_range + 1)

            sample1 = rdm[d_start:d_guard_start, r_start:r_end]
            sample1 = np.sort(sample1, kind='mergesort')
            sample2 = rdm[d_guard_end:d_end, r_start:r_end]
            sample2 = np.sort(sample2, kind='mergesort')
            sample3 = np.append(rdm[d_guard_start:d_guard_end, r_start:r_guard_start], rdm[d_guard_start:d_guard_end, r_guard_end:r_end], axis=-1)
            sample3 = np.sort(sample3, kind='mergesort')
            
            # Calcul des paramètres pour l'OS-CFAR
            total_cells12 = r_end-r_start
            os_idx12 = int(np.floor(total_cells12 * ordered_statistic_idx))
            actual_os_idx12 = min(os_idx12, len(sample1[0]) - 1) if sample1.size != 0 else 0
            total_cells3 = d_end-d_guard_end + d_guard_start-d_start
            os_idx3 = int(np.floor(total_cells3 * ordered_statistic_idx))
            actual_os_idx3 = min(os_idx3, len(sample3[0]) - 1) if sample3.size != 0 else 0

            mean = 0
            for i in range(len(sample1)):
                mean += sample1[i][actual_os_idx12]
            for i in range(len(sample2)):
                mean += sample2[i][actual_os_idx12]
            for i in range(len(sample3)):
                mean += sample3[i][actual_os_idx3]
            mean /= (len(sample1) + len(sample2) + len(sample3))

            threshold = alpha*mean
            thresholds[d, r] = threshold
            if rdm[d, r] > threshold:
                # Vérification supplémentaire pour rejeter les lobes secondaires
                is_lobe = False
                
                # Vérifier l'asymétrie en Doppler (typique des lobes secondaires)
                if d_start > 0 and d_end < n_doppler:
                    up_vals = rdm[d_start:d, r]
                    down_vals = rdm[d+1:d_end, r]
                    
                    # Si forte asymétrie (un côté beaucoup plus fort que l'autre)
                    if np.mean(up_vals) > 2 * rdm[d, r] or np.mean(down_vals) > 2 * rdm[d, r]:
                        is_lobe = True
                
                # Vérifier l'asymétrie en Range
                if r_start > 0 and r_end < n_range:
                    left_vals = rdm[d, r_start:r]
                    right_vals = rdm[d, r+1:r_end]
                    
                    # Si forte asymétrie (un côté beaucoup plus fort que l'autre)
                    if np.mean(left_vals) > 2 * rdm[d, r] or np.mean(right_vals) > 2 * rdm[d, r]:
                        is_lobe = True
                
                # Accepter uniquement si ce n'est pas un lobe secondaire
                mask[d, r] = not is_lobe
    
    return mask, thresholds

def extract_targets_from_cfar(rdm, mask, min_distance_doppler=20, min_distance_range=15):
    """
    Extrait les cibles à partir du masque CFAR en fusionnant les détections proches.
    
    Arguments:
    - rdm: Matrice Range-Doppler d'origine
    - mask: Masque binaire des détections CFAR
    - min_distance_doppler/range: Distance minimale entre deux cibles distinctes
    
    Retourne:
    - targets: Liste de tuples ((d, r), amplitude)
    """
    # Trouver tous les points détectés
    detected_points = np.argwhere(mask)
    targets = []
    
    # Si aucune détection, retourner liste vide
    if len(detected_points) == 0:
        return targets
    
    # Trier les points par amplitude décroissante
    amplitudes = [rdm[d, r] for d, r in detected_points]
    sorted_indices = np.argsort(amplitudes)[::-1]
    
    processed_mask = np.zeros_like(mask, dtype=bool)
    
    for idx in sorted_indices:
        d, r = detected_points[idx]
        
        # Vérifier si ce point n'a pas déjà été traité (fusion)
        if not processed_mask[d, r]:
            amp = rdm[d, r]
            targets.append(((d, r), amp))
            
            # Marquer tous les points proches comme traités
            d_min = max(0, d - min_distance_doppler)
            d_max = min(rdm.shape[0], d + min_distance_doppler + 1)
            r_min = max(0, r - min_distance_range)
            r_max = min(rdm.shape[1], r + min_distance_range + 1)
            
            # Zone elliptique de fusion
            for dd in range(d_min, d_max):
                for rr in range(r_min, r_max):
                    # Condition elliptique
                    if ((dd - d) / min_distance_doppler)**2 + ((rr - r) / min_distance_range)**2 <= 1:
                        processed_mask[dd, rr] = True
    
    return targets

def cfar_2d_adaptive(rdm, guard_size=(3, 2), window_size=(12, 8), alpha=1.5, 
                     use_os_cfar=True, os_percentile=75, directional_weights=(0.6, 0.4),
                     min_distance=(5, 3)):
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
    - directional_weights: Pondération (doppler, range) pour l'analyse directionnelle
    - min_distance: Distance minimale entre cibles pour fusion
    
    Retourne:
    - targets: Liste de tuples ((d, r), amplitude)
    - mask: Masque binaire des détections
    - thresholds: Matrice des seuils calculés
    """
    # Convertir le percentile en indice normalisé [0,1]
    os_idx = os_percentile / 100.0

    # Utiliser OS-CFAR avec analyse orthogonale
    mask, thresholds = os_cfar(
        rdm, 
        guard_size_doppler=guard_size[0], 
        guard_size_range=guard_size[1],
        window_size_doppler=window_size[0], 
        window_size_range=window_size[1],
        alpha=alpha, 
        ordered_statistic_idx=os_idx
    )
    
    # Extraction et fusion des cibles
    targets = extract_targets_from_cfar(
        rdm, 
        mask, 
        min_distance_doppler=min_distance[0], 
        min_distance_range=min_distance[1]
    )
    
    return targets, mask, thresholds

def create_star_shaped_footprint(window_size):
    """
    Creates a star-shaped footprint with the given window size.
    
    The footprint is created with 1's along the central row and column 
    and 0's elsewhere. This creates a cross-shaped or star-shaped mask.
    
    Arguments:
    - window_size: Tuple (doppler, range) representing the size of the window.
    
    Returns:
    - footprint: A 2D numpy array representing the star-shaped footprint.
    """
    doppler_size, range_size = window_size

    # Create a zero matrix with the given window size
    footprint = np.zeros((doppler_size, range_size), dtype=int)
    
    # Set the middle row and middle column to 1's to form a star shape
    center_doppler = doppler_size // 2
    center_range = range_size // 2
    
    footprint[center_doppler, :] = 1  # Set the central row to 1's
    footprint[center_doppler-1, :] = 1  
    footprint[center_doppler+1, :] = 1  
    footprint[center_doppler-2, :] = 1  
    footprint[center_doppler+2, :] = 1  
    footprint[center_doppler-3, :] = 1  
    footprint[center_doppler+3, :] = 1  
    footprint[center_doppler-4, :] = 1  
    footprint[center_doppler+4, :] = 1  
    footprint[:, center_range] = 1    # Set the central column to 1's
    footprint[:, center_range-1] = 1
    footprint[:, center_range+1] = 1
    footprint[:, center_range-2] = 1
    footprint[:, center_range+2] = 1
    footprint[:, center_range-3] = 1
    footprint[:, center_range+3] = 1
    footprint[:, center_range-4] = 1
    footprint[:, center_range+4] = 1

    return footprint

def cfar_2d(rdm, guard_size, window_size, alpha, 
                     use_os_cfar, os_percentile,
                     min_distance):
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
    # Convertir le percentile en indice normalisé [0,1]
    os_idx = os_percentile / 100.0

    start_time = time.time()
    mask, thresholds = ca_cfar_convolve(
        rdm, 
        guard_size_doppler=guard_size[0],
        guard_size_range=guard_size[1],
        window_size_doppler=window_size[0], 
        window_size_range=window_size[1],
        alpha=alpha
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
        min_distance_doppler=min_distance[0], 
        min_distance_range=min_distance[1]
    )
    
    return targets, mask, thresholds

def cfar_2d_osca(rdm, guard_size, window_size, alpha, 
                     use_os_cfar, os_percentile,
                     min_distance):
    # Convertir le percentile en indice normalisé [0,1]
    os_idx = os_percentile / 100.0

    # Utiliser OS-CFAR avec analyse orthogonale
    mask, thresholds = osca_cfar(
        rdm, 
        guard_size_doppler=guard_size[0],
        guard_size_range=guard_size[1],
        window_size_doppler=window_size[0], 
        window_size_range=window_size[1],
        alpha=alpha, 
        ordered_statistic_idx=os_idx
    )
    
    # Extraction et fusion des cibles
    targets = extract_targets_from_cfar(
        rdm, 
        mask, 
        min_distance_doppler=min_distance[0], 
        min_distance_range=min_distance[1]
    )
    
    return targets, mask, thresholds

def visualize_cfar_results(rdm, mask, thresholds, targets=None, figsize=(12, 10)):
    """
    Visualisation des résultats CFAR: données originales, seuil et détections.
    
    Arguments:
    - rdm: Matrice Range-Doppler originale
    - mask: Masque binaire des détections
    - thresholds: Matrice des seuils calculés
    - targets: Liste optionnelle des cibles extraites
    - figsize: Taille de la figure
    """

    rdm_display = rdm
    thresholds_display = thresholds
    vmin = None
    vmax = None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Affichage RDM originale
    im0 = axes[0, 0].imshow(rdm_display, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('RDM Originale')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Affichage des seuils CFAR
    im1 = axes[0, 1].imshow(thresholds_display, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Seuils CFAR')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Affichage des détections
    im2 = axes[1, 0].imshow(rdm_display, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 0].contour(mask, colors='r', linewidths=1)
    axes[1, 0].set_title('Détections CFAR')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Affichage RDM avec cibles extraites
    im3 = axes[1, 1].imshow(rdm_display, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Cibles Extraites')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # Afficher les cibles si fournies
    if targets:
        for (d, r), amp in targets:
            axes[1, 1].plot(r, d, 'rx', markersize=10)
            amp_text = f"{amp:.2e}"
            axes[1, 1].annotate(amp_text, (r+2, d+2), color='white', fontsize=8)
    
    # Étiquettes communes
    for ax in axes.flat:
        ax.set_xlabel('Range (bins)')
        ax.set_ylabel('Doppler (bins)')
    
    plt.tight_layout()
    return fig

# Paramètres WORK IN PROGRESS
guard_size = (30, 30)  # (doppler, range)
window_size = (10, 20)  # (doppler, range)
alpha = 5 # Facteur multiplicatif du seuil
use_os_cfar = True # True pour OS-CFAR, False pour CA-CFAR
os_percentile = 99  # Percentile pour OS-CFAR
