import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

# ===========================================================================
# Paramètres physiques 
# ===========================================================================

OFFSETS = np.asarray([2.28, 4.06, 3.93, 5.75])
ANTENNA_POS = np.asarray([
    [0.0, 0.0],     # Channel 0 (common focus)
    [-2.55, 1.6],   # Channel 1
    [2.05, 0.0],    # Channel 2
    [2.4, 0.9],     # Channel 3
])

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
        bins_5m = int(np.round(5.0 / delta_r))
        # on ne garde que la moitié de l'axe des distances
        rdm = rdm[:, : (PAD_R * Ms)//2 - bins_5m]

        # décalage pour compenser l'offset
        offset_bins = int(np.round(OFFSETS[ch] / (3e8 / (2 * B / PAD_R))))
        rdm = np.roll(rdm, -offset_bins, axis=1)
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
        dist = max(r_idx * delta_r - OFFSETS[ch], 0.0)
        distances.append(2*dist)

    def resid(p):
        x, y = p
        r_tx = np.hypot(x - ANTENNA_POS[0,0], y - ANTENNA_POS[0,1])
        res = []
        for ch in range(0, 3):
            r_rx = np.hypot(x - ANTENNA_POS[ch,0], y - ANTENNA_POS[ch,1])
            # somme de TX→C + C→RX
            res.append((r_tx + r_rx) - distances[ch])
        return res


    p0 = np.mean(ANTENNA_POS, axis=0)
    sol = least_squares(resid, p0, method='lm')
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

# ===========================================================================
# Partie 3 : Multicible Bistatique
# ===========================================================================

def dirichlet_kernel(Ndata, Nfft, k):
    k = np.asarray(k, dtype=float)
    phase = np.exp(-1j * np.pi * (Ndata - 1) * k / Nfft)
    ratio = Ndata * np.sinc(k * Ndata / Nfft) / np.sinc(k / Nfft)
    return phase * ratio

def psf_rectangular(Ms, Mc, N_r_half, N_d):
    # returns a (N_d × 2*N_r_half) PSF, pre–half‑range cropping
    Dk = dirichlet_kernel(Ms, 2*N_r_half, np.arange(2*N_r_half))
    Dl = dirichlet_kernel(Mc, N_d,       np.arange(N_d))
    B  = np.outer(Dl, Dk)
    return np.fft.fftshift(B, axes=0)

def center_psf(psf):
    """
    Centre automatiquement la PSF empirique sur son maximum.
    """
    # Trouver l'indice du pic maximum
    peak_idx = np.unravel_index(np.argmax(np.abs(psf)), psf.shape)

    # Calcul du déplacement nécessaire pour centrer le pic
    center = (psf.shape[0] // 2, psf.shape[1] // 2)
    shift_amount = (center[0] - peak_idx[0], center[1] - peak_idx[1])

    # Décalage pour recentrer exactement le pic
    psf_centered = np.roll(psf, shift=shift_amount, axis=(0, 1))

    return psf_centered

def gaussian_window(shape, center, sigma_doppler=11.851385129604585, sigma_range=3.922807157442878):
    """
    Génère une fenêtre gaussienne centrée sur `center`.
    """
    D, R = np.indices(shape)
    gauss = np.exp(-(((D - center[0])**2)/(2*sigma_doppler**2) 
                     + ((R - center[1])**2)/(2*sigma_range**2)))
    return gauss
#
def clean_rdm(rdm, psf_full, threshold=0.1, max_iter=6, 
              sigma_doppler = 6.304204033891018, sigma_range = 5.145518693492693):
    rdm_clean = np.copy(rdm)
    targets = []

    # Normalisation préalable de la PSF
    psf_full = psf_full / np.max(psf_full)

    for _ in range(max_iter):
        peak_value = np.max(rdm_clean)
        if peak_value < threshold * np.max(rdm):
            break

        peak_idx = np.unravel_index(np.argmax(rdm_clean), rdm_clean.shape)
        targets.append((peak_idx, peak_value))

        # Décalage de la PSF
        shift_amount = (peak_idx[0] - psf_full.shape[0] // 2,
                        peak_idx[1] - psf_full.shape[1] // 2)
        psf_shifted = np.roll(psf_full, shift=shift_amount, axis=(0, 1))

        # Facteur optimal par moindres carrés
        scale_factor = np.sum(rdm_clean * psf_shifted) / np.sum(psf_shifted ** 2)

        # Soustraction précise
        rdm_clean -= scale_factor * psf_shifted

        # Fenêtre gaussienne douce autour du pic détecté
        gauss_win = gaussian_window(
            rdm_clean.shape, 
            peak_idx, 
            sigma_doppler=sigma_doppler, 
            sigma_range=sigma_range
        )

        # Suppression douce locale (gaussienne)
        rdm_clean *= (1 - gauss_win)

        # Forcer la positivité finale
        rdm_clean = np.maximum(rdm_clean, 0)

    return targets, rdm_clean

def fuse_targets(targets, max_dist_doppler=47, max_dist_range=38):
    """
    Regroupe les cibles proches par fusion pondérée avec des seuils distincts
    pour Doppler et Range.
    
    - targets: liste de ((doppler_idx, range_idx), amplitude)
    - max_dist_doppler: distance maximale en Doppler (en bins)
    - max_dist_range: distance maximale en Range (en bins)
    """
    targets = targets.copy()
    i = 0
    while i < len(targets):
        merged = False
        (di, ri), ai = targets[i]
        j = i + 1
        while j < len(targets):
            (dj, rj), aj = targets[j]
            dd = abs(di - dj)
            dr = abs(ri - rj)
            if dd < max_dist_doppler and dr < max_dist_range:
                # Moyenne pondérée par amplitude
                total_amp = ai + aj
                new_d = (di * ai + dj * aj) / total_amp
                new_r = (ri * ai + rj * aj) / total_amp
                new_target = ((new_d, new_r), total_amp)

                # Supprimer les deux anciennes cibles
                targets.pop(j)
                targets.pop(i)

                # Ajouter la nouvelle cible à la fin (à reconsidérer plus tard)
                targets.append(new_target)
                merged = True
                break  # redémarrage de i

            else:
                j += 1

        if not merged:
            i += 1

    # Optionnel : arrondir les positions à l’indice
    targets = [(tuple(map(int, pos)), amp) for pos, amp in targets]
    return targets


def build_empirical_psf(data_file, frame_idx, ch):
    rdm = compute_RDM(data_file, frame_idx)[ch]
    peak = np.unravel_index(np.argmax(rdm), rdm.shape)
    shift = (rdm.shape[0] // 2 - peak[0], rdm.shape[1] // 2 - peak[1])
    psf = np.roll(rdm, shift=shift, axis=(0, 1))
    return psf / np.max(psf)


psf_empirique_centered = build_empirical_psf("data/18-04/calibration6m.npz", frame_idx=5, ch=0)