import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.ndimage import shift as subpixel_shift


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

# ===========================================================================
# Partie 3 : Multicible Bistatique
# ===========================================================================

# --- 1. PSF rectangulaire analytique corrigée ------------------------------
def dirichlet_kernel(Ndata, Nfft, k):
    k = np.asarray(k, dtype=float)
    phase = np.exp(-1j * np.pi * (Ndata - 1) * k / Nfft)
    ratio = Ndata * np.sinc(k * Ndata / Nfft) / np.sinc(k / Nfft)  # robuste
    return phase * ratio

def psf_rectangular(Ms, Mc, N_r_half, N_d, normalize=True):
    """
    PSF 2-D (doppler × range) pour une ouverture rectangulaire uniforme.
    - N_r_half : ½-longueur FFT en range (après padding)
    - N_d      : longueur FFT en Doppler (après padding)
    Retourne une matrice (N_d  × 2*N_r_half) **centrée Doppler** (fftshift).
    """
    # noyaux de Dirichlet 1-D
    Dk = dirichlet_kernel(Ms, 2*N_r_half, np.arange(2*N_r_half))
    Dl = dirichlet_kernel(Mc, N_d, np.arange(N_d)) #- dirichlet_kernel(1, N_d, np.arange(N_d))
    #Dl = (Dl + np.flipud(Dl))
    B  = np.outer(Dl, Dk)                # produit séparant range/doppler
    B  = np.fft.fftshift(B, axes=0)      # centrage Doppler
    psf = np.abs(B) ** 2                 # puissance
    if normalize:
        psf /= np.max(psf)           # pic de puissance = 1
    return psf

# --- 2. PSF théorique centrée, offset corrigé ------------------------------
def build_theoretical_psf(Ms, Mc, PAD_R, PAD_D, delta_r, offset_m):
    """
    Construit la PSF théorique : même tronquage range que compute_RDM
    et recentrage pour compenser l'OFFSET du canal 0.
    """
    N_r_half = PAD_R * Ms // 2
    N_d      = PAD_D * Mc
    psf_th = psf_rectangular(Ms, Mc, N_r_half, N_d)          # pic = 1

    # décalage "quarter-plane" FFT + compensation géométrique
    shift_bins = int(np.round(PAD_R * Ms / 4 - offset_m / delta_r))
    psf_th = np.roll(psf_th, shift_bins, axis=1)

    # tronquage : on garde la moitié utile (comme dans compute_RDM)
    bins_5m = int(np.round(5.0 / delta_r))
    psf_th = psf_th[:, : N_r_half - bins_5m]
    return psf_th

# --- 3. PSF empirique -------------------------------------------------------
def build_empirical_psf(data_file, frame_idx, ch):
    """
    Recentre la PSF autour de son pic (symétrisation pour réduire le bruit).
    Sortie normalisée (pic = 1).
    """
    rdm = compute_RDM(data_file, frame_idx)[ch]
    peak = np.unravel_index(np.argmax(rdm), rdm.shape)
    shift = (rdm.shape[0] // 2 - peak[0], rdm.shape[1] // 2 - peak[1])
    psf = np.roll(rdm, shift=shift, axis=(0, 1))
    #psf = 0.5 * (psf + np.flipud(psf))
    #psf = 0.5 * (psf + np.fliplr(psf))
    return psf / np.max(psf)

def gaussian_window(shape, center, sigma_doppler=11.851385129604585, sigma_range=3.922807157442878):
    """
    Génère une fenêtre gaussienne centrée sur `center`.
    """
    D, R = np.indices(shape)
    gauss = np.exp(-(((D - center[0])**2)/(2*sigma_doppler**2) 
                     + ((R - center[1])**2)/(2*sigma_range**2)))
    return gauss

def binary_mask(shape, center, radius_doppler=10, radius_range=8):
    """
    Crée un masque binaire où les 1 indiquent les zones à conserver
    et les 0 les zones à supprimer.
    
    Arguments:
    - shape: forme du masque (identique à la RDM)
    - center: tuple (doppler_idx, range_idx) indiquant le centre de la suppression
    - radius_doppler: rayon de l'ellipse en indices Doppler
    - radius_range: rayon de l'ellipse en indices Range
    
    Retourne:
    - Un masque binaire de taille shape avec des 0 dans la zone à supprimer
    """
    D, R = np.indices(shape)
    mask = np.ones(shape)
    
    # Création d'une ellipse binaire
    condition = ((D - center[0])**2 / radius_doppler**2 + 
                 (R - center[1])**2 / radius_range**2) <= 1.0
    
    # Mettre à 0 les points à l'intérieur de l'ellipse
    mask[condition] = 0
    
    return mask

def clean_rdm(rdm, psf_full, threshold= 0.010790605558155648, max_iter=7,
              radius_doppler=45.059643672455984, radius_range= 9.999565022794656, use_binary_mask=True):
    """
    CLEAN avec masque binaire pour suppression des lobes secondaires.
    
    Arguments:
    - rdm: matrice RDM brute
    - psf_full: PSF empirique centrée
    - threshold: pourcentage du pic max à atteindre
    - max_iter: nombre maximal d'itérations
    - radius_doppler, radius_range: rayons de l'ellipse en bins
    - use_binary_mask: si True, utilise un masque binaire; sinon, utilise la gaussienne
    """
    rdm_clean = np.copy(rdm)
    targets = []
    iteration = []
    # Normalisation préalable de la PSF
    psf_full = psf_full / np.max(psf_full)

    for _ in range(max_iter):
        peak_value = np.max(rdm_clean)
        if peak_value < threshold*np.max(rdm) :
            break

        peak_idx = np.unravel_index(np.argmax(rdm_clean), rdm_clean.shape)
        targets.append((peak_idx, peak_value))

        """# Décalage de la PSF
        shift_amount = (peak_idx[0] - psf_full.shape[0] // 2,
                        peak_idx[1] - psf_full.shape[1] // 2)
        psf_shifted = np.roll(psf_full, shift=shift_amount, axis=(0, 1))"""
        # Décalage subpixel de la PSF centrée sur le pic détecté
        shift_amount = (
            peak_idx[0] - psf_full.shape[0] // 2 + 0.04/delta_v,
            peak_idx[1] - psf_full.shape[1] // 2
        )
        psf_shifted = subpixel_shift(
            psf_full,
            shift=shift_amount,
            order=3,        # Interpolation linéaire
            mode='constant',
            cval=0.0
        )


        # Facteur optimal par moindres carrés
        # scale_factor = np.sum(rdm_clean * psf_shifted) / np.sum(psf_shifted ** 2)
        scale_factor = 1.2 * np.sum(rdm_clean * psf_shifted) / np.sum(psf_shifted ** 2)

        # Soustraction précise
        rdm_clean -= scale_factor * psf_shifted

        if use_binary_mask:
            # Suppression par masque binaire
            mask = binary_mask(
                rdm_clean.shape, 
                peak_idx,
                radius_doppler=radius_doppler,
                radius_range=radius_range
            )
            rdm_clean *= mask
        else:
            # Ancienne méthode avec fenêtre gaussienne
            gauss_win = gaussian_window(
                rdm_clean.shape,
                peak_idx,
                sigma_doppler=11.851385129604585,
                sigma_range=3.922807157442878
            )
            rdm_clean *= (1 - gauss_win)
        iteration.append(rdm_clean.copy())
        # Forcer la positivité finale
        rdm_clean = np.maximum(rdm_clean, 0)

    return targets, rdm_clean, iteration

def fuse_targets(targets, max_dist_doppler=57, max_dist_range=41):
    """
    Regroupe les cibles proches par fusion pondérée dans une zone elliptique
    avec des semi-axes distincts pour Doppler et Range.
    
    - targets: liste de ((doppler_idx, range_idx), amplitude)
    - max_dist_doppler: semi-axe en Doppler (en bins)
    - max_dist_range: semi-axe en Range (en bins)
    """
    targets = targets.copy()
    i = 0
    while i < len(targets):
        merged = False
        (di, ri), ai = targets[i]
        j = i + 1
        while j < len(targets):
            (dj, rj), aj = targets[j]
            dd = di - dj
            dr = ri - rj
            # Condition elliptique : (dd/a)^2 + (dr/b)^2 <= 1
            if (dd / max_dist_doppler) ** 2 + (dr / max_dist_range) ** 2 <= 1:
                # Moyenne pondérée par amplitude
                total_amp = ai + aj
                new_d = (di * ai + dj * aj) / total_amp
                new_r = (ri * ai + rj * aj) / total_amp
                new_target = ((new_d, new_r), total_amp)

                # Supprimer les deux anciennes cibles
                targets.pop(j)
                targets.pop(i)

                # Ajouter la nouvelle cible à la fin (à reconsidérer)
                targets.append(new_target)
                merged = True
                break  # Recommencer la fusion pour l'indice i
            else:
                j += 1

        if not merged:
            i += 1

    # Arrondir les positions aux indices entiers
    targets = [(tuple(map(int, pos)), amp) for pos, amp in targets]
    return targets



"""
=== Meilleurs paramètres trouvés TOUT SCENARIO===
threshold = 0.010790605558155648
size_doppler = 45.059643672455984
size_range = 9.999565022794656
eps_doppler = 57
eps_range = 41
max_iter = 7
alpha = 0.0

F1-score: 0.8676
Précision: 0.9056
Rappel: 0.8744
"""


"""
=== Meilleurs paramètres trouvés POUR DIMINUER EPS===
'threshold': 0.010698706428397686,
'size_doppler': 45.861773377677714,
'size_range': 11.790914451040322,
'eps_doppler': 45,
'eps_range': 26,
'max_iter': 3,
'alpha': 1

F1-score: 0.8236
Précision: 0.8767
Rappel: 0.8211 
"""


# --- 4. Construction & tracés ----------------------------------------------
data_file = "data/18-04/calibration6m.npz"
frame_idx = 5
channel   = 0                   # canal de référence (OFFSET[0])

# A) PSF empirique
psf_emp = build_empirical_psf(data_file, frame_idx, channel)

# B) PSF théorique
psf_th  = build_theoretical_psf(Ms, Mc, PAD_R, PAD_D,
                                delta_r, OFFSETS[channel])

# C) Différence (empirique − théorique)
peak_emp = np.unravel_index(np.argmax(psf_emp), psf_emp.shape)
peak_th  = np.unravel_index(np.argmax(psf_th),  psf_th.shape)
shift_bins = (peak_emp[0] - peak_th[0],   # Doppler (axe 0)
              peak_emp[1] - peak_th[1])   # Range   (axe 1)
print("Décalage (Doppler, Range) =", shift_bins)
scale = (psf_emp * psf_th).sum() / (psf_th**2).sum()
psf_th *= scale
print("Facteur de mise à l'échelle =", scale)
psf_th_aligned = np.roll(psf_th, shift=shift_bins, axis=(0, 1))
diff = (psf_emp - psf_th_aligned)

err = psf_emp - psf_th
rmse   = np.sqrt((err**2).mean())        # erreur quadratique
nrmse  = rmse / 1.0                      # pic=1 → NRMSE direct
psnr   = -20*np.log10(rmse)              # dB
print(f"NRMSE = {nrmse:.3f}  →  PSNR = {psnr:.1f} dB")

# --- 5. Affichage -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

im0 = axes[0].imshow(psf_emp, cmap="jet", aspect="auto")
axes[0].set_title("PSF empirique (pic = 1)")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(psf_th_aligned, cmap="jet", aspect="auto")
axes[1].set_title("PSF théorique (pic = 1)")
fig.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(diff, cmap="jet", aspect="auto",
                     vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
axes[2].set_title("Écart empirique – théorique")
fig.colorbar(im2, ax=axes[2], fraction=0.046)

plt.show()
