# Creates 6 (x,y) points in a 2D space [-10,10][-10,10]
# One is the target, one is the emitter and the rest are the receivers
# Finds the distances r_q between the target and the receivers and the target and the emitter
# Plots the target in red, the emitter in blue and the receivers in green
# Plots all the ellipses of the receivers and the emitter with the distances r_q 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import minimize
def plot_perfect_distances():
    # Seed for reproducibility
    np.random.seed(0)
    # Generate points
    target = np.asarray([3, 5])
    emitter = np.asarray([-1, -1])
    receivers = np.asarray([[-0.35,2.7],[1.8,0.5],[5.1,-2.3],[0.0,0.0]])
    
    # Compute distances
    r_e = np.linalg.norm(target - emitter)
    r_q = np.linalg.norm(receivers - target, axis=1)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(*target, color='red', label='Cible', s=100)
    ax.scatter(*emitter, color='blue', label='Emetteur', s=50)
    ax.scatter(receivers[:, 0], receivers[:, 1], color='green', label='Recepteur', s=50)
    
    # Plot ellipses for each receiver
    t = np.linspace(0, 2 * np.pi, 400)
    for i, recv in enumerate(receivers):
        s = r_e + r_q[i]   # sum of distances for ellipse
        a = s / 2
        c = np.linalg.norm(recv - emitter) / 2
        b = np.sqrt(max(a**2 - c**2, 0))
        
        # Ellipse center and rotation
        center = (emitter + recv) / 2
        angle = np.arctan2(recv[1] - emitter[1], recv[0] - emitter[0])
        
        # Parametric ellipse
        x_ell = a * np.cos(t)
        y_ell = b * np.sin(t)
        
        # Rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse = R @ np.vstack([x_ell, y_ell]) + center.reshape(2, 1)
        
        ax.plot(ellipse[0, :], ellipse[1, :], label=f'Ellipse {i+1}', alpha=0.6)
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Ellipses dans le cas parfait')
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

def plot_noisy_distances(noise_std=2):
    # Fixed points from the perfect-distance scenario
    target = np.asarray([3, 5])
    emitter = np.asarray([-1, -1])
    receivers = np.asarray([[-0.35, 2.7], [1.8, 0.5], [5.1, -2.3], [0.0, 0.0]])
    
    # Compute true distances
    r_e_true = np.linalg.norm(target - emitter)
    r_q_true = np.linalg.norm(receivers - target, axis=1)
    
    # Add Gaussian noise
    np.random.seed(42)
    r_e_noisy = r_e_true + np.random.normal(0, noise_std)
    r_q_noisy = r_q_true + np.random.normal(0, noise_std, size=r_q_true.shape)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(*target, color='red', label='Cible', s=100)
    ax.scatter(*emitter, color='blue', label='Emetteur', s=50)
    ax.scatter(receivers[:, 0], receivers[:, 1], color='green', label='Recepteurs', s=50)
    
    # Plot ellipses for each receiver with noisy sums
    t = np.linspace(0, 2 * np.pi, 400)
    for i, recv in enumerate(receivers):
        s_noisy = r_e_noisy + r_q_noisy[i]
        a = s_noisy / 2
        c = np.linalg.norm(recv - emitter) / 2
        b = np.sqrt(max(a**2 - c**2, 0))
        
        center = (emitter + recv) / 2
        angle = np.arctan2(recv[1] - emitter[1], recv[0] - emitter[0])
        
        x_ell = a * np.cos(t)
        y_ell = b * np.sin(t)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse = R @ np.vstack([x_ell, y_ell]) + center.reshape(2, 1)
        
        ax.plot(ellipse[0, :], ellipse[1, :], label=f'Ellipse bruitée {i+1}', alpha=0.6)
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Ellipses avec mesures de distances bruitées")
    #ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()


def plot_least_square_target(noise_std=2):
    # Fixed points
    target_true = np.asarray([3, 5])
    emitter = np.asarray([-1, -1])
    receivers = np.asarray([[-0.35, 2.7], [1.8, 0.5], [5.1, -2.3], [0.0, 0.0]])
    
    # Compute true distances
    r_e_true = np.linalg.norm(target_true - emitter)
    r_q_true = np.linalg.norm(receivers - target_true, axis=1)
    
    # Add Gaussian noise
    np.random.seed(42)
    r_e_noisy = r_e_true + np.random.normal(0, noise_std)
    r_q_noisy = r_q_true + np.random.normal(0, noise_std, size=r_q_true.shape)
    
    # Objective: sum of squared residuals to measured ellipse sums
    def residuals(point):
        point = np.asarray(point)
        res = []
        for recv, rq in zip(receivers, r_q_noisy):
            s_meas = r_e_noisy + rq
            res.append((np.linalg.norm(point - emitter) + np.linalg.norm(point - recv) - s_meas))
        return np.array(res)
    
    def cost(point):
        return np.sum(residuals(point)**2)
    
    # Initial guess (centroid of receivers)
    x0 = receivers.mean(axis=0)
    result = least_squares(cost, x0, loss = 'cauchy')

    est = result.x
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot noisy ellipses
    t = np.linspace(0, 2 * np.pi, 400)
    for i, recv in enumerate(receivers):
        s_noisy = r_e_noisy + r_q_noisy[i]
        a = s_noisy / 2
        c = np.linalg.norm(recv - emitter) / 2
        b = np.sqrt(max(a**2 - c**2, 0))
        center = (emitter + recv) / 2
        angle = np.arctan2(recv[1] - emitter[1], recv[0] - emitter[0])
        x_ell = a * np.cos(t)
        y_ell = b * np.sin(t)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipse = R @ np.vstack([x_ell, y_ell]) + center.reshape(2, 1)
        ax.plot(ellipse[0, :], ellipse[1, :], alpha=0.5)
    
    # Plot true points and estimate
    ax.scatter(*target_true, color='red', label='Cible vraie', s=100)
    ax.scatter(*emitter, color='blue', label='Emetteur', s=50)
    ax.scatter(receivers[:, 0], receivers[:, 1], color='green', label='Recepteurs', s=50)
    ax.scatter(*est, color='magenta', label='Estimation LS', s=100)
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Estimation du point cible par moindres carrés")
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

noise_std = 0.3
# Example usage
plot_perfect_distances()
plot_noisy_distances(noise_std)
plot_least_square_target(noise_std)


