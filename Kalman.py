import numpy as np
from scipy.optimize import least_squares

kalman_x = None
kalman_P = None

if(t == 0.0):
    # print("t equals 0")
    kalman_x = np.array([5,4,4.9,0,1.5]).T #initial state, change according to file for now
    kalman_P = np.array([[2,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,0.5]]) #initial state covariance matrix
# print('=====================Frame ' + str(frameCount) + " ====================================")
# print(kalman_P)

kalman_F = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]])
kalman_Q = np.array([[0.1,0,0,0],[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.1]])
kalman_H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
kalman_R = np.array([[3,0,0,0],[0,3,0,0],[0,0,5,0],[0,0,0,5]])
kalman_xp = kalman_F @ kalman_x
kalman_Pp = kalman_F @ kalman_P @ kalman_F.T + kalman_Q

kalman_K = kalman_Pp @ kalman_H.T @ np.linalg.inv(kalman_H @ kalman_P @ kalman_H.T + kalman_R)
takeRDM = np.ones(4, dtype=int)
outlierRadius = 40.0
z = np.zeros(4)
point_est = speed_est = np.zeros(2)
for q in range(len(takeRDM)):
    dp = (kalman_xp[0]**2+kalman_xp[1]**2)**(1/2) + ((kalman_xp[0]-foyers[q][0])**2 + (kalman_xp[1]-foyers[q][1])**2)**(1/2)
    vp = 0.5 * ((np.array([kalman_xp[0]-foyers[q][0], kalman_xp[1]-foyers[q][1]]) @ kalman_xp[2:])*(1/np.sqrt((kalman_xp[0]-foyers[q][0])**2+(kalman_xp[1]-foyers[q][1])**2)) +
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
            d = (x**2+y**2)**(1/2) + ((x-foyers[q][0])**2 + (y-foyers[q][1])**2)**(1/2)
            res.append(dmes-d)
        return res
    
    x0 = [0.0,0.0]
    point_est = least_squares(diff,x0,loss='cauchy').x
    # print(point_est)
    N = np.fromfunction(
        lambda q, i: 0.5 * (
            (point_est[i.astype(int)] - foyers[q.astype(int), i.astype(int)]) *
            (1 / np.sqrt((point_est[0] - foyers[q.astype(int), 0])**2 + (point_est[1] - foyers[q.astype(int), 1])**2)) +
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