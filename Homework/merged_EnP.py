# slam_integration.py
import particle_filter as pf
import ekf_slam as ekf
import matplotlib.pyplot as plt
import numpy as np

# Particle filter params
Q = np.diag([0.2]) ** 2  # range error
R = np.diag([2.0, np.deg2rad(40.0)]) ** 2  # input error

# Simulation parameters
Q_sim = np.diag([0.2]) ** 2
R_sim = np.diag([1.0, np.deg2rad(30.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range

show_animation = True

def main():
    # EKF state covariance
    Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
    # Simulation parameters for EKF SLAM
    Q_sim_ekf = np.diag([0.2, np.deg2rad(1.0)]) ** 2
    R_sim_ekf = np.diag([1.0, np.deg2rad(10.0)]) ** 2
    STATE_SIZE = 3
    NP = 100  # Number of Particle
    NTh = NP / 2.0  # Number of particles for re-sampling

    particle = pf.PF(Q_sim, R_sim, NP=NP)
    px = np.zeros((STATE_SIZE, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight

    Ekf = ekf.EKF(Cx, Q_sim_ekf, R_sim_ekf, STATE_SIZE=STATE_SIZE)

    print(__file__ + " start!!")

    # EKF SLAM history
    time = 0.0
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0]])
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)
    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    h_x_est =xEst
    # Particle Filter history
    h_x_dr = np.zeros((STATE_SIZE, 1))

    while SIM_TIME >= time:
        time += DT
        u = Ekf.calc_input()

        # EKF SLAM
        xTrue, z, xDR, ud = Ekf.observation(xTrue, xDR, u, RFID)
        xEst, PEst = Ekf.ekf_slam(xEst, PEst, ud, z)

        # Particle Filter Localization
        x_true, z_, x_dr, ud_ = particle.observation(xTrue, xDR, u, RFID)
        x_est, _, px, pw = particle.pf_localization(px, pw, z_, ud_)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        h_x_dr = np.hstack((h_x_dr, x_dr))
        hxTrue = np.hstack((hxTrue, xTrue))

        h_x_est = np.hstack((h_x_est, x_est))

        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".b", label="EKF Estimate")
            plt.plot(px[0, :], px[1, :], ".r", label="PF Estimate")

            # plot landmark
            for i in range(Ekf.calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg", label="EKF Landmark")

            # for i in range(particle.calc_n_lm(x_est)):
            #     plt.plot(x_est[STATE_SIZE + i * 2],
            #              x_est[STATE_SIZE + i * 2 + 1], "oy", label="PF Landmark")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-k", label="True Trajectory")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-.k", label="Dead Reckoning")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-b", label="EKF Estimated Trajectory")
            plt.plot(np.array(h_x_est[0, :]).flatten(),
                     np.array(h_x_est[1, :]).flatten(), "-r",label="PF Estimated Trajectory")
        
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
