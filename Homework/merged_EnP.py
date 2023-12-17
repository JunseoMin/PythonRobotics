import particle_filter as pf
import ekf_slam as ekf
import utils
import matplotlib.pyplot as plt
import numpy as np

################### particle filter params #######################
# Estimation parameter of PF
Q = np.diag([0.2]) ** 2  # range error
R = np.diag([2.0, np.deg2rad(40.0)]) ** 2  # input error

#  Simulation parameter
Q_sim = np.diag([0.2]) ** 2
R_sim = np.diag([1.0, np.deg2rad(30.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

show_animation = True
##################################################################

def main():
    # EKF state covariance
    Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

    #  Simulation parameter
    Q_sim_ekf = np.diag([0.2, np.deg2rad(1.0)]) ** 2
    R_sim_ekf = np.diag([1.0, np.deg2rad(10.0)]) ** 2

    STATE_SIZE = 3


    Ekf = ekf.EKF(Cx,Q_sim_ekf,R_sim_ekf,STATE_SIZE=STATE_SIZE)
    print(__file__ + " start!!")

    ################################EKF############################
    time = 0.0
    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    ##############################################################
    while SIM_TIME >= time:
        time += DT
        u = Ekf.calc_input()

        xTrue, z, xDR, ud = Ekf.observation(xTrue, xDR, u, RFID)

        xEst, PEst = Ekf.ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
    
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(Ekf.calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


    #pf.main()
    pass

if __name__ == '__main__':
    main()
    pass