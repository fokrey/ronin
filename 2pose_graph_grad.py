import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)
import matplotlib.pyplot as plt

# computes full numerical gradient for residual of odometry
def dr_dz(T1,T2,Tobs):
    result = np.zeros((6,6))
    epsilon = 1e-4
    initial_ln = (T1*Tobs*T2.inv()).Ln()
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = epsilon
        dTobs = mrob.SE3(xi)
        result[:,i] = ((T1*dTobs*Tobs*T2.inv()).Ln() - initial_ln)/epsilon
    return result 

def derivative_simple_ln(T1):
    result = np.zeros((6,6))
    epsilon = 1e-4
    initial_ln = T1.Ln()
    for i in range(6):
        xi = np.zeros(6) 
        xi[i] = epsilon
        dT = mrob.geometry.SE3(xi)
        result[:,i] = (( dT * T1 ).Ln() - initial_ln)/epsilon
    return result


if __name__ == "__main__":
    # consider this pair of nodes
    T_1 =   mrob.SE3([0, 0, 0, 0,   0, 0])
    T_2 =   mrob.SE3([0, 0, 0, 1,   0, 0])
    T_obs = mrob.SE3([0, 0, 0, 0.9, 0, 0]) # small error in odometry

    # information matrices
    inf_obs_odo = np.identity(6)*1e+0
    inf_obs_gps = np.identity(6)*1e+2

    # composing graph
    graph = mrob.FGraph()
    
    n1 = graph.add_node_pose_3d(T_1)
    n2 = graph.add_node_pose_3d(T_2)

    graph.add_factor_2poses_3d(T_obs,n1,n2,inf_obs_odo)
    graph.add_factor_1pose_3d(T_1,n1, inf_obs_gps)
    graph.add_factor_1pose_3d(T_2,n2, inf_obs_gps)

    # cheking the residual value
    r = (T_1*T_obs*T_2.inv())
    print(f"{r=}\n")

    print(f"{r.Ln()=}")

    # numerical grad  dr/dz of odometry factor residual
    dr_dz_ = dr_dz(T_1, T_2, T_obs)

    print(dr_dz_)
    plt.imshow(dr_dz_)
    
    plt.figure()
    plt.spy(dr_dz_)

    plt.show()

    # doing single optimization step with Gauss Newton solver to have adjacency matrix available from graph
    graph.solve(mrob.GN) # Gonzalo said it is ok to do single step
    

    dr_dx_ = graph.get_adjacency_matrix().todense()[:6,6:]
    print(dr_dx_)
    W = graph.get_W_matrix().todense()[:6,:6]

    chi2_dx_dz = dr_dx_*W*dr_dz_ # TODO Aikun check this expression if it mataches the Implicit theorem expression

    plt.imshow(chi2_dx_dz)
    print(chi2_dx_dz)
    plt.show()

