import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)
import matplotlib.pyplot as plt

import seaborn as sns

# computes full numerical gradient with reference to z for residual of odometry
def dr_dz_twist_factor(T1,T2,Tobs):
    result = np.zeros((6,6))
    epsilon = 1e-4
    initial_ln = (T1*Tobs*T2.inv()).Ln()
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = epsilon
        dTobs = mrob.SE3(xi)
        result[:,i] = ((T1*dTobs*Tobs*T2.inv()).Ln() - initial_ln)/epsilon
    return result 

# computes full numerical gradient with reference to z for residual of gps
def dr_dz_gps_factor(T, Tobs):
    result = np.zeros((6,6))
    epsilon = 1e-4
    initial_ln = (T * Tobs.inv()).Ln()
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = epsilon
        dTobs = mrob.SE3(xi)
        result[:,i] = ((T * dTobs * Tobs.inv()).Ln() - initial_ln)/epsilon
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

def draw_array(data, title):
    # return
    fig,ax = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(title)
    ax[0].imshow(data)
    ax[1].spy(data)
    sns.heatmap(data, annot=True,ax=ax[2])
    plt.show()


if __name__ == "__main__":
    # consider this pair of nodes
    T_1 =   mrob.SE3([0, 0, 0, 0,   0, 0])
    T_2 =   mrob.SE3([0, 0, 0, 1,   0, 0])
    T_obs = mrob.SE3([0, 0, 0, 0.9, 0, 0]) # small error in odometry

    dr_dz_gps1 = dr_dz_gps_factor(T_1, T_1)
    draw_array(dr_dz_gps1,'dr_dz_gps1')

    dr_dz_gps2 = dr_dz_gps_factor(T_2, T_2)
    draw_array(dr_dz_gps2,'dr_dz_gps2')

    # information matrices
    inf_obs_odo = np.identity(6)*1e+0
    inf_obs_gps = np.identity(6)*1e+2

    # composing graph
    graph = mrob.FGraph()
    
    n1 = graph.add_node_pose_3d(T_1)
    n2 = graph.add_node_pose_3d(T_2)

    # odometry factor added first
    graph.add_factor_2poses_3d(T_obs,n1,n2,inf_obs_odo)

    # then 2 GPS factors added
    graph.add_factor_1pose_3d(T_1,n1, inf_obs_gps)
    graph.add_factor_1pose_3d(T_2,n2, inf_obs_gps)

    # cheking the residual value
    r = (T_1*T_obs*T_2.inv())
    # print(f"{r=}\n")

    # print(f"{r.Ln()=}")

    # numerical grad  dr/dz of odometry factor residual
    dr_dz_odo = dr_dz_twist_factor(T_1, T_2, T_obs)

    draw_array(dr_dz_odo,'dr_dz_odo' )

    dr_dz = np.zeros((18,18))
    dr_dz[:6,:6] = dr_dz_odo
    dr_dz[6:12,6:12] = dr_dz_gps1
    dr_dz[12:, 12:] = dr_dz_gps2

    draw_array(dr_dz,'dr_dz')

    # doing single optimization step with Gauss Newton solver to have adjacency matrix available from graph
    graph.solve(mrob.GN) # Gonzalo said it is ok to do single step
    
    # preparing Hessian
    hessian = graph.get_information_matrix().todense()
    draw_array(hessian,'hessian')

    hessian = - np.linalg.inv(np.array(hessian))
    draw_array(hessian, 'Inverse Hessian with - : -H^(-1)')

    # matrix A
    
    dr_dx_ = graph.get_adjacency_matrix().todense()
    draw_array(dr_dx_,'dr_dx')

    W = graph.get_W_matrix().todense()

    draw_array(W,'W')

    chi2_dx_dz = dr_dx_.transpose() @ W @ dr_dz

    draw_array(chi2_dx_dz,'chi2_dx_dz')


    dx_dz = hessian @ chi2_dx_dz

    draw_array(dx_dz, 'dx_dz')

    graph.solve(mrob.LM, verbose=True)
    x_gt = [T_1, T_2]
    x_pred = graph.get_estimated_state()
    delta_x = np.array([(mrob.SE3(a)*mrob.SE3(b).inv()).Ln() for a,b in zip(x_gt,x_pred)]).reshape((1,-1))
    draw_array(delta_x,'delta_x')

    dL_dz = -delta_x @ dx_dz


    draw_array(dL_dz,'dL_dz')

    draw_array(dL_dz[0,:6],'dL_dz[:6]')

    print(f"answer = {dL_dz[0,3]}")