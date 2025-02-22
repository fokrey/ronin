import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# odometry residual
def r_odo(T1,T2,Tobs):
    return (T1*Tobs*T2.inv()).Ln()

# computes full numerical gradient with reference to z for residual of odometry
def dr_dz_twist_factor(T1,T2,Tobs):
    result = np.zeros((6,6))
    epsilon = 1e-4
    initial_ln = (T1*Tobs*T2.inv()).Ln()
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = epsilon
        dTobs = mrob.SE3(xi)
        result[:,i] = ((T1*(dTobs*Tobs)*T2.inv()).Ln() - initial_ln)/epsilon
    return result 

# computes full numerical mixed second order derivative with reference to z for residual of odometry
def d2r_dzdx_twist_factor(T1,T2,Tobs):
    result = np.zeros((6,12,6),dtype=np.float64)
    epsilon = 1e-4
    delta = 1e-4
    # initial_ln = (T1*Tobs*T2.inv()).Ln()
    for i in range(12):
        xi = np.zeros(12)
        xi[i] = epsilon
        dT1 = mrob.SE3(xi[:6])
        dT2 = mrob.SE3(xi[6:])
        for j in range(6):
            zeta = np.zeros(6)
            zeta[j] = delta
            dTobs = mrob.SE3(zeta)

            pp = ((dT1*T1)*(dTobs*Tobs)*(dT2*T2).inv()).Ln()
            pm = ((dT1.inv()*T1)*(dTobs*Tobs)*(dT2.inv()*T2).inv()).Ln()
            mp = ((dT1*T1)*(dTobs.inv()*Tobs)*(dT2*T2).inv()).Ln()
            mm = ((dT1.inv()*T1)*(dTobs.inv()*Tobs)*(dT2.inv()*T2).inv()).Ln()

            result[:,i,j] = (pp-pm-mp+mm)/(4*epsilon*delta)
    return result 

# gps residual
def r_gps(T,Tobs):
    return (T * Tobs.inv()).Ln()

# computes full numerical gradient with reference to z for residual of gps
def dr_dz_gps_factor(T, Tobs):
    result = np.zeros((6,6))
    epsilon = 1e-4
    initial_ln = (T * Tobs.inv()).Ln()
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = epsilon
        dTobs = mrob.SE3(xi)
        result[:,i] = ((T * (dTobs * Tobs).inv()).Ln() - initial_ln)/epsilon
    return result

# computes full numerical mixed second order derivative with reference to z for residual of gps
def d2r_dzdx_gps_factor(T,Tobs):
    result = np.zeros((6,6,6),dtype=np.float64)
    epsilon = 1e-4
    delta = 1e-4
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = epsilon
        dT = mrob.SE3(xi[:6])
        for j in range(6):
            zeta = np.zeros(6)
            zeta[j] = delta
            dTobs = mrob.SE3(zeta)

            pp = ((dT*T)*(dTobs*Tobs).inv()).Ln()
            pm = ((dT.inv()*T)*(dTobs*Tobs).inv()).Ln()
            mp = ((dT*T)*(dTobs.inv()*Tobs).inv()).Ln()
            mm = ((dT.inv()*T)*(dTobs.inv()*Tobs).inv()).Ln()

            result[:,i,j] = (pp-pm-mp+mm)/(4*epsilon*delta)
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
    return
    fig,ax = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(title)
    ax[0].imshow(data)
    ax[1].spy(data)
    sns.heatmap(data, annot=True,ax=ax[2],fmt='.2f',annot_kws={"fontsize":6},square=True)
    plt.show()


def compute_grad(T_1, T_2, T_obs, inf_obs_odo, inf_obs_gps):
    # consider this pair of nodes
    draw_array(T_1.T(),'T_1')
    draw_array(T_2.T(),'T_2')
    draw_array(T_obs.T(),'T_obs')


    d2r_dzdx_odo = d2r_dzdx_twist_factor(T_1,T_2, T_obs)
    # for i in range(6):
    #     draw_array(d2r_dzdx_odo[i],f"d2r_dz_dx_odo[{i}]")

    d2r_dzdx_gps_1 = d2r_dzdx_gps_factor(T_1, T_1)
    # for i in range(6):
    #     draw_array(d2r_dzdx_gps_1[i],f"d2r_dzdx_gps_1[{i}]")
    
    d2r_dzdx_gps_2 = d2r_dzdx_gps_factor(T_2, T_2)
    # for i in range(6):
    #     draw_array(d2r_dzdx_gps_2[i],f"d2r_dzdx_gps_2[{i}]")

    dr_dz_gps1 = dr_dz_gps_factor(T_1, T_1)
    draw_array(dr_dz_gps1,'dr_dz_gps1')

    dr_dz_gps2 = dr_dz_gps_factor(T_2, T_2)
    draw_array(dr_dz_gps2,'dr_dz_gps2')

    # information matrices

    draw_array(inf_obs_odo,'inf_obs_odo')
    
    draw_array(inf_obs_gps,'inf_obs_gps')

    # composing graph
    graph = mrob.FGraph()
    
    n1 = graph.add_node_pose_3d(T_1)
    n2 = graph.add_node_pose_3d(T_2)

    # odometry factor added first
    graph.add_factor_2poses_3d(T_obs,n1,n2,inf_obs_odo)

    # then 2 GPS factors added
    graph.add_factor_1pose_3d(T_1,n1, inf_obs_gps)
    graph.add_factor_1pose_3d(T_2,n2, inf_obs_gps)

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

    W = np.array(graph.get_W_matrix().todense())

    draw_array(W,'W')


    d2r_dzdx = np.zeros((18,12,18))
    d2r_dzdx[:6,:,:6] = d2r_dzdx_odo
    d2r_dzdx[6:12,:6,6:12] = d2r_dzdx_gps_1
    d2r_dzdx[12:,6:,12:] = d2r_dzdx_gps_2

    r_all = np.hstack((r_odo(T_1,T_2,T_obs),r_gps(T_1,T_1),r_gps(T_2,T_2))).reshape(-1,1)

    second_term = (d2r_dzdx.swapaxes(0,1)@W@r_all).squeeze()


    draw_array(second_term, "d2r_dzdx^T * W *r")

    chi2_dzdx = dr_dx_.transpose() @ W @ dr_dz

    draw_array(chi2_dzdx,'chi2_dzdx: first term only')

    draw_array(chi2_dzdx + second_term, 'chi2_dzdx: with second term')


    dx_dz = hessian @ chi2_dzdx

    draw_array(dx_dz, 'dx_dz')

    # graph.solve(mrob.LM, verbose=True)
    x_gt = [T_1, T_2]
    x_pred = graph.get_estimated_state()
    delta_x = np.array([(mrob.SE3(a)*mrob.SE3(b).inv()).Ln() for a,b in zip(x_pred,x_gt)]).reshape((1,-1))
    draw_array(delta_x,'delta_x')

    dL_dz = delta_x @ dx_dz

    draw_array(dL_dz,'dL_dz')

    draw_array(dL_dz[0,:6],'dL_dz[:6]')

    print(f"answer = {dL_dz[0,3]}")

    return dL_dz[0,3]

import torch

import torch.nn as nn



class TrivialModel(nn.Module):
    def __init__(self, initial_value = 2.0):
        super(TrivialModel,self).__init__()
        data = torch.Tensor([initial_value])
        self.scale = nn.Parameter(data, requires_grad=True)

    def forward(self, x):
        return self.scale*x


if __name__ == "__main__":
    T_1 =   mrob.SE3([0, 0, 0, 0,   0, 0])
    T_2 =   mrob.SE3([0, 0, 0, 1,   0, 0])

    initial_state = 1.3

    model = TrivialModel(initial_state)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    alpha = 0.9
    print(f"Noise scale = {alpha}")

    print(f'model parameter should converge to: 1/{alpha} = {1/alpha}')

    z_true = 1.0
    z_noisy = alpha*1.0

    learning_curve = [initial_state]
    target_gradient = []

    for i in tqdm(range(100)):
        optimizer.zero_grad()
        
        z_pred = model(z_noisy)

        T_obs = mrob.SE3([0, 0, 0, z_pred.item(), 0, 0])
        inf_obs_odo = np.identity(6)*1e+1
        inf_obs_gps = np.identity(6)*1e+1

        dL_dz = compute_grad(T_1, T_2, T_obs, inf_obs_odo, inf_obs_gps)
        target_gradient.append(dL_dz)

        z_pred.backward(torch.tensor(dL_dz).reshape((1,)))
        optimizer.step()
        learning_curve.append(model.scale.item())

        print(model.scale.item())

    fig,ax = plt.subplots(2,1)
    ax[0].plot(learning_curve, label='model parameter')
    ax[0].hlines(1/alpha,0,len(learning_curve),'red','dashed',label='true value')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title('Model training')

    ax[1].plot(target_gradient,'red',label='computed gradient')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title('Estimated_gradient')
    plt.tight_layout()
    plt.show()