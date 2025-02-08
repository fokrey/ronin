import numpy as np
import mrob
from tqdm import tqdm
import os

def info_matrix_from_triangular(elements):
    """
    Reconstruct a 6x6 information matrix from the triangular part in g2o/toro files.
    """
    full_matrix = np.zeros((6, 6))
    upper_tri_indices = np.triu_indices(6)
    full_matrix[upper_tri_indices] = elements
    return full_matrix + np.triu(full_matrix, 1).T


def read_graph_toro_description_3d(toro_file):
    """
    Reads a 3D Toro file that may contain 'VERTEX3', 'EDGE3', 'EDGE1'.
    Returns:
       vertex_ini: dict[node_id -> initial 6-vector (x,y,z,roll,pitch,yaw)]
       factors: dict[(origin,target) -> (meas(6), info(6x6))]
       factors_dictionary: adjacency-like dict[ node_id -> list of connected node_ids ]
    """
    vertex_ini = {}
    factors = {}
    factors_dictionary = {}

    with open(toro_file, 'r') as file:
        for line in file:
            d = line.strip().split()
            if not d:
                continue
            if d[0] == 'VERTEX3':
                # VERTEX3 index  x y z roll pitch yaw
                node_index = int(float(d[1]))
                pose = np.array([float(v) for v in d[2:8]], dtype='float64')
                vertex_ini[node_index] = pose
                factors_dictionary[node_index] = []
            elif d[0] == 'EDGE3':
                # EDGE3 origin_id target_id  dx dy dz droll dpitch dyaw  I_?? (21 triangular info entries)
                node_origin = int(float(d[1]))
                node_target = int(float(d[2]))
                meas = np.array([float(v) for v in d[3:9]], dtype='float64')
                info_values = [float(v) for v in d[9:]]
                info = info_matrix_from_triangular(info_values)
                factors[(node_origin, node_target)] = (meas, info)
                # adjacency
                if node_target in factors_dictionary:
                    factors_dictionary[node_target].append(node_origin)
                else:
                    factors_dictionary[node_target] = [node_origin]
            elif d[0] == 'EDGE1':
                # EDGE1 node_index x y z roll pitch yaw  I_?? (21 triangular info entries)
                node_index = int(float(d[1]))
                meas = np.array([float(v) for v in d[2:8]], dtype='float64')
                info_values = [float(v) for v in d[8:]]
                info = info_matrix_from_triangular(info_values)
                factors[(node_index, node_index)] = (meas, info)
                if node_index in factors_dictionary:
                    factors_dictionary[node_index].append(node_index)
                else:
                    factors_dictionary[node_index] = [node_index]
    return vertex_ini, factors, factors_dictionary


def compose_graph_3d(
    vertex_ini,
    factors,
    factors_dictionary,
    perturb_node=None,   # (node_id, coord_idx, dX)
    perturb_factor=None  # (origin_id, target_id, coord_idx_z, dZ)
):

    graph = mrob.FGraph()

    # 1) Add nodes to the graph
    for node_index in sorted(vertex_ini.keys()):
        x = vertex_ini[node_index].copy()
        if perturb_node is not None:
            node_id_pert, coord_idx_pert, dx_val = perturb_node
            if node_id_pert == node_index:
                x[coord_idx_pert] += dx_val

        pose = mrob.SE3(x) 
        graph.add_node_pose_3d(pose)

    for (nodeOrigin, nodeTarget), (measurement, information_matrix) in factors.items():
        obs = measurement.copy()
        if perturb_factor is not None:
            origin_f, target_f, coord_idx_z, dz_val = perturb_factor
            if origin_f == nodeOrigin and target_f == nodeTarget:
                obs[coord_idx_z] += dz_val

        obs_se3 = mrob.SE3(obs)
        if nodeOrigin != nodeTarget:
            graph.add_factor_2poses_3d(obs_se3, nodeOrigin, nodeTarget, information_matrix)
        else:
            graph.add_factor_1pose_3d(obs_se3, nodeOrigin, information_matrix)

    return graph

def find_factor_coord_idx(i_z, coord_num=6):
    """
    If i_z is in [0, 6*num_factors-1], decode which factor index & which dimension in [0..5].
    factor_idx_z = i_z // coord_num
    coord_idx_z  = i_z %  coord_num
    """
    factor_idx_z = i_z // coord_num
    coord_idx_z  = i_z % coord_num
    return factor_idx_z, coord_idx_z


def compute_perturbed_chi2(
    vertex_ini,
    factors,
    factors_dictionary,
    node_id,
    node_coord_idx,
    dx_val,
    factor_perturb=None
):

    graph = compose_graph_3d(
        vertex_ini,
        factors,
        factors_dictionary,
        perturb_node=(node_id, node_coord_idx, dx_val),
        perturb_factor=factor_perturb
    )
    graph.solve(mrob.LM, verbose=False)

    return graph.chi2(evaluateResidualsFlag=True)


def numerical_diff2_3d(toro_file, dx=1e-4, dz=1e-4):
    """
    Returns the matrix d^2(chi^2)/(dx dz) \in R^{(6*N) x (6*M)},
    i.e. partial derivatives of the final optimized chi^2
    w.r.t. each node dimension (x) and each factor dimension (z).
    Then multiplies by inv(H) from the factor graph's linearization: inv(H) * ...
    """
    vertex_ini, factors, factors_dictionary = read_graph_toro_description_3d(toro_file)

    graph_0 = compose_graph_3d(vertex_ini, factors, factors_dictionary)
    graph_0.solve(mrob.LM, verbose=False)

    x_0 = graph_0.get_estimated_state()
    hessian = graph_0.get_information_matrix().todense()

    node_keys_sorted = sorted(vertex_ini.keys())
    num_nodes = len(node_keys_sorted)
    dim_x = num_nodes * 6

    factor_keys = list(factors.keys())  
    num_factors = len(factor_keys)
    dim_z = num_factors * 6

    chi2_matrix = np.zeros((dim_x, dim_z))

    for i_x in tqdm(range(dim_x), desc="Diff2 w.r.t. states (x)"):
        node_idx = i_x // 6
        coord_x  = i_x %  6
        node_id  = node_keys_sorted[node_idx]

        for i_z in range(dim_z):
            factor_idx_z, coord_z = find_factor_coord_idx(i_z, coord_num=6)
            (origin_z, target_z) = factor_keys[factor_idx_z]

            chi2_pp = compute_perturbed_chi2(
                vertex_ini, factors, factors_dictionary,
                node_id, coord_x, +dx,
                factor_perturb=(origin_z, target_z, coord_z, +dz)
            )
            chi2_pm = compute_perturbed_chi2(
                vertex_ini, factors, factors_dictionary,
                node_id, coord_x, +dx,
                factor_perturb=(origin_z, target_z, coord_z, -dz)
            )
            chi2_mp = compute_perturbed_chi2(
                vertex_ini, factors, factors_dictionary,
                node_id, coord_x, -dx,
                factor_perturb=(origin_z, target_z, coord_z, +dz)
            )
            chi2_mm = compute_perturbed_chi2(
                vertex_ini, factors, factors_dictionary,
                node_id, coord_x, -dx,
                factor_perturb=(origin_z, target_z, coord_z, -dz)
            )

            chi2_matrix[i_x, i_z] = (chi2_pp - chi2_pm - chi2_mp + chi2_mm) / (4.0 * dx * dz)

    inv_hessian = np.linalg.inv(hessian)
    return inv_hessian @ chi2_matrix

import matplotlib.pyplot as plt
def visualize_gradient(gradient):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].imshow(gradient)
    ax[1].spy(gradient, precision=1e-7)
    plt.show()


if __name__ == "__main__":
    toro_file_path = './out/toro_file_49.txt'

    result = numerical_diff2_3d(toro_file_path, dx=1e-5, dz=1e-5)

    print("Final shape of numeric partial derivatives:", result.shape)
    visualize_gradient(result)