import numpy as np

#########################################################
# Functions for computing joint probability distributions
#########################################################

def get_p_joint(Xs) :
    """
    For binary variables. Xs assumed to have time in axis 0.
    """
    # Calculate joint probability distribution
    p_joint = np.zeros(tuple([2] * Xs.shape[1]))
    for s in Xs :
        # print(s)
        p_joint[tuple(s)] += 1
    p_joint = p_joint / np.sum(p_joint) + 10e-30
    # p_joint = p_joint / np.sum(p_joint)
    
    return p_joint

def get_p_joint_np(Xs, bins=8) :
    """
    For binary variables. Xs assumed to have time in axis 0.
    """
    vals, _ = np.histogramdd(Xs, bins=bins)
    p = vals / np.sum(vals) + 10e-30
    return p

def get_p_joint_cont(Xs, bins=10) :
    """
    Joint probability distribution for continuous variables.
    """
    pass    # TODO

def get_inf_from_res(res, N) :
    """
    Get 2d array of node states from list of lists of infected nodes.
    """
    ts, infs = list(zip(*res))
    inf_track = np.zeros((len(res), N), dtype='int8')
    for i, inf in enumerate(infs) :
        inf_track[i,inf] = 1
    return ts, inf_track

def get_p_joint_from_res(res, nodes, lag=0) :
    """
    Get joint state distribution for nodes in `nodes` list.
    
    Places first node index as state in axis 0 of distribution, and fills other 
    axes based on samples from state of other nodes at a time lag of `lag`. 
    A lag of 0 (default) corresponds to same-time samples.
    """
    p = np.zeros((2,)*len(nodes))
    for i_res, (t, infs) in enumerate(res[lag:]) :    # Start from lag-th time step
        s = [int(nodes[0] in infs)]     # State of target node
        s += [int(n in res[i_res][1]) for n in nodes[1:]]   # State of neighbours lag steps before
        p[tuple(s)] += 1
    p = p / np.sum(p)
    return p

def get_p_joint_from_inftrack(ts, inf_track, nodes, lag=0) :
    """
    Get joint state distribution for nodes in `nodes` list.
    
    Places first node index as state in axis 0 of distribution, and fills other 
    axes based on samples from state of other nodes at a time lag of `lag`. 
    A lag of 0 (default) corresponds to same-time samples.

    Compared to `get_p_joint_from_res`, is generally faster (once state matrix is computed)
    but since it relies on the inf_track matrix it has a higher memory footprint.
    This scales with number of nodes and samples, and thus should be used with
    care for larger networks (inf_track generally already ~2MB for 200 nodes).
    """
    p = np.zeros((2,)*len(nodes))
    nodes = list(nodes)
    for t in ts :
        s = [inf_track[t,nodes[0]]]             # State of target node
        s += inf_track[t-lag,nodes[1:]].tolist()    # State of neighbours lag steps before
        p[tuple(s)] += 1
    p = p / np.sum(p)
    return p

###########################
# Data processing functions
###########################

def get_pmf(data, edges) :
    vals, _ = np.histogram(data, bins=edges)
    pmf = vals / np.size(data)
    return pmf

####################
# Data I/O functions
####################

def save_simple(name, inf_track, G) :
    """
    Old function used for saving results and network.
    """
    import networkx as nx

    np.savetxt(f'sis_{name}.txt', inf_track)
    nx.write_adjlist(G, f'sis_{name}_adj.txt')