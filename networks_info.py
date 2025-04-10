import sys, os, pickle
from itertools import product, combinations

# Add parent dir to path to make oinfo package visible
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)  

import numpy as np
rng = np.random.default_rng()

import matplotlib.pyplot as plt

import networkx as nx
# import hypergraphx as hgx
# from hypergraphx.generation import random_hypergraph

# Now you can import the package
import oinfo.oinfo as oinfo

# TODO: maybe split this file into two files, for dynamics / gen and for information measures ?

#####################
# Dynamics simulaiton
#####################

def run_sis_sync_nx(G, beta, mu=1, t_max=100, init=0.1, hois_dict=None, beta_tri=0) :
    """
    Discrete-time, synchronous update, probability-based SIS on HO networks with `networkx` package.
    
    hois_dict should be a dict (keys and nodes) of lists of sets (hyperedges)
    """

    beta = min(1, beta)        # Prevent beta > 1 (as it affects p_inf below, but could occur by accident)

    N = G.number_of_nodes()     # ONLY WORKS WHEN NODE INDICES ARE CONSECUTIVE
    nodes = G.nodes()           # e.g. breaks if we took GC of disconnected graph

    res = []

    state = np.zeros(N)
    inf_init = []
    for n in nodes :
        if rng.random() < init :
            inf_init.append(n)
    state[inf_init] = 1
    res.append((0, np.where(state==1)[0].tolist()))

    if hois_dict is None :
        hois_dict = {n : [] for n in nodes}

    t = 1
    n_i = np.sum(state)
    while t < t_max :
        state_new = state.copy()
        # Update each node based on previous time state
        for n_target in nodes :
            # Update state of node
            if state[n_target] == 1 :
                # Recovery
                if rng.random() < mu and n_i > 1 :  # Prevent steady state
                    state_new[n_target] = 0
                    n_i -= 1
            else :
                # Infection, based on neighbours
                nn_inf = sum(state[list(nx.neighbors(G, n_target))])
                p_inf = 1 - (1 - beta)**nn_inf
                if rng.random() < p_inf :
                    state_new[n_target] = 1
                    n_i += 1

                for hyperedge in hois_dict[n_target] :
                    if state[list(hyperedge - {n_target})].all() and rng.random() < beta_tri :
                        state_new[n_target] = 1
                        n_i += 1
                        break

        state = state_new

        res.append((t, np.where(state==1)[0].tolist()))

        # Increment and repeat
        t += 1

    return res

def run_sis_sync_hgx(hg, betas, mu=1, t_max=100, init=0.1, rng=np.random.default_rng()) :
    """
    Discrete-time, synchronous update, probability-based SIS on HO networks with `hypergraphx` package.

    hg is a hypergraphx.Hypergraph object
    betas is a dictionary of beta values, with keys being the hyperedge **sizes**
    """

    # Prevent beta > 1 (as it affects p_inf below, but could occur by accident)
    betas = {size : min(1, beta) for size, beta in betas.items()}

    N = hg.num_nodes()      # ONLY WORKS WHEN NODE INDICES ARE CONSECUTIVE
    nodes = hg.get_nodes()  # e.g. breaks if we took GC of disconnected graph
    
    res = []

    state = np.zeros(N)
    if type(init) is float :
        inf_init = []
        for n in nodes :
            if rng.random() < init :
                inf_init.append(n)
    elif type(init) is list :
        inf_init = init
    elif type(init) is np.ndarray :
        inf_init = init.tolist()
    else :
        raise ValueError("Unrecognized type for `init` parameter: should be either a fraction (float) or a list/array of nodes.")
    state[inf_init] = 1
    res.append((0, np.where(state==1)[0].tolist()))

    t = 1
    n_i = np.sum(state)
    while t < t_max :
        state_new = state.copy()
        # Update each node based on previous time state
        for n_target in nodes :
            if state[n_target] == 1 :   # Recovery, independent
                if rng.random() < mu and n_i > 1 :  # Prevent steady state
                    state_new[n_target] = 0
                    n_i -= 1
            else :      # Infection, from neighbours and hyperedges of all orders (incl. pairwise)
                for hyperedge in hg.get_incident_edges(n_target) :
                    if sum(state[list(hyperedge)]) == len(hyperedge) - 1 and rng.random() < betas[len(hyperedge)] :
                    # if state[list(set(hyperedge) - {n_target})].all() and rng.random() < betas[len(hyperedge)] :
                        state_new[n_target] = 1
                        n_i += 1
                        break   # Avoid unnecessary checks if target gets infected

        state = state_new

        res.append((t, np.where(state==1)[0].tolist()))

        # Increment and repeat
        t += 1

    return res

def run_sis_async(G, beta, mu=1, t_max=1000, init=0.1) :
    """
    Discrete-time, asynchronous update, probability-based SIS on pairwise networks.
    """
    N = G.number_of_nodes()
    nodes = G.nodes()

    res = []

    state = np.zeros(N)
    inf_init = []
    for n in nodes :
        if rng.random() < init :
            inf_init.append(n)
    state[inf_init] = 1
    res.append((0, np.where(state==1)[0].tolist()))

    t = 1
    n_i = np.sum(state)
    while t < t_max :
        # Pick a random node
        n_target = rng.integers(0, len(nodes))

        # Update state of node
        if state[n_target] == 1 :
            # Recovery
            if rng.random() < mu :
                state[n_target] = 0
                n_i -= 1
        else :
            # Infection, based on neighbours
            nn_inf = 0
            for nn in nx.neighbors(G, n_target) :
                if state[nn] == 1 :     # can optimize
                    nn_inf += 1
            p_inf = 1 - (1 - beta)**nn_inf
            if rng.random() < p_inf :
                state[n_target] = 1
                n_i += 1

        res.append((t, np.where(state==1)[0].tolist()))

        if n_i == 0 :
            break

        # Increment and repeat
        t += 1

    # If exited before final time (reached absorbing state), add entry
    # at the end, with no infections at final t
    if res[-1][0] < t_max - 1 :
        res.append((t_max, []))

    return res

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

def save_simple(name, inf_track, G) :
    """
    Old function used for saving results and network.
    """
    np.savetxt(f'sis_{name}.txt', inf_track)
    nx.write_adjlist(G, f'sis_{name}_adj.txt')

#####################################
# Network generation and manipulation
#####################################

def get_rsc(N, p, p_tri) :
    """
    Generate a Random Simplicial Complex (Iacopini et al.).

    Returns pairwise networks as networkx.Graph object, and a dictionary with
    node labels as keys and list of incident hyperedges as values.

    Relies on `simplagion` package.
    """
    from simplagion.utils_simplagion_on_RSC import generate_my_simplicial_complex_d2

    nbr_dict, tri_list = generate_my_simplicial_complex_d2(N, p, p_tri)
    G = nx.Graph(nbr_dict)
    tri_dict = {n : [] for n in G.nodes()}
    for tri in tri_list :
        for n in tri :
            tri_dict[n].append(set(tri))
    # tri_dict = {n : [set(tri) for tri in tri_list if n in tri] for n in G.nodes()}
    return G, tri_dict

def get_tri_clique_simplex_nx(G, simp_dict, n_triplets=10**3) :
    """
    Return sorted lists of triplets, 3-cliques and 2-simplices (exclusive).
    """
    N = G.number_of_nodes()

    # Create set of 2-simpleces
    simplex_2 = set([tuple(simp3) for tris in simp_dict.values() for simp3 in tris])

    # Create set of all triplets that are 3-cliques and not 2-simplices
    cliques_3 = set()
    for n in range(N) :
        for c in cliques_of_node_nx(G, n, minsize=3, maxsize=3) :
            cliques_3.add(tuple(c))
    cliques_3 = cliques_3 - simplex_2

    # Create intermediate set of all triplets
    tri_all = simplex_2 | cliques_3     # Set union

    # Generate fixed amount of random triplets
    # CAREFUL! Will be stuck in infinite loop if # required triplets < # available triplets,
    # which is not impossible for smaller networks and if n_triplets is large
    nodes = range(N)
    while len(tri_all) - len(cliques_3) - len (simplex_2) < n_triplets :    # While "space" for triplets
        tri = tuple(rng.choice(nodes, 3, replace=False))
        if tri not in tri_all :
            # Add to set of all triplets
            tri_all.add(tri)

    # Convert to sorted lists (useful for consistent indexing downstream)
    tri_all = sorted(tri_all)
    cliques_3 = sorted(cliques_3)
    simplex_2 = sorted(simplex_2)

    return tri_all, cliques_3, simplex_2

def get_tri_clique_hyperedge_hgx(hg, n_triplets=100) :
    """
    Return sorted lists of triplets, 3-cliques and hyperedges of all sizes (exclusive).
    The hyperedges are provided as a dictionary with sizes as keys and sorted lists as values.
    hg is a hypergraphx.Hypergraph object
    """
    N = hg.num_nodes()

    # Get sets simpleces
    hyperedges = {s : set(hg.get_edges(size=s)) for s in range(3, hg.max_size()+1)}

    # Create set of all triplets that are 3-cliques and not 2-simplices
    cliques_3 = set()
    for n in range(N) :
        for c in cliques_of_node_hgx(hg, n, minsize=3, maxsize=3) :
            cliques_3.add(tuple(c))
    cliques_3 = cliques_3 - hyperedges[3]

    # Create intermediate set of all triplets
    tri_all = hyperedges[3] | cliques_3     # Set union

    # Generate fixed amount of random triplets
    # CAREFUL! Will be stuck in infinite loop if # required triplets < # available triplets,
    # which is not impossible for smaller networks and if n_triplets is large
    nodes = range(N)
    while len(tri_all) - len(cliques_3) - len (hyperedges[3]) < n_triplets :    # While "space" for triplets
        tri = tuple(rng.choice(nodes, 3, replace=False).tolist())
        if tri not in tri_all :
            tri_all.add(tri)

    tri_all = sorted(tri_all)
    cliques_3 = sorted(cliques_3)
    hyperedges = {s : sorted(simp) for s, simp in hyperedges.items()}

    return tri_all, cliques_3, hyperedges

def simp_dict_from_list(simp_list, N) :
    simp_dict = {n : [] for n in range(N)}
    for simp in simp_list :
        for n in simp :
            simp_dict[n].append(set(simp))
    return simp_dict

def get_pmf(data, edges) :
    vals, _ = np.histogram(data, bins=edges)
    pmf = vals / np.size(data)
    return pmf

def cliques_of_node_nx(G, n, minsize=None, maxsize=None) :
    """
    Get all cliques of given sizes that node n participates in.

    G is a networkx.Graph object.
    """
    cliques = [c for c in list(nx.enumerate_all_cliques(G)) if n in c]
    if minsize is None and maxsize is None :
        return cliques
    elif maxsize is None :
        return [c for c in cliques if (len(c) >= minsize)]
    elif minsize is None :
        return [c for c in cliques if (len(c) <= maxsize)]
    else :
        return [c for c in cliques if (len(c) <= maxsize) and (len(c) >= minsize)]

def cliques_of_node_hgx(hg, n, minsize=3, maxsize=3) :
    """
    Get all cliques of given sizes that node n participates in.

    G is a hypergraphx.Hypergraph object.
    """
    cliques = []
    for size in range(minsize, maxsize+1) :
        for nbrs in combinations(hg.get_neighbors(n, size=2), size-1) :
            for i, j in combinations(nbrs, 2) :
                if not hg.check_edge((i, j)) :
                    break
            else :  # If no break, then clique exists
                cliques.append(tuple(sorted([n, *nbrs])))
    return sorted(cliques)

def make_hypergraph_simplicial(hg) :
    """
    Modifies a hypergraph in place to include all possible lower-order hyperedges 
    for each existing hyperedge.
    """
    for size in range(2, hg.max_size()+1) :
        top_edges = hg.get_edges(size=size)
        for e in top_edges :
            for sub_size in range(2, size) :
                hg.add_edges(combinations(e, sub_size))

######################
# Information measures
######################

def run_pis_triplets(beta_factor, output_fname='tri_pis.txt', network_pkl_fname='G_tri_1.pkl') :
    """
    Run and record PID atoms for triplets, 3-cliques, and 2-simpleces, for given beta factor.

    Uses Williams and Beer's I_min redundancy measure.
    """
    from dit import Distribution
    from dit.pid import PID_WB

    outcomes = [''.join(p) for p in product("01", repeat=3)]
    G_tri, tri_all, cliques_3, simplex_2 = pickle.load(open(network_pkl_fname, 'rb'))
    simp_dict = simp_dict_from_list(simplex_2, G_tri.number_of_nodes())

    k_mean = 2 * nx.number_of_edges(G_tri) / nx.number_of_nodes(G_tri)
    mu = 1
    beta = mu/k_mean * beta_factor
    
    print(f"Running for beta = {beta}")

    t_max = 10**4       # Minimum for accurate stats is 10^4 samples
    res = run_sis_sync_nx(G_tri, beta, mu, t_max, hois_dict=simp_dict, beta_tri=beta*2)

    tri_pis = np.zeros((len(tri_all), 4))
    for i, tri in enumerate(tri_all) :
        p = get_p_joint_from_res(res, tri)

        # Calculate synergy and redundancy with node in index 0 as target
        d = Distribution(outcomes, p.flatten())
        pid = PID_WB(d)
        tri_pis[i,0] = pid.get_pi(((0,),(1,)))  # Redundant
        tri_pis[i,1] = pid.get_pi(((0,),))      # Unique
        tri_pis[i,2] = pid.get_pi(((1,),))      # Unique
        tri_pis[i,3] = pid.get_pi(((0,1),))     # Synergistic

    np.savetxt(output_fname, tri_pis)

###################
# Utility funcitons
###################

def get_p_joint_np(Xs, bins=8) :
    """
    For binary variables. Xs assumed to have samples (e.g. time) in axis 0.
    """
    vals, _ = np.histogramdd(Xs, bins=bins)
    p = vals / np.sum(vals) + 10e-30    # TODO: Reduce error with proper normalization
    return p

