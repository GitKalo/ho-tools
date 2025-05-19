import pickle
from itertools import product, combinations

import numpy as np
rng = np.random.default_rng()

import networkx as nx

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

def run_sis_sync_hgx(hg, betas, mu=1, t_max=100, init=0.1, rng=None) :
    """
    Discrete-time, synchronous update, probability-based SIS on HO networks with `hypergraphx` package.

    hg is a hypergraphx.Hypergraph object
    betas is a dictionary of beta values, with keys being the hyperedge **sizes**
    """
    # If no rng provided, create a fresh one.
    # Useful for working with multiple processes when this is the only 
    # function using random numbers.
    if rng is None :
        rng = np.random.default_rng()

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

def mean_prev(prev, drop_frac=0.1) :
    """
    Calculate the mean pravelence after drop_frac of the series has passed.
    """
    M = len(prev)
    prev = prev[int(M*drop_frac):]
    return np.mean(prev)

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
    # Try to add n_triplets random triplets
    while len(tri_all) - len(cliques_3) - len(hyperedges[3]) < n_triplets :    # While "space" for triplets
        tri = tuple(rng.choice(nodes, 3, replace=False).tolist())
        if tri not in tri_all :
            tri_all.add(tri)

    tri_all = sorted(tri_all)
    cliques_3 = sorted(cliques_3)
    hyperedges = {s : sorted(simp) for s, simp in hyperedges.items()}

    return tri_all, cliques_3, hyperedges

def get_quad_clique_hyperedge_hgx(hg, n_quadruplets=100) :
    """
    Return sorted lists of quadruplets, 4-cliques and hyperedges of all sizes (exclusive).
    The hyperedges are provided as a dictionary with sizes as keys and sorted lists as values.
    hg is a hypergraphx.Hypergraph object
    """
    N = hg.num_nodes()

    # Get sets simpleces
    hyperedges = {s : set(hg.get_edges(size=s)) for s in range(3, hg.max_size()+1)}

    # Create set of all triplets that are 3-cliques and not 2-simplices
    cliques_4 = set()
    for n in range(N) :
        for c in cliques_of_node_hgx(hg, n, minsize=4, maxsize=4) :
            cliques_4.add(tuple(c))
    cliques_4 = cliques_4 - hyperedges[4]

    # Create intermediate set of all triplets
    quad_all = hyperedges[4] | cliques_4     # Set union

    # Generate fixed amount of random triplets
    # CAREFUL! Will be stuck in infinite loop if # required triplets < # available triplets,
    # which is not impossible for smaller networks and if n_triplets is large
    nodes = range(N)
    # Try to add n_triplets random triplets
    while len(quad_all) - len(cliques_4) - len(hyperedges[4]) < n_quadruplets :    # While "space" for triplets
        quad = tuple(rng.choice(nodes, 4, replace=False).tolist())
        if quad not in quad_all :
            quad_all.add(quad)

    quad_all = sorted(quad_all)
    cliques_4 = sorted(cliques_4)
    hyperedges = {s : sorted(simp) for s, simp in hyperedges.items()}

    return quad_all, cliques_4, hyperedges

def simp_dict_from_list(simp_list, N) :
    simp_dict = {n : [] for n in range(N)}
    for simp in simp_list :
        for n in simp :
            simp_dict[n].append(set(simp))
    return simp_dict

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
    for size in range(2, hg.max_size()+1) :     # TODO: this should be range(3,max+1)?
        top_edges = hg.get_edges(size=size)
        for e in top_edges :
            for sub_size in range(2, size) :
                hg.add_edges(combinations(e, sub_size))

def random_simplicial_complex(N, ks_mean) :
    """
    Generate a random simplicial complex with given size and (approximate) average degrees.
    Works for maximum size 3 (triangles).

    Algorithm is as described in Robiglio et al. (2025), producing essentially
    the same graph as Iacopini et al. (2019), as implemented in the `simplagion` 
    package and `get_rsc` function.
    """
    # Funciton-specific imports
    from math import factorial, prod
    from hypergraphx.generation import random_hypergraph

    # Construct connection probabilities for obtaining a simplicial complex
    # with the desired average degrees
    ps = {}
    ps[2] = (ks_mean[2] - 2*ks_mean[3]) / (N-1 - 2*ks_mean[3])
    ps[3] = 2*ks_mean[3] / ((N-1)*(N-2))

    # Obtian the number of edges from the connection probabilties
    ms = {s : p*prod([N-i for i in range(0,s)])/factorial(s) for s, p in ps.items()}

    # Create a random hypergraph using hgx function
    hg = random_hypergraph(N, ms)
    
    # Fill in lower order hyperedges to make it simplicial (in-place)
    make_hypergraph_simplicial(hg)
    
    return hg

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

##########################
# Backbone (edge rankings)
##########################

def get_ranked_edges(edge_measures) :
    """
    Rankes edges by the average of the given measure.

    Expects a dictionary with hyperedge sizes as keys and values being
    a dictionary with hyperedges (ids) as keys and values being
    lists of measure values (e.g. over different nodes as targets) or scalars.

    Returns a dictionary with hyperedge sizes as keys and values being a tuple
    of ranked edges and ranked edge measure values.
    """
    # TODO: Generalized to any aggregation — e.g. min, max..., not just avg

    sizes = list(edge_measures.keys())

    # Average measures for each hyperedge
    edge_measures_avg = {size : {} for size in sizes}
    for size in sizes :
        for he, measure in edge_measures[size].items() :
            edge_measures_avg[size][he] = sum(measure)/size

    # Sort edges based on averaged measures
    edges_ranked = {}
    for size in sizes :
        edges, vals = zip(*sorted(edge_measures_avg[size].items(), key=lambda k : k[1], reverse=True))
        edges_ranked[size] = (edges, vals)  # Issue that they are long tuples rather than lists?
    
    return edges_ranked