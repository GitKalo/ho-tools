import numpy as np

def h(p) :
    """
    Shannon entropy [bits]. Works for multi-dimensional (joint probability distribution) as well.
    """
    return -np.sum(p * np.log2(p))

def O_triple(p_xyz) :
    """
    Equivalent to the interaction information of three variables.

    Same as calling `O_info` with a three-dimensional probability disttribution.

    p_xyz : joint probability distribution
    """
    p_x = np.sum(p_xyz, axis=(1,2))  # Prob dist of x
    p_y = np.sum(p_xyz, axis=(0,2))  # Prob dist of y
    p_z = np.sum(p_xyz, axis=(0,1))  # Prob dist of z

    p_xy = np.sum(p_xyz, axis=2)
    p_xz = np.sum(p_xyz, axis=1)
    p_yz = np.sum(p_xyz, axis=0)

    O = h(p_xyz) + h(p_x) + h(p_y) + h(p_z) - h(p_xy) - h(p_xz) - h(p_yz)

    return O

def O_info(p_joint) :
    """
    O-information between variables, based on joint probability disitrbution.

    Refers to *static*, not dynamic, O-information.
    """
    n = p_joint.ndim
    # ps_marginal = np.zeros((2,n))
    all_idx = set(range(n))

    O = (n-2) * h(p_joint)

    for i in range(n) :
        # Add marginal and subtract joint without individual
        O += h(np.sum(p_joint, axis=tuple(all_idx - {i}))) - h(np.sum(p_joint, axis=i))

    return O

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

def cmi(y, xs, y0=None) :
    """
    Conditional mutual information between `y` and `xs`, conditioned on `y0`.
    If `y0` is None or empty, do not perform conditioning (normal MI).
    """
    n = xs.shape[-1]
    if y0 is None or y0.size == 0 :
        # If none provided or empty array — no conditioning
        yxsy0 = np.concatenate([y,xs], axis=1)
    else :
        yxsy0 = np.concatenate([y,xs,y0], axis=1)
    
    p_yxsy0 = get_p_joint(yxsy0)
    # p_yxsy0 = get_p_joint_np(yxsy0)
    mi = 0
    for i in list(range(1, n+1)) :  # i : 1 to n
        p_joint = np.sum(p_yxsy0, axis=tuple(range(i+1, n+1)))  # 2 to n
        p_cond_terms = np.stack([np.stack([np.sum(p_joint, axis=(0,i))]*2, axis=0)]*2, axis=i)
        mi += np.sum(p_joint * np.log2(
            (p_joint * p_cond_terms) / 
            (np.stack([np.sum(p_joint, axis=i)]*2, axis=i) * np.stack([np.sum(p_joint, axis=0)]*2, axis=0))
        ))

    return mi

def cmi_from_pjoint(p_yxsy0, n) :
    """
    Conditional mutual information between variables, assuming target is in index 0,
    target history at last m indeces, and sources in rest.
    """
    nbins = p_yxsy0.shape[0]     # To make more general (applicable to any hist distribution)
    mi = 0
    for i in list(range(1, n+1)) :  # i : 1 to n
        p_joint = np.sum(p_yxsy0, axis=tuple(range(i+1, n+1)))  # 2 to n
        p_cond_terms = np.stack([np.stack([np.sum(p_joint, axis=(0,i))]*nbins, axis=0)]*nbins, axis=i)
        mi += np.sum(p_joint * np.log2(
            (p_joint * p_cond_terms) / 
            (np.stack([np.sum(p_joint, axis=i)]*nbins, axis=i) * np.stack([np.sum(p_joint, axis=0)]*nbins, axis=0))
        ))

    return mi

def dO(target, source, m=1, return_cmi=False, sample_period=1) :
    """
    Assume same index of axis 0 corresponds to same t in target and source.
    `m` is the order of time-dependence (number of steps back for conditioning).
    """
    n = source.shape[1]     # Determine number of source vars
    if m > 0 :
        y = target[m:]      # Target at t+1
        y0 = np.concatenate([target[m-i-1:-i-1] for i in range(m)], axis=1)    # Target at times t to t-m+1
        xn = source[m-1:-1]    # Source at t
    else :  # Interpret zero m as no conditioning
        y = target[1:]
        y0 = None
        xn = source[:-1]

    # Construct joint probability distribution
    if y0 is None :
        # If none provided or empty array — no conditioning
        yxsy0 = np.concatenate([y,xn], axis=1)
    else :
        yxsy0 = np.concatenate([y,xn,y0], axis=1)
    yxsy0 = yxsy0[::sample_period]  # Subsample
    p_yxsy0 = get_p_joint(yxsy0)

    # mi_yxn = cmi(y, xn, y0)
    mi_yxn = cmi_from_pjoint(p_yxsy0, n)
    assert mi_yxn >= 0, f"Incorrect group MI {mi_yxn} < 0"
    mi_yxj = [cmi_from_pjoint(np.sum(p_yxsy0, axis=j+1), n-1) for j in range(n)]
    assert np.all(np.array(mi_yxj) >= 0), f"Incorrect source indep. MI {mi_yxj} < 0"

    dOn = (1-n)*mi_yxn + np.sum(mi_yxj)

    if return_cmi :
        return dOn, mi_yxn, mi_yxj
    else :
        return dOn

# def dO(target, source, m=1, return_cmi=False) :
#     """
#     Assume same index of axis 0 corresponds to same t in target and source.
#     `m` is the order of time-dependence (number of steps back for conditioning).
#     """
#     n = source.shape[1]     # Determine number of source vars
#     if m > 0 :
#         y = target[m:]      # Target at t+1
#         y0 = np.concatenate([target[m-i-1:-i-1] for i in range(m)], axis=1)    # Target at times t to t-m+1
#         xn = source[m-1:-1]    # Source at t
#     else :  # Interpret zero m as no conditioning
#         y = target[1:]
#         y0 = None
#         xn = source[:-1]

#     mi_yxn = cmi(y, xn, y0)
#     assert mi_yxn >= 0
#     mi_yxj = [cmi(y, np.delete(xn, j, 1), y0) for j in range(n)]
#     assert np.all(np.array(mi_yxj) >= 0), f"Incorrect source indep. MI {mi_yxj} < 0"

#     dOn = (1-n)*mi_yxn + np.sum(mi_yxj)

#     if return_cmi :
#         return dOn, mi_yxn, mi_yxj
#     else :
#         return dOn

def dO_cond_cmi(target, source) :
    """
    Assume same index of axis 0 corresponds to same t in target and source.

    Compute with mutual_information script by Jannis Teunissen <https://github.com/jannisteunissen/mutual_information/blob/main/run_tests.py>.
    """
    from mutual_information import mutual_info

    n = source.shape[1]     # Dynamically determine number of source vars
    y = target[1:]      # Target at t+1
    y0 = target[:-1]    # Target at t
    xn = source[:-1]    # Source at t

    
    mi_yxn = mutual_info.compute_cmi(y, xn, y0, n_neighbors=3)
    assert mi_yxn >= 0
    mi_yxj = [mutual_info.compute_cmi(y, np.delete(xn, j, 1), y0, n_neighbors=3) for j in range(n)]
    assert np.all(np.array(mi_yxj) >= 0), f"Incorrect {mi_yxj}"

    dO3 = (1-n)*mi_yxn + np.sum(mi_yxj)
    return dO3

def dO_cond_knncmi(target, source) :
    """
    Assume same index of axis 0 corresponds to same t in target and source.

    Compute with mutual_information script by .
    """
    import knncmi as k

    n = source.shape[1]     # Dynamically determine number of source vars
    y = target[1:]      # Target at t+1
    y0 = target[:-1]    # Target at t
    xn = source[:-1]    # Source at t
    
    df = pd.DataFrame(np.concatenate((y,xn,y0), axis=1))

    mi_yxn = k.cmi(['0'], ['1', '2', '3'], ['4'], k=3, data=df)
    assert mi_yxn >= 0
    mi_yxj = [k.cmi(['0'], [i for i in ['1', '2', '3'] if i != str(j)], ['4'], k=3, data=df) for j in range(n)]
    assert np.all(np.array(mi_yxj) >= 0), f"Incorrect {mi_yxj}"

    dO3 = (1-n)*mi_yxn + np.sum(mi_yxj)
    return dO3