import numpy as np

from utils import get_p_joint, get_p_joint_np

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
    
    # p_yxsy0 = get_p_joint(yxsy0)
    p_yxsy0 = get_p_joint_np(yxsy0, bins=2)
    mi = 0
    for i in list(range(1, n+1)) :  # i : 1 to n
        p_joint = np.sum(p_yxsy0, axis=tuple(range(i+1, n+1)))  # 2 to n
        p_cond_terms = np.stack([np.stack([np.sum(p_joint, axis=(0,i))]*2, axis=0)]*2, axis=i)
        mi += np.sum(p_joint * np.log2(
            (p_joint * p_cond_terms) / 
            (np.stack([np.sum(p_joint, axis=i)]*2, axis=i) * np.stack([np.sum(p_joint, axis=0)]*2, axis=0))
        ))
    # TODO: edit to work with variable number of bins (like below function)
    return mi

def cmi_from_pjoint(p_yxsy0, n) :
    """
    Conditional mutual information between a single target and a set of source variables.
    Assumes that:
        - dim 0 of dist corresponds to target (y);
        - dims 1 to ndim-m correspond to sources (xs); and
        - last m dims correspond to target history (y0).
    """
    nbins = p_yxsy0.shape[0]     # To make more general (applicable to any hist distribution)
    mi = 0
    for i in range(1, n+1) :  # i : 1 to n
        p_joint = np.sum(p_yxsy0, axis=tuple(range(i+1, n+1)))  # 2 to n
        p_cond_terms = np.stack([np.stack([np.sum(p_joint, axis=(0,i))]*nbins, axis=0)]*nbins, axis=i)
        mi += np.sum(p_joint * np.log2(
            (p_joint * p_cond_terms) / 
            (np.stack([np.sum(p_joint, axis=i)]*nbins, axis=i) * np.stack([np.sum(p_joint, axis=0)]*nbins, axis=0))
        ))

    # TODO: Could this funciton be made more efficient by calculating via entropy?

    return mi

def cmi_from_pjoint_ent(p_yxsy0, n) :
    """
    Conditional mutual information between a single target and a set of source variables.
    Assumes that:
        - dim 0 of dist corresponds to target (y);
        - dims 1 to ndim-m correspond to sources (xs); and
        - last m dims correspond to target history (y0).

    Based on testing, this entropy-based version is slightly faster than 
    the direct probability-based version above.
    """
    nbins = p_yxsy0.shape[0]     # To make more general (applicable to any hist distribution)
    
    p_y0 = np.sum(p_yxsy0, axis=tuple(range(0, n+1)))
    p_yy0 = np.sum(p_yxsy0, axis=tuple(range(1, n+1)))
    p_cond_y0 = p_yy0 / np.sum(p_yy0, axis=0)[None,:] # Add "empty" axis, similar to keepdims but controlled
    # Can also do this via the following:
    # print(p_yy0 / np.stack([np.sum(p_yy0, axis=0)]*nbins, axis=0))
    # print(p_yy0 / np.sum(p_yy0, axis=0, keepdims=True))  # Keepdims is fine if you operate along same axis

    h_cond_y0 = -np.sum(np.stack([p_y0]*nbins, axis=0) * p_cond_y0 * np.log2(p_cond_y0))

    p_xsy0 = np.sum(p_yxsy0, axis=0)
    p_cond_xsy0 = p_yxsy0 / np.sum(p_yxsy0, axis=0)[None,...]
    
    h_cond_xsy0 = -np.sum(p_xsy0[None,...] * p_cond_xsy0 * np.log2(p_cond_xsy0))
    
    mi = float(h_cond_y0 - h_cond_xsy0)
    
    return mi

def dO(target, source, m=1, return_cmi=False, sample_period=1) :
    """
    Assume same index of axis 0 corresponds to same t in target and source.
    `m` is the order of time-dependence (number of steps back for conditioning).
    """
    #TODO: currently fails silently if empty arrays are passed
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

    return dO_from_pjoint(p_yxsy0, m=m, return_cmi=return_cmi)

def dO_from_pjoint(p_joint, m=1, return_cmi=False) :
    """
    Assumes that:
        - dim 0 of p_joint corresponds to target;
        - dims 1 to ndim-m correspond to sources; and
        - last m dims correspond to target history.
    """
    n = p_joint.ndim-1-m    # Number of sources

    # mi_yxn = cmi(y, xn, y0)
    mi_yxn = cmi_from_pjoint(p_joint, n)
    # assert mi_yxn >= 0, f"Incorrect group MI {mi_yxn} < 0"
    mi_yxj = [cmi_from_pjoint(np.sum(p_joint, axis=j+1), n-1) for j in range(n)]
    # assert np.all(np.array(mi_yxj) >= 0), f"Incorrect source indep. MI {mi_yxj} < 0"
    # The assertions above are useful, but they are too restrictive due to small rounding errors
    # TODO: either fix rounding error possibility or find a more flexible assertion check
    
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

def redundancy(p_joint, return_pw_mis=False) :
    """
    Bottom-level redundancy atom of PID assuming I_MMI redundancy function.

    Expect target in axis 0, sources in axes 1 to len(sources), and target history in last axis.

    Works for single-target, single-step history.

    TODO: Make for more than one-step target history (arbitrary m), also for `synergy` function.
    """
    ndim = p_joint.ndim
    s_ids = list(range(1, ndim-1))
    # Marginal MIs
    mis_pw = []
    for s in s_ids :
        # Marginalize over other source histories
        p = np.sum(p_joint, axis=tuple([s_id for s_id in s_ids if s_id != s]))
        mis_pw.append(cmi_from_pjoint_ent(p, 1))
    
    if return_pw_mis :
        return min(mis_pw), mis_pw
    else :
        return min(mis_pw)

def synergy(p_joint, mi_all=None, return_mi=False) :
    """
    Top-level synergy atom of PID assuming I_MMI redundancy function.

    Expect target in axis 0, sources in axes 1 to len(sources), and target history in last axis.

    Works for single-target, single-step history.
    """
    ndim = p_joint.ndim
    s_ids = list(range(1, ndim-1))
    if mi_all is None :
        mi_all = cmi_from_pjoint_ent(p_joint, len(s_ids))
    # One-out MIs
    mis_oo = []
    for s in s_ids :
        # Marginalize over source history
        p = np.sum(p_joint, axis=s)
        mis_oo.append(cmi_from_pjoint_ent(p, len(s_ids)-1))
    
    if return_mi :
        return mi_all - max(mis_oo), mi_all
    else :
        return mi_all - max(mis_oo)

def get_stat_dist_for_index(i, measure, mask_g1, mask_g2, n_bins=20) :
    """
    Get the statistical distance ("delta") between the distributions of values given by
    the `measure` parameter at index `i` for the two groups defined by
    array masks `mask_g1` and `mask_g2`.
    """
    # Distance needs to be over common alphabet (specific to each i)
    measure = np.nan_to_num(measure)  # Convert NaNs to 0s for computing distance
    bin_min = min(np.min(measure[i][mask_g1]), np.min(measure[i][mask_g2]))
    bin_max = max(np.max(measure[i][mask_g1]), np.max(measure[i][mask_g2]))
    common_bin_range = (bin_min, bin_max)

    # Get probability distributions
    dist_g1, _ = np.histogram(measure[i][mask_g1], bins=n_bins, range=common_bin_range)
    dist_g2, _  = np.histogram(measure[i][mask_g2],  bins=n_bins, range=common_bin_range)
    dist_g1 = dist_g1.astype(float) / np.sum(dist_g1)
    dist_g2 = dist_g2.astype(float) / np.sum(dist_g2)

    # Calculate statistical distance
    return np.sum(np.abs(dist_g2 - dist_g1))/2