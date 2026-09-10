def run(rec):
    pos = rec[0]
    obs = rec[1]
    n = len(obs)
    if 3 == 0: raise ValueError("zero denominator")
    v3 = pos // 3
    if not 0 <= v3 <= 65535: raise OverflowError("value outside integer range")
    if 32 == 0: raise ValueError("zero denominator")
    v5 = v3 % 32
    if not 0 <= v5 <= 65535: raise OverflowError("value outside integer range")
    if 32 == 0: raise ValueError("zero denominator")
    v6 = v3 // 32
    if not 0 <= v6 <= 65535: raise OverflowError("value outside integer range")
    # hoisted out of both loops: the anchor pixel's three bytes, which the
    # module recomputed on every iteration of both loops
    a1 = pos + 1
    if not 0 <= a1 <= 65535: raise OverflowError("value outside integer range")
    a2 = pos + 2
    if not 0 <= a2 <= 65535: raise OverflowError("value outside integer range")
    if not 0 <= pos < n: raise IndexError("index outside tuple")
    c0 = obs[pos]
    if not 0 <= a1 < n: raise IndexError("index outside tuple")
    c1 = obs[a1]
    if not 0 <= a2 < n: raise IndexError("index outside tuple")
    c2 = obs[a2]
    for i7 in range(1, 32):
        v12 = 3 * i7
        if not 0 <= v12 <= 65535: raise OverflowError("value outside integer range")
        v13 = pos + v12
        if not 0 <= v13 <= 65535: raise OverflowError("value outside integer range")
        v15 = min(v13, 3069)
        if not 0 <= v15 <= 65535: raise OverflowError("value outside integer range")
        t19 = v15 + 1
        if not 0 <= t19 <= 65535: raise OverflowError("value outside integer range")
        t20 = v15 + 2
        if not 0 <= t20 <= 65535: raise OverflowError("value outside integer range")
        if not 0 <= v15 < n: raise IndexError("index outside tuple")
        t21 = obs[v15]
        if not 0 <= t19 < n: raise IndexError("index outside tuple")
        t22 = obs[t19]
        if not 0 <= t20 < n: raise IndexError("index outside tuple")
        t23 = obs[t20]
        t29 = t21 == c0
        t30 = t22 == c1
        t31 = t23 == c2
        t32 = (True, True, True, False)[2 * t29 + t30]
        v18 = (False, True, False, False)[2 * t32 + t31]
        v37 = v5 + i7
        if not 0 <= v37 <= 65535: raise OverflowError("value outside integer range")
        v38 = v37 < 32
        if not (v18 and v38):
            f8 = i7
            break
    else:
        f8 = 1
    for i42 in range(1, 32):
        v47 = 96 * i42
        if not 0 <= v47 <= 65535: raise OverflowError("value outside integer range")
        v48 = pos + v47
        if not 0 <= v48 <= 65535: raise OverflowError("value outside integer range")
        v50 = min(v48, 3069)
        if not 0 <= v50 <= 65535: raise OverflowError("value outside integer range")
        u19 = v50 + 1
        if not 0 <= u19 <= 65535: raise OverflowError("value outside integer range")
        u20 = v50 + 2
        if not 0 <= u20 <= 65535: raise OverflowError("value outside integer range")
        if not 0 <= v50 < n: raise IndexError("index outside tuple")
        u21 = obs[v50]
        if not 0 <= u19 < n: raise IndexError("index outside tuple")
        u22 = obs[u19]
        if not 0 <= u20 < n: raise IndexError("index outside tuple")
        u23 = obs[u20]
        u29 = u21 == c0
        u30 = u22 == c1
        u31 = u23 == c2
        u32 = (True, True, True, False)[2 * u29 + u30]
        v53 = (False, True, False, False)[2 * u32 + u31]
        v57 = v6 + i42
        if not 0 <= v57 <= 65535: raise OverflowError("value outside integer range")
        v59 = v57 < 32
        if not (v53 and v59):
            f43 = i42
            break
    else:
        f43 = 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    if not 0 <= v72 <= 16777215: raise OverflowError("value outside integer range")
    v73 = pos - 3
    if not 0 <= v73 <= 65535: raise OverflowError("value outside integer range")
    if not 0 <= v73 < n: raise IndexError("index outside tuple")
    v74 = obs[v73]
    v75 = v73 + 1
    if not 0 <= v75 <= 65535: raise OverflowError("value outside integer range")
    if not 0 <= v75 < n: raise IndexError("index outside tuple")
    v76 = obs[v75]
    v77 = v73 + 2
    if not 0 <= v77 <= 65535: raise OverflowError("value outside integer range")
    if not 0 <= v77 < n: raise IndexError("index outside tuple")
    v78 = obs[v77]
    v80 = (int(v74) << 0) | (int(v76) << 8) | (int(v78) << 16)
    if not 0 <= v80 <= 16777215: raise OverflowError("value outside integer range")
    return (v5, v6, f8, f43, v72, v80)
