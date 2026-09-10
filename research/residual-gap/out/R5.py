def run(rec):
    pos = rec[0]
    obs = rec[1]
    v3 = pos // 3
    v5 = v3 % 32
    v6 = v3 // 32
    c0 = obs[pos]
    c1 = obs[pos + 1]
    c2 = obs[pos + 2]
    for i7 in range(1, 32):
        v15 = min(pos + 3 * i7, 3069)
        v18 = obs[v15] == c0 and obs[v15 + 1] == c1 and obs[v15 + 2] == c2
        v38 = v5 + i7 < 32
        if not (v18 and v38):
            f8 = i7
            break
    else:
        f8 = 1
    for i42 in range(1, 32):
        v50 = min(pos + 96 * i42, 3069)
        v53 = obs[v50] == c0 and obs[v50 + 1] == c1 and obs[v50 + 2] == c2
        v59 = v6 + i42 < 32
        if not (v53 and v59):
            f43 = i42
            break
    else:
        f43 = 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    v73 = pos - 3
    v80 = (int(obs[v73]) << 0) | (int(obs[v73 + 1]) << 8) | (int(obs[v73 + 2]) << 16)
    return (v5, v6, f8, f43, v72, v80)
