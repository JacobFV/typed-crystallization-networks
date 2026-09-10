def run(rec):
    pos = rec[0]
    obs = rec[1]
    v3 = pos // 3
    v5 = v3 % 32
    v6 = v3 // 32
    c0 = obs[pos]
    c1 = obs[pos + 1]
    c2 = obs[pos + 2]
    w = 1
    while v5 + w < 32 and obs[pos + 3 * w] == c0 and obs[pos + 3 * w + 1] == c1 and obs[pos + 3 * w + 2] == c2:
        w += 1
    h = 1
    while v6 + h < 32 and obs[pos + 96 * h] == c0 and obs[pos + 96 * h + 1] == c1 and obs[pos + 96 * h + 2] == c2:
        h += 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    v73 = pos - 3
    v80 = (int(obs[v73]) << 0) | (int(obs[v73 + 1]) << 8) | (int(obs[v73 + 2]) << 16)
    return (v5, v6, w, h, v72, v80)
