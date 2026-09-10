def run(rec):
    W = H = 32
    pos, px = rec
    t = pos // 3
    x = t % W
    y = t // W
    i = pos
    col = (px[i], px[i + 1], px[i + 2])
    w = 1
    while x + w < W and (px[i + 3 * w], px[i + 3 * w + 1], px[i + 3 * w + 2]) == col:
        w += 1
    h = 1
    while y + h < H and (px[i + 3 * W * h], px[i + 3 * W * h + 1], px[i + 3 * W * h + 2]) == col:
        h += 1
    return (x, y, w, h,
            col[0] | col[1] << 8 | col[2] << 16,
            px[i - 3] | px[i - 2] << 8 | px[i - 1] << 16)
