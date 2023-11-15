import taichi as ti 

@ti.func
def sample(field, i, j):
    i = ti.max(0, ti.min(field.shape[0] - 1, i))
    j = ti.max(0, ti.min(field.shape[1] - 1, j))
    idx = ti.Vector([int(i), int(j)])
    return field[idx]


def diff_x(field, i, j):
    """Central Difference x"""
    return 0.5 * (sample(field, i + 1, j) - sample(field, i - 1, j))


@ti.func
def diff_y(field, i, j):
    """Central Difference y"""
    return 0.5 * (sample(field, i, j + 1) - sample(field, i, j - 1))


@ti.func
def diff2_x(field, i, j):
    return sample(field, i + 1, j) - 2.0 * sample(field, i, j) + sample(field, i - 1, j)


@ti.func
def diff2_y(field, i, j):
    return sample(field, i, j + 1) - 2.0 * sample(field, i, j) + sample(field, i, j - 1)