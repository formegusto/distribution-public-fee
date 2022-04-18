def get_nugin_step(usage, fee):
    step = 0
    for s, f in enumerate(fee):
        if usage < f[1]:
            step = s
            break

    return step
