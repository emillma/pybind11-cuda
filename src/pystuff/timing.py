import time


def timeit(func, *args, times=100):
    output = func(*args)

    t0 = time.perf_counter()
    for i in range(times):
        func(*args)
    t_final = time.perf_counter()

    return output, (t_final - t0)/times
