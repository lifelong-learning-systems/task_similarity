import numpy as np

import tasksim
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim

from pyinstrument import Profiler

if __name__ == '__main__':
    tasksim.init_ray()

    profiler = Profiler(interval=0.0001)
    profiler.start()
    n = 15
    val = gen.compare_shapes2_norm((n, n), (n, n))
    print('Score:', val)
    profiler.stop()

    profiler.open_in_browser()