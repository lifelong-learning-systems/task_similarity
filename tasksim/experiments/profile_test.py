import numpy as np

import tasksim
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim

from pyinstrument import Profiler

if __name__ == '__main__':

    profiler = Profiler(interval=0.0001)
    profiler.start()
    val = gen.compare_shapes2_norm((8, 8), (11, 11))
    print('Score:', val)
    profiler.stop()

    profiler.open_in_browser()