import ray
import psutil
if not ray.is_initialized():
    num_cpus = psutil.cpu_count(logical=False)
    print('TaskSim: Ray is not initialized; initializing now...')
    ray.init(num_cpus=num_cpus)
NUM_CPU = ray.available_resources()['CPU']
print(f'TaskSim: Ray initialized with {NUM_CPU} CPUs')

from .structural_similarity import DEFAULT_CA, DEFAULT_CS, InitStrategy
from .gridworld_generator import MDPGraph
