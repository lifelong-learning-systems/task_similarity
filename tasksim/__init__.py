import ray
import psutil
NUM_CPU = psutil.cpu_count(logical=False)
if not ray.is_initialized():
    print('TaskSim: Ray is not initialized; initializing now...')
    ray.init(num_cpus=NUM_CPU)
print(f'TaskSim: Ray initialized with {NUM_CPU} CPUs')

from numba.extending import get_cython_function_address
import ctypes
from ctypes import CFUNCTYPE, POINTER, c_double, c_int
emd_c_name = 'emd_c_pure'
emd_c_addr = get_cython_function_address('process_chunk', emd_c_name)
emd_c_prototype = CFUNCTYPE(c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int)
emd_c = emd_c_prototype(emd_c_addr)

emd_c_chunk_name = 'emd_c_pure_chunk'
emd_c_chunk_addr = get_cython_function_address('process_chunk', emd_c_chunk_name)
emd_c_chunk_prototype = CFUNCTYPE(None,
                                  c_int, c_int, c_int, c_int, c_int, c_int,
                                  POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
                                  POINTER(c_double),
                                  c_double, c_int)
emd_c_chunk = emd_c_chunk_prototype(emd_c_chunk_addr)

from .structural_similarity import DEFAULT_CA, DEFAULT_CS, InitStrategy
from .gridworld_generator import MDPGraph
