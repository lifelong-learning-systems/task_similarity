import ray
import psutil

from numba.extending import get_cython_function_address
import ctypes
from ctypes import CFUNCTYPE, POINTER, c_double, c_int

import logging

def get_emd_c_chunk():
    emd_c_chunk_name = 'emd_c_pure_chunk'
    emd_c_chunk_addr = get_cython_function_address('process_chunk', emd_c_chunk_name)
    emd_c_chunk_prototype = CFUNCTYPE(None,
                                    c_int, c_int, c_int, c_int, c_int, c_int,
                                    POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
                                    POINTER(c_double),
                                    c_double, c_int)
    emd_c_chunk = emd_c_chunk_prototype(emd_c_chunk_addr)
    return emd_c_chunk

def get_emd_c():
    emd_c_name = 'emd_c_pure'
    emd_c_addr = get_cython_function_address('process_chunk', emd_c_name)
    emd_c_prototype = CFUNCTYPE(c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int)
    emd_c = emd_c_prototype(emd_c_addr)
    return emd_c

emd_c = get_emd_c()
emd_c_chunk = get_emd_c_chunk()

NUM_CPU = None
RAY_INFO = None
def init_ray():
    global NUM_CPU
    global RAY_INFO
    DEBUG_PRINT = True
    try:
        RAY_INFO = ray.init(address='auto', logging_level=logging.FATAL)
        if DEBUG_PRINT:
            print(f'TaskSim: Connected to Ray instance; monitor on {RAY_INFO["webui_url"]}')
    except:
        pass
    NUM_CPU = psutil.cpu_count(logical=False)
    if not ray.is_initialized():
        if DEBUG_PRINT:
            print('TaskSim: Ray is not initialized; initializing now...')
        RAY_INFO = ray.init(num_cpus=NUM_CPU, logging_level=logging.FATAL)
    NUM_CPU = ray.available_resources()['CPU'] if 'CPU' in ray.available_resources() else NUM_CPU
    if DEBUG_PRINT:
        print(f'TaskSim: Ray initialized with {NUM_CPU} CPUs')
        print(f'TaskSim: Preparing ray workers..')
    from .gridworld_generator import compare_shapes2_norm
    # TODO: use a proper Pool instead
    # Basically a clean pass through, initialize the workers, etc.
    _ = compare_shapes2_norm((1, 1), (1, 1))
    if DEBUG_PRINT:
        print(f'TaskSim: Ray initialization complete!')

def get_num_cpu():
    while NUM_CPU is None:
        print('TaskSim: attempting to invoke without Ray initialized...')
        init_ray()
    return NUM_CPU
