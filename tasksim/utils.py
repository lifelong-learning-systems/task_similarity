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
MIN_COMP = None
def init_ray():
    global NUM_CPU
    global RAY_INFO
    global MIN_COMP
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
    from .gridworld_generator import compare_shapes, ActionStrategy
    from .structural_similarity import final_score
    # TODO: invoking with (1, 1) doesn't truly produce lowest value...why?
    MIN_COMP = final_score(compare_shapes((1, 1), (1, 1), strat=ActionStrategy.NOOP_EFFECT_COMPRESS), norm=False)
    if DEBUG_PRINT:
        print(f'TaskSim: Ray initialization complete! Min score is {MIN_COMP}')

def get_num_cpu():
    while NUM_CPU is None:
        print('TaskSim: attempting to invoke without Ray initialized...')
        init_ray()
    return NUM_CPU

def get_min_comp():
    while MIN_COMP is None:
        print('TaskSim: attempting to invoke without Ray initialized...')
        init_ray()
    return MIN_COMP