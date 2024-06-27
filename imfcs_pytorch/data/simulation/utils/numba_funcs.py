"""Code from the imfcsnet.Utilities.ufunc file, with minor refactorings to remove required imports from the configuration file.

Changes as follows:
- Removed aliased names randu, randp etc, now uses explicit calls to the exact functions.

# Notes from original file:
Using numba cuda jit wrapper for generation of simulation data in parallel.
In compile decorator, we have to define separate wrapper for different length of arguments.
"""

import math
from numba import cuda, float32, uint32, int64
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)
from inspect import signature

# Typing imports
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from collections.abc import Callable


# Poisson random generators.
@cuda.jit(device=True)
def xoroshiro128p_poisson_mult_uint32(
    states: DeviceNDArray, index: int, lam: float
) -> int:
    """Return a Poisson distributed int32 and advance ``states[index]``. The return value is drawn from a Poisson distribution of mean=lam using Donald Knuth Poisson process method. Use only for small lam.

    Args:
        states (DeviceNDArray): RNG state array created by `create_xoroshiro128p_states`. 1D array, dtype=xoroshiro128p_dtype
        index (int): Offset in states to update
        lam (float): Poisson mean value. Use only for small lam.

    Returns:
        int: uint32 value sampled from a poisson distribution.
    """
    enlam = math.exp(-float32(lam))
    index = int64(index)
    x = uint32(0)
    prod = float32(1.0)
    while True:
        prod *= xoroshiro128p_uniform_float32(states, index)
        if prod > enlam:
            x += uint32(1)
        else:
            return x


@cuda.jit(device=True)
def xoroshiro128p_poisson_ptrs_uint32(
    states: DeviceNDArray, index: int, lam: float
) -> int:
    """Return a Poisson distributed int32 and advance ``states[index]``. The return value is drawn from a Poisson distribution of mean=lam using the method of W. Hoermann. Use for moderate to large lam.

    Args:
        states (DeviceNDArray): RNG state array created by `create_xoroshiro128p_states`. 1D array, dtype=xoroshiro128p_dtype
        index (int): Offset in states to update
        lam (float): Poisson mean value. Use for moderate to large lam.

    Returns:
        int: uint32 value sampled from a poisson distribution.
    """
    lam = float32(lam)
    index = int64(index)
    slam = math.sqrt(lam)
    loglam = math.log(lam)
    b = float32(0.931) + float32(2.53) * slam
    a = float32(-0.059) + float32(0.02483) * b
    invalpha = float32(1.1239) + float32(1.1328) / (b - float32(3.4))
    vr = float32(0.9277) - float32(3.6224) / (b - float32(2.0))
    while True:
        u = xoroshiro128p_uniform_float32(states, index) - float32(0.5)
        v = float32(1.0) - xoroshiro128p_uniform_float32(states, index)
        us = float32(0.5) - math.fabs(u)
        if us < float32(0.013) and v > us:
            continue
        fk = math.floor((float32(2.0) * a / us + b) * u + lam + float32(0.43))
        if (us >= float32(0.07)) and (v <= vr):
            return uint32(fk)
        if fk < 0.0:
            continue
        if math.log(v) + math.log(invalpha) - math.log(
            a / (us * us) + b
        ) <= -lam + fk * loglam - math.lgamma(fk + float32(1.0)):
            return uint32(fk)


@cuda.jit(device=True)
def xoroshiro128p_poisson_uint32(states: DeviceNDArray, index: int, lam: float) -> int:
    """Return a Poisson distributed int32 and advance ``states[index]``. Wraps around the 2 methods, using W. Hoermann for `lam>10` and Donald Knuth otherwise.

    Args:
        states (DeviceNDArray): RNG state array created by `create_xoroshiro128p_states`. 1D array, dtype=xoroshiro128p_dtype
        index (int): Offset in states to update
        lam (float): Poisson mean value. Use for moderate to large lam.

    Returns:
        int: uint32 value sampled from a poisson distribution.
    """
    if lam > 10.0:
        return xoroshiro128p_poisson_ptrs_uint32(states, index, lam)
    if lam == 0.0:
        return uint32(0)
    return xoroshiro128p_poisson_mult_uint32(states, index, lam)


absidx = cuda.jit(device=True)(lambda fromidx, idx: fromidx + idx)


def cuda_compile(max: int, seed: int, batch_size: int, device_id: int) -> Callable:
    """Compilation wrapper for functions that allow them to be executed on a CUDA-compatible GPU.

    Args:
        max (int): The maximum number of blocks/threads to initialize. Primarily affects the RNG state generation step. Generally calculated as batches * batch_size.
        seed (int): The global seed to use for RNG generation. Might need to be changed if resuming training from an existing save state to prevent re-seen data.
        batch_size (int): The expected batch size to be used for training. Will be used to compile the kernel for execution on a corresponding number of blocks.
        device_id (int): The CUDA device to compile simulations for. This is here to ensure that the simulations and PyTorch training run on the same GPU, as it would default to device ID 0 otherwise.

    Returns:
        Callable: Decorator function for JIT compilation to CUDA-compatible GPUs.
    """
    rng = create_xoroshiro128p_states(n=max, seed=seed)

    # in both cases, 'rng.shape[0]' equals 'max'
    def decorator(func):
        args = len(signature(func).parameters)
        func = cuda.jit(device=True)(func)

        def kernel(rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4, d_ary5):
            idx = cuda.grid(1)
            if idx < toidx - fromidx:
                func(rng, fromidx, toidx, idx, d_ary1, d_ary2, d_ary3, d_ary4, d_ary5)

        kernel = cuda.jit(kernel)

        def wrapper(fromidx, toidx, ary1, ary2, ary3, ary4, ary5):
            if device_id is not None:
                cuda.select_device(device_id)
            if (
                fromidx < 0
                or toidx > max
                or toidx > ary1.shape[0]
                or toidx > ary2.shape[0]
                or toidx > ary3.shape[0]
                or toidx > ary4.shape[0]
                or toidx > ary5.shape[0]
            ):
                raise ValueError("index out of bound")
            d_ary1 = cuda.to_device(ary1[fromidx:toidx])
            d_ary2 = cuda.to_device(ary2[fromidx:toidx])
            d_ary3 = cuda.to_device(ary3[fromidx:toidx])
            d_ary4 = cuda.to_device(ary4[fromidx:toidx])
            d_ary5 = cuda.to_device(ary5[fromidx:toidx])
            blocks = (toidx - fromidx + batch_size - 1) // batch_size
            threads = batch_size
            kernel[blocks, threads](
                rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4, d_ary5
            )
            cuda.synchronize()

            return d_ary1.copy_to_host()

        return wrapper

    return decorator


def cuda_compile_4darray(
    max: int, seed: int, batch_size: int, device_id: int
) -> Callable:
    """Compilation wrapper for functions that allow them to be executed on a CUDA-compatible GPU.

    Args:
        max (int): The maximum number of blocks/threads to initialize. Primarily affects the RNG state generation step. Generally calculated as batches * batch_size.
        seed (int): The global seed to use for RNG generation. Might need to be changed if resuming training from an existing save state to prevent re-seen data.
        batch_size (int): The expected batch size to be used for training. Will be used to compile the kernel for execution on a corresponding number of blocks.
        device_id (int): The CUDA device to compile simulations for. This is here to ensure that the simulations and PyTorch training run on the same GPU, as it would default to device ID 0 otherwise.

    Returns:
        Callable: Decorator function for JIT compilation to CUDA-compatible GPUs.
    """
    rng = create_xoroshiro128p_states(n=max, seed=seed)

    # in both cases, 'rng.shape[0]' equals 'max'
    def decorator(func):
        args = len(signature(func).parameters)
        func = cuda.jit(device=True)(func)

        def kernel(rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4):
            idx = cuda.grid(1)
            if idx < toidx - fromidx:
                func(rng, fromidx, toidx, idx, d_ary1, d_ary2, d_ary3, d_ary4)

        kernel = cuda.jit(kernel)

        def wrapper(fromidx, toidx, ary1, ary2, ary3, ary4):
            if device_id is not None:
                cuda.select_device(device_id)
            if (
                fromidx < 0
                or toidx > max
                or toidx > ary1.shape[0]
                or toidx > ary2.shape[0]
                or toidx > ary3.shape[0]
                or toidx > ary4.shape[0]
            ):
                raise ValueError("index out of bound")
            d_ary1 = cuda.to_device(ary1[fromidx:toidx])
            d_ary2 = cuda.to_device(ary2[fromidx:toidx])
            d_ary3 = cuda.to_device(ary3[fromidx:toidx])
            d_ary4 = cuda.to_device(ary4[fromidx:toidx])
            blocks = (toidx - fromidx + batch_size - 1) // batch_size
            threads = batch_size
            kernel[blocks, threads](rng, fromidx, toidx, d_ary1, d_ary2, d_ary3, d_ary4)
            cuda.synchronize()
            # import numpy as np
            # print(np.unique(d_ary1.copy_to_host(), return_counts=True))
            # raise Exception("Check")
            # d_ary1.copy_to_host(ary1[fromidx:toidx])
            # d_ary2.copy_to_host(ary2[fromidx:toidx])
            # d_ary3.copy_to_host(ary3[fromidx:toidx])
            # d_ary4.copy_to_host(ary4[fromidx:toidx])
            return d_ary1.copy_to_host()

        return wrapper

    return decorator
