from numba import cuda
import numpy as np


@cuda.jit
def plus(a_in_gpu):
    cuda.atomic.add(a_in_gpu, (0, 0), 1)


if __name__ == "__main__":
    a = np.array([[0]], dtype=np.int_)
    print(a.shape)
    a_in_gpu = cuda.to_device(a)
    cuda.synchronize()
    plus[2048, 1024](a_in_gpu)
    cuda.synchronize()
    a = a_in_gpu.copy_to_host()
    print(a[0, 0])
