# runtime build helper using PyTorch's cpp_extension
import os
from torch.utils.cpp_extension import load

this_dir = os.path.dirname(__file__)
cpp = os.path.join(this_dir, 'postproc_cuda.cpp')
cu = os.path.join(this_dir, 'postproc_cuda.cu')

if __name__ == '__main__':
    print('Building CUDA extension (this may take a minute)...')
    load(name='postproc_cuda', sources=[cpp, cu], verbose=True)
    print('Built postproc_cuda extension.')
