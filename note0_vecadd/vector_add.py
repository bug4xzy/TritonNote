# vector_add.py

import torch
import triton
import triton.language as tl

def vector_add(x, y, out):
    for i in range(x.size(0)):
        out[i] = x[i] + y[i]

@triton.jit
def vector_add_kernel(
    x_ptr,# 向量x的指针
    y_ptr,# 向量y的指针
    out_ptr,# 向量out的指针
    n_elements,# 向量长度
    BLOCK_SIZE: tl.constexpr,# block大小
):
    pid = tl.program_id(axis = 0)# 获取当前program全局索引

    block_start = pid * BLOCK_SIZE#获取当前block的起始索引  
    offsets = block_start + tl.arange(0, BLOCK_SIZE)# 获取当前block的结束索引
    mask = offsets < n_elements# 掩码，防止越界

    #tl.load:从指针指向的内存位置加载数据
    x = tl.load(x_ptr + offsets, mask=mask)# load向量x
    y = tl.load(y_ptr + offsets, mask=mask)# load向量y

    out = x + y# 加法

    #tl.store:将数据存储到指针指向的内存位置
    tl.store(out_ptr + offsets, out, mask=mask)# load 向量out

def vector_add_launcher(x, y):
    assert x.device == y.device
    assert x.dtype == y.dtype
    assert x.is_cuda
    assert y.is_cuda
    assert x.numel() == y.numel()

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)#grid 
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE = BLOCK_SIZE)
    return out

def test_vector_add():
    x = torch.randn(100000, device="cuda")
    y = torch.randn(100000, device="cuda")
    out = vector_add_launcher(x, y)
    if torch.allclose(out, x + y):
        print("test_vector_add passed")
    else:
        print("test_vector_add failed")



if __name__ == "__main__":
    test_vector_add()
