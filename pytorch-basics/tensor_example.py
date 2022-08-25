import torch
import numpy as np


# 1. initialize a tensor
my_tensor = torch.tensor([[1,2,3],[4,5,6]], 
                         dtype=torch.float32,
                         device="cpu", requires_grad=True)


print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape) # [H,W]  
print(my_tensor.requires_grad)

# 2. Tensor from numpy array

np_array = np.array([1,2,3,4])
tensor = torch.from_numpy(np_array)
print(tensor)

# 3. Random or constant values
shape = (2,3)  
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
range_tensor = torch.arange(9)
print(rand_tensor)

# 4. operation on tensors
# 4.1 indexing
print(rand_tensor.shape) # (2,3) Height = 2 rows, Width = 3 columns 
take_row_0 = rand_tensor[0,:]  # from shape (2,3) -> (1,3) 
take_row_1 = rand_tensor[1,:]

# 5. joining tensors
concat_tensor = torch.cat([ones_tensor, zeros_tensor], dim=0)
print(concat_tensor) # (2,3) and (2,3) -> (4, 3)

concat_tensor = torch.cat([ones_tensor, zeros_tensor], dim=1)
print(concat_tensor) # (2,3) and (2,3) -> (2, 6)

# 6. operation
    # 6.1 matrix multiplication
matrix_A = torch.rand((3,3)) 
column_vector = torch.rand((3,1))
print(matrix_A)
print(column_vector)
out = torch.matmul(matrix_A, column_vector)
print(out)

    # 6.2 element-wise operation
ones_tensor = torch.ones((2,3))
twos_tensor = 2 * torch.ones((2,3))
    ## add
out = ones_tensor + twos_tensor
print(out)
    ## multiplication
out = ones_tensor * twos_tensor
print(out)

    # 6.3 broadcasting
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    # when shape is not equal
    # pytorch perform broadcasting
x1 = torch.rand((5,5))
x2 = torch.zeros((1,5)) 
out = x1 * x2  # x2 is copied 5 times 
print(out)  

a = torch.tensor([[1,2,3]])
b = torch.tensor([[0],[10],[20],[30]])
out = a + b
print(out)


# 7. tensor reshape
x = torch.arange(9)
print(x)
x_3x3 = x.reshape(3, 3)




# 8. use Nvidia GPU or Apple GPU
import platform
import re
this_device = platform.platform()
if torch.cuda.is_available():
    device = "cuda"
elif re.search("arm64", this_device):
    # use Apple GPU
    device = "mps"
else:
    device = "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], 
                         dtype=torch.float32,
                         device=device, requires_grad=True)

