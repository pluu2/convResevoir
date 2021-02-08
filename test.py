import numpy as np
import jax.numpy as jnp 
import jaxconv 
from jaxconv import * 

#tests: 
print('Test 1: generate am')
try: 
    am = adj_matrix(28,28,1) 
    print("passed")
except: 
    print("failed")


print('Test 2: generate im2col') 
try: 
    im2col_matrix = im2col(28,28,1,1)
    print('passed')
except: 
    print('failed')


print('Test 3: generate conv weights') 
try: 
    params = init_conv_parameters([1,1],[3,3],1) 
    print('passed') 
except: 
    print('failed') 

