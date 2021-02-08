import jax.numpy as np
import jax
import numpy as onp

from jax import grad,jit,vmap

#max pool function
def max_pool(image, nrows, ncols): 
    #Splits into submatrix and output max value
    output=[]
    d,ow,oh = image.shape
    for i in range(image.shape[0]): #this is repeated for each filter. 
      array=image[i].reshape(ow,oh)
      r, h = array.shape
     
      temp=(array.reshape(h//nrows, nrows, -1, ncols)
                  .swapaxes(1, 2)
                  .reshape(-1, nrows, ncols))
        
      output.append(np.max(temp.reshape(-1,4),axis=1))
    output=np.array(output)
     
    output=output.reshape(image.shape[0],int(ow/nrows),int(oh/ncols))
    return output


def adj_matrix(row,height,padding):  
  padding=padding*2
  row=row+padding
  height=height+padding
  am =onp.zeros(((row)*(height),(row-padding)*(height-padding)))
  start=0
  end=3
  jump = row #The jump is a row length
  mod=0
  ctr=0
  for i in range(row-padding):
    for j in range(height-padding):
      am[start+mod:end+mod,ctr]=1
      am[start+mod+jump:end+mod+jump,ctr]=1
      am[start+mod+jump+jump:end+mod+jump+jump,ctr]=1
      mod+=1
      ctr+=1 
    start=end-1
    end =start+3

  return am 


def im2col(row,height,depth,padding):
  padded_image=onp.ones(((row+(padding*2))*(row+(padding*2)))).astype('float16')
  am=adj_matrix(row,height,padding)
  indices=am.T*padded_image
  window_indices=[]
  image_indices=[]
  for i in range(row*height):
    window_indices.append(onp.nonzero(indices[i])[0])
  window_indices=np.array(window_indices).T
  image_indices.append(window_indices)
  for i in range(1,depth): 
    window_indices=window_indices+((row+(padding*2))*(row+(padding*2)))
    image_indices.append(window_indices)
  return onp.array(image_indices)




def add_pad (image_,kernelw_,kernelh_):  #input image are channel,h,w
  #new add_pad
  d,h,w=image_.shape
  #d=1
  padw=onp.zeros((d,w,onp.divmod(kernelw_,2)[0]))

  padded=np.concatenate([padw,image_],axis=2) #you want to add, at the start if a 
  padded=np.concatenate([padded,padw[:]],axis=2)

  padh=onp.zeros((d,onp.divmod(kernelh_,2)[0],padded.shape[2]))

  padded=np.concatenate([padh,padded],axis=1)
  padded=np.concatenate([padded,padh],axis=1)
  return padded



def conv (input_image,parameter,im2col_matrix): #parameter will be shaped. [layer,weight/bias,filter#], on input the parameter will be just [weight/bias,filter#]
  d,h,w = input_image.shape
  kh,kw=parameter[0][0][0].shape #fetches the kernel size
  
  weights=parameter[0].reshape(-1,d*kh*kw) #changes the kernel into [filter,3*3] to linearlize. 

  input_image=add_pad(input_image,kh,kw) #add padding 
  input_image = input_image.flatten() 
  conv_image=input_image[im2col_matrix].T 
  conv_image=conv_image.reshape(w*h,kh*kh*d) #Flattens the kernel into a row, allowing a dot product to be performed on all the filters at once. 
  convolved=(np.dot(conv_image,weights.T) + parameter[1]).T
  convolved=convolved.reshape(len(parameter[0]),h,w)

  return convolved




def init_conv_parameters(filters,size,image_depth):  
  trainable_v=[]
  size_f=size[0]*size[1] 
  trainable_v.append([]) #new layer
  trainable_v[0].append(onp.random.rand(filters[0],image_depth, size[0],size[1])/100) #this assumes first channel is 1 so this will not work with RBG channels for now. 
  trainable_v[0].append(onp.random.randn(filters[0])/100)#bias
  #trainable_v[0].append(onp.zeros(filters[0]))#bias
  for i in range(1,len(filters)):  #for a given layer 
    trainable_v.append([])
    trainable_v[i].append(onp.random.rand(filters[i],filters[i-1],size[0],size[1])/100) #The kernel will be previous layer size. for now assume it's square. 
    trainable_v[i].append(onp.random.randn(filters[i])/100)
    #trainable_v[i].append(onp.zeros(filters[i]))
  return trainable_v


def init_parameters(shapes,input_shape=784):  
    onp.random.seed(1000)
    trainable_v=[[]]
    #first layer
    trainable_v[0].append(onp.random.randn(shapes[0],input_shape)/100) #input
    trainable_v[0].append( onp.random.randn (shapes[0])/100) #bbias 
    for i in range(1,len(shapes)): 
      trainable_v.append([]) #creates new layer?
      trainable_v[i].append(onp.random.randn(shapes[i],shapes[i-1])/100)
      trainable_v[i].append(onp.random.randn(shapes[i]))
    return trainable_v
