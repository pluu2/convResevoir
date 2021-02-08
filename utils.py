import numpy as onp
from mlxtend.data import loadlocal_mnist
import platform



def to_categorical(data): 
    category=onp.zeros((len(data),10))
    for i in range(len(data)): 
        category[i][data[i]]=1
    return category

def tanh_act(x): 
    return np.tanh(x)
def sigmoid_act(x): 
  return jax.nn.sigmoid(x)
def softmax_act(x): 
    #return np.exp(x)/(np.sum(np.exp(x)))
    return jax.nn.softmax(x)
def binary_crossentropy(x,y): #x=input, y= target
    return -y*np.log(x)-(1-y)*np.log(1-x)
    #return jax.nn.binary_crossentropy(x,y)
def relu_act(x): 
  return jax.nn.relu(x)
def normalize(x): 
  return jax.nn.normalize(x,axis=0)


def load_mnist(directory):
    if not platform.system() == 'Windows':
        mnist_train_data, mnist_train_labels = loadlocal_mnist(
                images_path=directory+'train-images.idx3-ubyte', 
                labels_path=directory+'train-labels.idx1-ubyte')

    else:
        mnist_train_data, mnist_train_labels = loadlocal_mnist(
                images_path='train-images.idx3-ubyte', 
                labels_path='train-labels.idx1-ubyte')
   

    if not platform.system() == 'Windows':
        mnist_test_data, mnist_test_labels = loadlocal_mnist(
                images_path=directory+'t10k-images.idx3-ubyte', 
                labels_path=directory +'t10k-labels.idx1-ubyte')

    else:
        mnist_test_data, mnist_test_labels = loadlocal_mnist(
                images_path='t10k-images.idx3-ubyte', 
                labels_path='t10k-labels.idx1-ubyte')

    return (mnist_train_data,mnist_train_labels),(mnist_test_data,mnist_test_labels)