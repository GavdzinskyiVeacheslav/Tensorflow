import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.executing_eagerly())
# True
a = tf.constant(1)
print(a)
# tf.Tensor(1, shape=(), dtype=int32)
a = tf.constant(1, shape=(1, 1))
b = tf.constant([1, 2, 3, 4])
print(a)
# tf.Tensor([[1]], shape=(1, 1), dtype=int32)
print(b)
# tf.Tensor([1 2 3 4], shape=(4,), dtype=int32)
c = tf.constant([[1, 2],
                [3, 4], # [3, 4, 5] will result in an error
                [5, 6]], dtype=tf.float16)
print(c)
# tf.Tensor(
# [[1. 2.]
#  [3. 4.]
#  [5. 6.]], shape=(3, 2), dtype=float16)
a2 = tf.cast(a, dtype=tf.float32)
print(a2)
# tf.Tensor([[1.]], shape=(1, 1), dtype=float32)
print(a)
b1 = np.array(b)
print(b1)
# [1 2 3 4]
b2 = b.numpy()
print(b2)
# [1 2 3 4]
v1 = tf.Variable(-1.2)
v2 = tf.Variable([4, 5, 6, 7], dtype=tf.float32)
v3 = tf.Variable(b)
print(v1, v2, v3, sep="\n\n")
# <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-1.2>
# <tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([4., 5., 6., 7.], dtype=float32)>
# <tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([1, 2, 3, 4])>
v1.assign(0)
print(v1)
# <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>
v2.assign([0, 1, 6, 7]) # v2.assign([0, 1, 6]) will result in an error - disproportion
print(v2)
# <tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([0., 1., 6., 7.], dtype=float32)>
v3.assign_add([1, 1, 1, 1])
print(v3)               # adding values
# <tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([2, 3, 4, 5])>
v1.assign_sub(5)
print(v1)               # subtraction of values
# <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-5.0>
'''
v3 = tf.Variable([-1, -2, -3, -4]) 
print(v3)
'''
# <tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([-1, -2, -3, -4])>
v4 = tf.Variable(v1)
print(v4.shape)
# ()
print(v2.shape)
# (4,)
val_0 = v3[0]             # first element
val_12 = v3[1:3]          # item two to three
print(v3, val_0, val_12, sep="\n")
# <tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([2, 3, 4, 5])>
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor([3 4], shape=(2,), dtype=int32)
val_0.assign(10)
print(v3, val_0, val_12, sep="\n")
# <tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([10,  3,  4,  5])>
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor([3 4], shape=(2,), dtype=int32)
x = tf.constant(range(10)) + 5
x_index = tf.gather(x, [0, 4])
print(x, x_index, sep="\n")
# tf.Tensor([ 5  6  7  8  9 10 11 12 13 14], shape=(10,), dtype=int32)
# tf.Tensor([5 9], shape=(2,), dtype=int32)
""" 
val_indx = v2[(1, 2)]  will result in an error
"""
v2 = tf.constant([[1, 2, 7], [3, 4, 8], [5, 6, 9]])
val_indx = v2[1, 2]
print(val_indx)
# tf.Tensor(8, shape=(), dtype=int32)
val_indx = v2[1][2]
print(val_indx)
# tf.Tensor(8, shape=(), dtype=int32)
val_indx = v2[0]
print(val_indx)
# tf.Tensor([1 2 7], shape=(3,), dtype=int32)
val_indx = v2[:, 1] #start:stop:step
print(val_indx)
# tf.Tensor([2 4 6], shape=(3,), dtype=int32)
val_indx = v2[:2, 1]
print(val_indx)
# tf.Tensor([2 4], shape=(2,), dtype=int32)
val_indx = v2[:2, 2]
print(val_indx)
# tf.Tensor([7 8], shape=(2,), dtype=int32)
val_indx = v2[:2, -1]
print(val_indx)
# tf.Tensor([7 8], shape=(2,), dtype=int32)
a = tf.constant(range(30))
print(a)
# tf.Tensor(
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29], shape=(30,), dtype=int32)
a = tf.constant(range(30))
b = tf.reshape(a, [5, 6]) # [5, 4] - will result in an error
print(b.numpy())
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]
#  [24 25 26 27 28 29]]
b = tf.reshape(a, [6, -1])
print(b.numpy())
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]
#  [25 26 27 28 29]]
b = tf.reshape(a, [6, -1])
b_T = tf.transpose(b, perm=[1, 0])
print(b_T.numpy())
# [[ 0  5 10 15 20 25]
#  [ 1  6 11 16 21 26]
#  [ 2  7 12 17 22 27]
#  [ 3  8 13 18 23 28]
#  [ 4  9 14 19 24 29]]














































































