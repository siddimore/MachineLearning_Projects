# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops


ops.reset_default_graph()
# Start tensorflow session
sess = tf.Session()
iris = datasets.load_iris()

x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# declar parameters for our model

learning_rate = 0.05
batch_Size = 25

x_data = tf.placeholder(shape=[None,1], dtype = tf.float32)
y_target = tf.placeholder(shape=[None,1], dtype = tf.float32)

A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Write the formula for linear model y = Ax+b:
model_output = tf.add(tf.matmul(x_data,A) ,b)

# Declar L2 Loss function
loss = tf.reduce_mean(tf.square(y_target - model_output))

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

loss_vec = []

for i in range(100):
    rand_index = np.random.choice(len(x_vals),size=batch_Size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict= {x_data:rand_x, y_target:rand_y})
    temp_loss = sess.run(loss, feed_dict= {x_data:rand_x, y_target:rand_y})
    loss_vec.append(temp_loss)
    if (i+1) % 25 == 0:
        print('Step #' + str(i+1) + 'A =' + str(sess.run(A))
        + 'b = ' + str(sess.run(b)))
        print('Loss = ''' + str(temp_loss))
