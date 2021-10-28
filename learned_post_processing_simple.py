import numpy as np
import tensorflow as tf
import glob
import random
import matplotlib.pyplot as plt

folder = 'data/learning'
size = 128

# Read data
class DataReader(object):
    def __init__(self, folder):
        self.files = glob.glob(folder + '/*.npz')

    def get_next(self):
        loaded = np.load(random.choice(self.files))
        return (loaded['fbp'][None, ..., None],
                loaded['phantom'][None, ..., None])

# Define "operator" in TensorFlow
phantom_ph = tf.placeholder('float32', shape=[None, None, None, 1])
fbp_ph = tf.placeholder('float32', shape=[None, None, None, 1])

phantom = tf.image.resize_images(phantom_ph, [size, size])
fbp = tf.image.resize_images(fbp_ph, [size, size])

dx = tf.layers.conv2d(fbp, filters=1, kernel_size=13, padding='SAME',
                      use_bias=False)
result = fbp + 1e-2 * dx

# Define loss
loss = tf.reduce_mean((result - phantom) ** 2)

# Define training algorithm
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
train_op = optimizer.minimize(loss)

# Run training
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

data = DataReader(folder)

for i in range(1000000):
    fbp_arr, phantom_arr = data.get_next()

    _, loss_result = session.run([train_op, loss],
                                 feed_dict={phantom_ph: phantom_arr,
                                            fbp_ph: fbp_arr})

    input_error = np.mean((fbp_arr - phantom_arr) ** 2)
    print(loss_result / input_error)

result_result = session.run(result, feed_dict={fbp_ph: fbp_arr})
plt.figure()
plt.imshow(result_result.squeeze(), cmap='bone', clim=[0.015, 0.025])

plt.figure()
plt.imshow(fbp_arr.squeeze(), cmap='bone', clim=[0.015, 0.025])

plt.figure()
plt.imshow(phantom_arr.squeeze(), cmap='bone', clim=[0.015, 0.025])