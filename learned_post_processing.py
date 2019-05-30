import numpy as np
import tensorflow as tf
import glob
import random
import matplotlib.pyplot as plt

folder = 'data/learning'
size = 256

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

x = tf.layers.conv2d(fbp, filters=5, kernel_size=7, padding='SAME')

for i in range(5):
    dx = tf.layers.conv2d(x, filters=32, kernel_size=3, padding='SAME',
                          activation=tf.nn.relu)
    dx = tf.layers.conv2d(dx, filters=32, kernel_size=3, padding='SAME',
                          activation=tf.nn.relu)
    dx = tf.layers.conv2d(dx, filters=5, kernel_size=3, padding='SAME')
    x = x + dx

update = tf.layers.conv2d(x, filters=1, kernel_size=3, padding='SAME')
result = fbp + update

# Define loss
loss = tf.reduce_mean((result - phantom) ** 2)

# Define training algorithm
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta2=0.99)
train_op = optimizer.minimize(loss)

# Run training
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

data = DataReader(folder)

for i in range(1000000):
    fbp_arr, phantom_arr = data.get_next()
    fbp_arr[fbp_arr<0.001] = 0.0

    _, loss_result = session.run([train_op, loss],
                                 feed_dict={phantom_ph: phantom_arr,
                                            fbp_ph: fbp_arr})

    input_error = np.mean((fbp_arr - phantom_arr) ** 2)

    if i % 100 == 0:
        print(i, loss_result / input_error)

r, f, p = session.run([result, fbp, phantom],
                      feed_dict={phantom_ph: phantom_arr, fbp_ph: fbp_arr})
plt.figure()
plt.imshow(r.squeeze(), cmap='bone', clim=[0.02, 0.022])

plt.figure()
plt.imshow(f.squeeze(), cmap='bone', clim=[0.02, 0.022])

plt.figure()
plt.imshow(p.squeeze(), cmap='bone', clim=[0.02, 0.022])

plt.figure()
plt.imshow((r - p).squeeze(), cmap='bone', clim=[-0.001, 0.001])

plt.figure()
plt.imshow((f - p).squeeze(), cmap='bone', clim=[-0.001, 0.001])

plt.figure()
plt.imshow((r - p).squeeze() ** 2, cmap='bone', clim=[0, 0.003 ** 2])

plt.figure()
plt.imshow((f - p).squeeze() ** 2, cmap='bone', clim=[0, 0.003 ** 2])