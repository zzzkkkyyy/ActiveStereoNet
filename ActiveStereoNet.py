# Reference: Similar Pyramid Stereo Matching Network

import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc
import argparse

siamese_channels = 128
batch_size = 16
length = 192
iterations = 2000
initial_lr = 1e-3
disparity_range = 8

def residual_block(image):
    layer1 = tf.layers.conv2d(image, filters=siamese_channels, kernel_size=3, padding='same')
    layer1 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer1))
    layer2 = tf.layers.conv2d(layer1, filters=siamese_channels, kernel_size=3, padding='same')
    return tf.nn.leaky_relu(image + layer2)

def siamese_network(image):
    layer = image
    for i in range(3):
        layer = tf.layers.conv2d(layer, filters=(2 ** (i - 2)) * siamese_channels, kernel_size=3, padding='same', strides=2)
        layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
    for i in range(3):
        layer = residual_block(layer)
    return layer

def cost_volume(left_image, right_image):
    cost_volume_list = []
    constant_disp_shape = right_image.get_shape().as_list()

    for disp in range(disparity_range):
        right_moved = image_bias_move_v2(right_image, tf.constant(disp, dtype=tf.float32, shape=constant_disp_shape))
        tf.summary.image('right_siamese_moved', right_moved[:, :, :, :3], 2)
        cost_volume_list.append(tf.concat([left_image, right_moved], axis=-1))
    cost_volume = tf.stack(cost_volume_list, axis=1)

    for i in range(4):
        cost_volume = tf.layers.conv3d(cost_volume, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
        cost_volume = tf.nn.leaky_relu(tf.layers.batch_normalization(cost_volume))
    cost_volume = tf.layers.conv3d(cost_volume, filters=1, kernel_size=3, padding='same', strides=1)
    cost_volume = tf.nn.dropout(cost_volume, keep_prob=0.9)

    disparity_volume = tf.reshape(tf.tile(tf.expand_dims(tf.range(disparity_range), axis=1), [1, constant_disp_shape[1] * constant_disp_shape[2] * cost_volume.get_shape().as_list()[-1]]), [1, -1])
    disparity_volume = tf.reshape(tf.tile(disparity_volume, [batch_size, 1]), [-1] + cost_volume.get_shape().as_list()[1: ])

    new_batch_slice = []
    batch_slice = tf.unstack(cost_volume, axis=0)
    for item in batch_slice:
        new_batch_slice.append(tf.nn.softmax(-item, axis=0))

    return tf.reduce_sum(tf.to_float(disparity_volume) * tf.stack(new_batch_slice, axis=0), axis=1)

def invalidation_network(image):
    layer = image
    for i in range(3):
        with tf.variable_scope('conv' + str(i), reuse=tf.AUTO_REUSE):
            layer = tf.layers.conv2d(layer, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
            layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
    output = tf.layers.conv2d(layer, filters=1, kernel_size=3, padding='same', strides=1)
    return output

def active_stereo_net(left_input, right_input):
    with tf.variable_scope('first_part', reuse=tf.AUTO_REUSE):
        left_siamese = siamese_network(left_input)
        right_siamese = siamese_network(right_input)
        tf.summary.image('left_siamese', left_siamese[:, :, :, :3], 1)
        tf.summary.image('right_siamese', right_siamese[:, :, :, :3], 1)

    with tf.variable_scope('second_part', reuse=tf.AUTO_REUSE):
        cost_map = cost_volume(left_siamese, right_siamese)
        new_shape = cost_map.get_shape().as_list()
        new_shape[1] *= 8
        new_shape[2] *= 8
        cost_map = tf.image.resize_images(cost_map, [new_shape[1], new_shape[2]])

    with tf.variable_scope('third_part', reuse=tf.AUTO_REUSE):
        layer = tf.concat([cost_map, left_input], axis=-1)
        for i in range(2):
            layer = tf.layers.conv2d(layer, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
            layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
        for i in range(3):
            layer = residual_block(layer)
        output = tf.layers.conv2d(layer, filters=1, kernel_size=3, padding='same', strides=1)

    return tf.add(output, cost_map)
    
def image_bias_move_v2(image, disparity_map):
    image = tf.pad(image, paddings=[[0, 0], [0, 0], [1, 1], [0, 0]])
    disparity_map = tf.pad(disparity_map, paddings=[[0, 0], [0, 0], [1, 1], [0, 0]])

    # create fundamental matrix
    each_1d_arr = tf.range(image.get_shape()[2])
    each_2d_arr = tf.tile(tf.expand_dims(each_1d_arr, axis=0), [image.get_shape()[1], 1])
    each_batch_2d_arr = tf.tile(tf.expand_dims(each_2d_arr, axis=0), [batch_size, 1, 1])
    each_batch_2d_arr = tf.to_float(each_batch_2d_arr)

    # sub/add bias value
    if len(disparity_map.get_shape().as_list()) == 3:
        tf.expand_dims(disparity_map, axis=-1)
    biased_batch_2d_arr = tf.clip_by_value(tf.to_float(each_batch_2d_arr - disparity_map[:, :, :, 0]), 0., tf.to_float(image.get_shape()[2] - 1))

    # set start index for each batch and row
    initial_arr = tf.tile(tf.expand_dims(tf.range(image.get_shape()[1] * batch_size) * image.get_shape()[2], axis=-1), [1, image.get_shape()[2]])

    # finally add together without channels dim
    biased_batch_2d_arr_high = tf.clip_by_value(tf.to_int32(tf.floor(biased_batch_2d_arr + 1)), 0, image.get_shape()[2] - 1)
    biased_batch_2d_arr_low = tf.clip_by_value(tf.to_int32(tf.floor(biased_batch_2d_arr)), 0, image.get_shape()[2] - 1)

    index_arr_high = tf.to_int32(tf.reshape(initial_arr, [-1])) + tf.reshape(biased_batch_2d_arr_high, [-1])
    index_arr_low = tf.to_int32(tf.reshape(initial_arr, [-1])) + tf.reshape(biased_batch_2d_arr_low, [-1])
    index_arr = tf.to_float(tf.reshape(initial_arr, [-1])) + tf.reshape(biased_batch_2d_arr, [-1])

    weight_low = tf.to_float(index_arr_high) - tf.to_float(index_arr)
    weight_high = tf.to_float(index_arr) - tf.to_float(index_arr_low)

    # consider channel
    weight_low = tf.expand_dims(weight_low, axis=-1)
    weight_high = tf.expand_dims(weight_high, axis=-1)
    n_image = tf.reshape(image, [-1, image.get_shape()[-1]])
    new_image = weight_low * tf.gather(n_image, index_arr_low) + weight_high * tf.gather(n_image, index_arr_high)

    return tf.reshape(new_image, image.get_shape())[:, :, 1: -1, :]

# add LCN, window-optimizer and invalidation net later
def loss_func(left_input, right_input, disparity_map):
    right = image_bias_move_v2(right_input, disparity_map)
    left = left_input

    left_2 = image_bias_move_v2(left_input, -disparity_map)
    right_2 = right_input

    tf.summary.image('right_input', right_input, 1)
    tf.summary.image('right_moved', right, 1)
    tf.summary.image('disparity', disparity_map, 1)
    l = tf.abs(right - left) + tf.abs(right_2 - left_2)
    return l

class batch_dataset:
    def __init__(self, batch_size=batch_size):
        self.current_index = 2000
        self.batch_size = batch_size

        self.training_dataset_left = []
        self.training_dataset_right = []
        self.test_dataset_left = []
        self.test_dataset_right = []

        self.left_l = []
        self.right_l = []
        
        path = "/home/zhoukeyang/ActiveStereoNet/StereoImages/*.png"
        file_list = []
        file_list.extend(glob.glob(path))
        if not file_list:
            print("file not found")
            return

        for f in file_list:
            file_name = os.path.split(f)[-1]
            if file_name.count('_') != 1:
                continue
            image = np.array(misc.imread(f, mode='RGB'))
            half_len = (image.shape)[1] // 2
            im_left = image[:, 0: half_len, :]
            im_right = image[:, half_len: (image.shape)[1], :]

            self.left_l.append(im_left)
            self.right_l.append(im_right)
        
        self.training_dataset_left = self.left_l[self.batch_size: ]
        self.training_dataset_right = self.right_l[self.batch_size: ]
        self.test_dataset_left = self.left_l[: self.batch_size]
        self.test_dataset_right = self.right_l[: self.batch_size]

        del self.left_l, self.right_l

    def read_next_batch(self, is_training=True):
        if is_training is True:
            if self.current_index + self.batch_size > len(self.training_dataset_left):
                training_index = list(np.arange(len(self.training_dataset_left)))
                random.shuffle(training_index)
                self.training_dataset_left = [self.training_dataset_left[item] for item in training_index]
                self.training_dataset_right = [self.training_dataset_right[item] for item in training_index]
                self.current_index = 0
            self.current_index += self.batch_size

            left_ll = self.training_dataset_left[self.current_index - self.batch_size: self.current_index]
            right_ll = self.training_dataset_right[self.current_index - self.batch_size: self.current_index]
        else:
            left_ll = self.test_dataset_left
            right_ll = self.test_dataset_right

        left = np.stack([misc.imresize(item, (length, length)) for item in left_ll], axis=0)
        right = np.stack([misc.imresize(item, (length, length)) for item in right_ll], axis=0)

        left = (left - np.min(left)) / (np.max(left) - np.min(left)) * 2 - 1
        right = (right - np.min(right)) / (np.max(right) - np.min(right)) * 2 - 1

        return left, right

def main(argv=None):
    print("constructing network...")
    x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, length, length, 3), name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, shape=(batch_size, length, length, 3), name='y_placeholder')
    disparity = active_stereo_net(x_placeholder, y_placeholder)
    global_steps = 0
    learning_rate = initial_lr

    print("constructing loss function and optimizer...")
    with tf.name_scope('loss'):
        disparity_mean = tf.reduce_mean(disparity)
        loss = tf.reduce_mean(loss_func(x_placeholder, y_placeholder, disparity))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
        train_op = optimizer.apply_gradients(grads)

    print("init reading...")
    reader = batch_dataset(batch_size)

    print("begin running...")
    merged = tf.summary.merge_all()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("/home/zhoukeyang/ActiveStereoNet/output/board", sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        global_steps += 1
        left_train, right_train = reader.read_next_batch(is_training=True)
        summary, Loss, m, _ = sess.run([merged, loss, disparity_mean, train_op], feed_dict={x_placeholder: left_train, y_placeholder: right_train})
        if i % 20 == 0:
            print("Step: {} ------------> training loss: {}, average disparity: {}".format(i, Loss, m))
            summary_writer.add_summary(summary, i)
        if i % 100 == 0:
            left_test, right_test = reader.read_next_batch(is_training=False)
            Loss, m, _ = sess.run([loss, disparity_mean, tf.no_op()], feed_dict={x_placeholder: left_test, y_placeholder: right_test})
            print("Step: {} ------------> test loss: {}, average disparity: {}".format(i, Loss, m))

    return


if __name__ == "__main__":
    tf.app.run()
