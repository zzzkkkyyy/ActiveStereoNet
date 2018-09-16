# Reference: Similar Pyramid Stereo Matching Network

import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc
from skimage import io
from stn import spatial_transformer_network as transformer

siamese_channels = 32
batch_size = 8
length = 64
iterations = 200
initial_lr = 1e-4
disparity_range = 128

def residual_block(image):
    layer1 = tf.layers.conv2d(image, filters=siamese_channels, kernel_size=3, padding='same')
    layer1 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer1))
    layer2 = tf.layers.conv2d(layer1, filters=siamese_channels, kernel_size=3, padding='same')
    #layer3 = tf.layers.batch_normalization(layer2)
    return tf.add(layer2, image)

def siamese_network(image):
    layer = tf.stack([tf.image.per_image_standardization(item) for item in tf.unstack(image, num=batch_size)])
    for i in range(3):
        with tf.variable_scope('conv' + str(i), reuse=tf.AUTO_REUSE):
            layer = tf.layers.conv2d(layer, filters=siamese_channels, kernel_size=5, padding='same', strides=2)
            #layer = tf.nn.dropout(layer, keep_prob=0.5)
    for i in range(3):
        with tf.variable_scope('residual' + str(i), reuse=tf.AUTO_REUSE):
            layer = residual_block(layer)
    return layer

#change 3d-conv to 2d-conv, parameter numbers may be larger
def cost_volume(image):
    layer = image
    for i in range(5):
        with tf.variable_scope('conv' + str(i), reuse=tf.AUTO_REUSE):
            if i != 0:
                layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))    
            layer = tf.layers.conv2d(layer, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
    #return tf.layers.conv2d(layer, filters=1, kernel_size=1, padding='same', strides=1)

    layer_batch = tf.unstack(layer, num=batch_size)

    l = []
    for bs in layer_batch:
        each_layer = tf.split(bs, num_or_size_splits=layer.get_shape().as_list()[3], axis=2)
        sum_layer = tf.multiply(each_layer[0], tf.nn.softmax(each_layer[0]))
        for item in each_layer[1:]:
            sum_layer = tf.add(sum_layer, tf.multiply(-item, tf.nn.softmax(item)))
        l.append(sum_layer)

    return tf.stack(l)

def cost_volume_v2(left_image, right_image):
    new_volume = []
    for bs in range(batch_size):
        new_per_volume = []
        left_each_split = left_image[bs, :, :, :]
        for disp in range(disparity_range):
            right_each_split_0 = right_image[bs: bs + 1, :, :, :]
            right_each_split = image_bias_move(right_each_split_0, tf.constant(disp, shape=right_each_split_0.get_shape()), 1)
            new_per_volume.append(tf.concat([left_each_split, tf.squeeze(right_each_split, axis=0)], axis=-1))
        new_volume.append(tf.stack(new_per_volume))
    cost_volume = tf.stack(new_volume)
    
    for i in range(4):
        cost_volume = tf.layers.conv3d(cost_volume, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
        cost_volume = tf.nn.dropout(cost_volume, keep_prob=0.5)
        cost_volume = tf.nn.leaky_relu(tf.layers.batch_normalization(cost_volume))
    cost_volume = tf.layers.conv3d(cost_volume, filters=1, kernel_size=3, padding='same', strides=1)
    
    layer_batch = tf.unstack(cost_volume, num=batch_size)
    l = []
    for bs in layer_batch:
        disp_list = tf.unstack(bs, num=disparity_range)
        sum_layer = tf.zeros(disp_list[0].get_shape())
        for item in disp_list:
            sum_layer += (item) * tf.nn.softmax(-item)
        l.append(sum_layer)
    
    return tf.stack(l)

def invalidation_network(image):
    layer = image
    for i in range(3):
        with tf.variable_scope('conv' + str(i), reuse=tf.AUTO_REUSE):
            layer = tf.layers.conv2d(layer, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
            layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
    output = tf.layers.conv2d(layer, filters=1, kernel_size=3, padding='same', strides=1)
    return output

#input is grey-scale
def active_stereo_net(left_input, right_input):
    with tf.variable_scope('first_part', reuse=tf.AUTO_REUSE):
        left_siamese = siamese_network(left_input)
        right_siamese = siamese_network(right_input)
        tf.summary.image("left_siamese", left_siamese[:,:,:,:1], 2)
        tf.summary.image("right_siamese", right_siamese[:,:,:,:1], 2)
    
    with tf.variable_scope('second_part', reuse=tf.AUTO_REUSE):
        #cost_input = tf.concat(values=[left_siamese, right_siamese], axis=3)
        #cost_map = cost_volume(cost_input)
        cost_map = cost_volume_v2(left_siamese, right_siamese)
        new_shape = cost_map.get_shape().as_list()
        new_shape[1] *= 8
        new_shape[2] *= 8
        cost_map = tf.image.resize_images(cost_map, [new_shape[1], new_shape[2]])
        tf.summary.image("cost_map", cost_map, 2)
        #print("cost_map:", cost_map.get_shape())
        """
        invalid_map = tf.squeeze(invalidation_network(cost_input))
        invalid_map = tf.image.resize_bilinear(invalid_map, 8 * invalid_map.shape())
        """

    with tf.variable_scope('third_part', reuse=tf.AUTO_REUSE):
        layer = tf.concat([cost_map, left_input], axis=3)
        for i in range(2):
            with tf.variable_scope('conv' + str(i), reuse=tf.AUTO_REUSE):
                layer = tf.layers.conv2d(layer, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
                layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
        with tf.variable_scope('conv_end', reuse=tf.AUTO_REUSE):
            output = tf.layers.conv2d(layer, filters=1, kernel_size=3, padding='same', strides=1)
        """
        layer = tf.concat(tf.expand_dims(invalid_map, 2), layer, axis=2)
        for i in range(4):
            layer = tf.layers.conv2d(layer, filter=siamese_channels, kernel_size=3, padding='same', strides=1)
            layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
        output_invalid = tf.layers.conv2d(layer, filter=1, kernel_size=3, padding='same', strides=1)
        """
    return tf.add(output, cost_map)#, tf.add(tf.squeeze(output_invalid), invalid_map)
    
def image_bias_move(image, disparity_map, batch_size=batch_size):
    new_shape = [batch_size, image.get_shape()[1], image.get_shape()[2], image.get_shape()[3]]
    per_image_length = image.get_shape()[1] * image.get_shape()[2] * image.get_shape()[3]
    arr = tf.reshape(tf.tile(tf.expand_dims(tf.range(image.get_shape()[1]), axis=1), [1, image.get_shape()[-1]]), [-1])
    per_arr = tf.reshape(tf.tile(tf.expand_dims(arr, axis=1), [1, image.get_shape()[1]]), [1, per_image_length])
    index_arr = tf.clip_by_value(tf.to_int32(tf.tile(per_arr, [batch_size, 1])) - tf.to_int32(tf.reshape(disparity_map, [batch_size, per_image_length])), 0, image.get_shape()[1])
    batch_arr = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size * per_image_length, delta=per_image_length), axis=1), [1, per_image_length]), [batch_size, per_image_length])
    index_array = tf.reshape(batch_arr + index_arr, [batch_size * per_image_length])
    return tf.reshape(tf.gather(tf.reshape(image, [batch_size * per_image_length]), index_array), new_shape)

# add LCN, window-optimizer and invalidation net later
def loss_func(left_input, right_input, disparity_map):
    right = stn(image_bias_move(right_input, disparity_map))
    left = stn(left_input)
    tf.summary.image('left', left, 2)
    tf.summary.image('right', right, 2)
    tf.summary.image('disparity', disparity_map, 2)
    #l = tf.log(tf.abs((right - left) ** 2) + 1)
    l = tf.abs((right - left))
    return l
    zero_shape = [batch_size, l.get_shape()[1], l.get_shape()[2], l.get_shape()[3]]
    mask = tf.where(tf.greater(l, 1), l, tf.zeros(zero_shape))
    return mask
    
def stn(image):
    with tf.variable_scope('stn', reuse=tf.AUTO_REUSE):
        # params
        n_fc = 6
        B = batch_size
        [H, W, C] = image.get_shape().as_list()[1:]
        # identity transform
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32').flatten()
        # localization network
        W_fc1 = tf.Variable(tf.zeros([H*W*C, n_fc]), name='W_fc1')
        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(tf.zeros([B, H*W*C]), W_fc1) + b_fc1
        # spatial transformer layer
        h_trans = transformer(image, h_fc1)
        return h_trans

class batch_dataset:
    def __init__(self, batch_size=16):
        self.current_index = 0
        self.batch_size = batch_size
        self.training_dataset = []
        self.test_dataset = []
        self.l = []
        
        path = os.path.join("/data", "StereoImages", "*.png")
        #path = os.path.join("StereoImages", "*.png")
        file_list = []
        file_list.extend(glob.glob(path))
        if not file_list:
            print("file not found")
            return

        for f in file_list:
            file_name = os.path.split(f)[-1]
            if file_name.count('_') != 1:
                continue
            image = np.array(io.imread(f, as_gray=True))
            half_len = (image.shape)[-1] // 2
            #print(image.shape, half_len)
            im_left = np.expand_dims(misc.imresize(image[:, 0: half_len], (length, length)), -1)
            im_right = np.expand_dims(misc.imresize(image[:, half_len: (image.shape)[-1]], (length, length)), -1)
            im_left = (im_left - np.mean(im_left)) / np.std(im_left)
            im_right = (im_right - np.mean(im_right)) / np.std(im_right)
            #im_left = (im_left - np.min(im_left)) / (np.max(im_left) - np.min(im_left))
            #im_right = (im_right - np.min(im_right)) / (np.max(im_right) - np.min(im_right))
            self.l.append([im_left, im_right])

        print("dataset length:", len(self.l))
        random.shuffle(self.l)
        self.test_dataset = self.l[0: batch_size]
        self.training_dataset = self.l[self.batch_size:]
    
    def read_next_batch(self, is_training=True):
        if is_training is True:
            if self.current_index + self.batch_size > len(self.training_dataset):
                print("**********all elements relisted**********")
                #random.shuffle(self.training_dataset)
                self.current_index = 0
            self.current_index += self.batch_size
            ll = self.training_dataset[self.current_index - self.batch_size: self.current_index]
        else:
            ll = self.test_dataset[:]
        return np.array([list(item[0]) for item in ll]), np.array([list(item[1]) for item in ll])

def main(argv=None):
    print("init reading...")
    reader = batch_dataset(batch_size)
    
    print("constructing network...")
    x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, length, length, 1), name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, shape=(batch_size, length, length, 1), name='y_placeholder')
    disparity = active_stereo_net(x_placeholder, y_placeholder)
    global_steps = 0
    learning_rate = initial_lr

    with tf.name_scope('loss'):
        coeff = tf.constant(1.0, shape=disparity.get_shape())
        loss = tf.reduce_mean(loss_func(x_placeholder, y_placeholder, disparity * coeff))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(loss, var_list = tf.trainable_variables())
        train_op = optimizer.apply_gradients(grads)
        #tf.scalar_summary('loss', loss)

    merged = tf.summary.merge_all()
    
    print("begin running...")    
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("/output/board", sess.graph)
    sess.run(tf.global_variables_initializer())
    
    for i in range(iterations):
        global_steps += 1
        left, right = reader.read_next_batch()
        #print(np.array(left).shape)
        _, Loss, _ = sess.run([merged, loss, train_op], feed_dict={x_placeholder: left, y_placeholder: right})
        if (i + 1) % 2 == 0:
            print("iteration:", i + 1, "------------> training loss:", Loss)
        if (i + 1) % 10 == 0:
            left_test, right_test = reader.read_next_batch(is_training=False)
            summary, disp, loss_test, _ = sess.run([merged, disparity, loss, tf.no_op()], feed_dict={x_placeholder: left_test, y_placeholder: right_test})
            print("iteration:", i + 1, "------------> testing loss:", loss_test)
            summary_writer.add_summary(summary, i + 1)
            """
            save_num = 2
            for j in range(save_num):
                im = np.squeeze(disp[j], axis=-1)
                im_max_value = 255.0 / np.max(im)
                misc.imsave(os.path.join("/output/", "disparity_" + str(i + 1) + "_" + str(j + 1) + ".png"), (im * im_max_value).astype(np.uint8))
            print(save_num, "images are saved.")
            """
    return


if __name__ == "__main__":
    tf.app.run()