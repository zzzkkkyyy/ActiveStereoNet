# Reference: Similar Pyramid Stereo Matching Network

import tensorflow as tf
import numpy as np
import os, random, glob
import scipy.misc as misc
from skimage import io
from stn import spatial_transformer_network as transformer

siamese_channels = 64
batch_size = 2
length = 64
iterations = 20000
initial_lr = 1e-3
disparity_range = 16

# reference: https://github.com/mrharicot/monodepth/blob/master/bilinear_sampler.py
def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch    = tf.shape(input_images)[0]
        _height       = tf.shape(input_images)[1]
        _width        = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output


def residual_block(image):
    layer1 = tf.layers.conv2d(image, filters=siamese_channels, kernel_size=3, padding='same')
    layer1 = tf.nn.leaky_relu(tf.layers.batch_normalization(layer1))
    layer2 = tf.layers.conv2d(layer1, filters=siamese_channels, kernel_size=3, padding='same')
    return tf.add(layer2, image)

def siamese_network(image):
    #layer = tf.stack([tf.image.per_image_standardization(item) for item in tf.unstack(image, num=batch_size)])
    layer = image
    for i in range(3):
        #with tf.variable_scope('conv' + str(i), reuse=tf.AUTO_REUSE):
        layer = tf.layers.conv2d(layer, filters=(2 ** (i - 2)) * siamese_channels, kernel_size=5, padding='same', strides=2)
    for i in range(3):
        #with tf.variable_scope('residual' + str(i), reuse=tf.AUTO_REUSE):
        layer = residual_block(layer)
    return layer

def cost_volume(left_image, right_image):
    cost_volume_list = []
    for disp in range(disparity_range):
        right_moved = image_bias_move(right_image, tf.constant(disp + 1, shape=right_image.get_shape()), batch_size)
        cost_volume_list.append(tf.concat([left_image, right_moved], axis=-1))
    cost_volume = tf.stack(cost_volume_list, axis=1)
    
    for i in range(4):
        cost_volume = tf.layers.conv3d(cost_volume, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
        cost_volume = tf.nn.leaky_relu(cost_volume)
        #cost_volume = tf.nn.leaky_relu(tf.layers.batch_normalization(cost_volume))
    cost_volume = tf.layers.conv3d(cost_volume, filters=1, kernel_size=3, padding='same', strides=1)
    cost_volume = tf.nn.dropout(cost_volume, keep_prob=0.5)
    
    return tf.reduce_sum(cost_volume * tf.nn.softmax(-cost_volume, axis=1), axis=1)

def cost_volume_v2(left_image, right_image):
    new_volume = []
    for bs in range(batch_size):
        new_per_volume = []
        left_each_split = left_image[bs, :, :, :]
        for disp in range(disparity_range):
            right_each_split_0 = right_image[bs: bs + 1, :, :, :]
            right_each_split = image_bias_move(right_each_split_0, tf.constant(disp + 1, shape=right_each_split_0.get_shape()), 1)
            new_per_volume.append(tf.concat([left_each_split, tf.squeeze(right_each_split, axis=0)], axis=-1))
        new_volume.append(tf.stack(new_per_volume))
    cost_volume = tf.stack(new_volume)

    for i in range(4):
        cost_volume = tf.layers.conv3d(cost_volume, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
        cost_volume = tf.nn.leaky_relu(tf.layers.batch_normalization(cost_volume))
    cost_volume = tf.layers.conv3d(cost_volume, filters=1, kernel_size=3, padding='same', strides=1)
    cost_volume = tf.nn.dropout(cost_volume, keep_prob=0.5)

    layer_batch = tf.unstack(cost_volume, num=batch_size)
    l = []
    for bs in layer_batch:
        bs = tf.nn.softmax(-bs, axis=0)
        disp_list = tf.unstack(bs, num=disparity_range, axis=0)
        sum_layer = tf.zeros(disp_list[0].get_shape())
        for i in range(len(disp_list)):
            sum_layer += (i + 1) * disp_list[i]
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
        #tf.summary.image("left_siamese", left_siamese[:,:,:,:1], 2)
        #tf.summary.image("right_siamese", right_siamese[:,:,:,:1], 2)
    
    with tf.variable_scope('second_part', reuse=tf.AUTO_REUSE):
        cost_map = cost_volume(left_siamese, right_siamese)
        for i in range(3):
            cost_map = tf.layers.conv2d_transpose(cost_map, filters=1, kernel_size=3, padding='same', strides=2)
        """
        new_shape = cost_map.get_shape().as_list()
        new_shape[1] *= 8
        new_shape[2] *= 8
        cost_map = tf.image.resize_images(cost_map, [new_shape[1], new_shape[2]])
        """
        tf.summary.image("cost_map", cost_map, 2)
        """
        invalid_map = tf.squeeze(invalidation_network(cost_input))
        invalid_map = tf.image.resize_bilinear(invalid_map, 8 * invalid_map.shape())
        """

    with tf.variable_scope('third_part', reuse=tf.AUTO_REUSE):
        layer = tf.concat([cost_map, left_input], axis=-1)
        for i in range(2):
            layer = tf.layers.conv2d(layer, filters=siamese_channels, kernel_size=3, padding='same', strides=1)
            layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
        for i in range(3):
            layer = residual_block(layer)
        output = tf.layers.conv2d(layer, filters=1, kernel_size=3, padding='same', strides=1)
        """
        layer = tf.concat(tf.expand_dims(invalid_map, 2), layer, axis=2)
        for i in range(4):
            layer = tf.layers.conv2d(layer, filter=siamese_channels, kernel_size=3, padding='same', strides=1)
            layer = tf.nn.leaky_relu(tf.layers.batch_normalization(layer))
        output_invalid = tf.layers.conv2d(layer, filter=1, kernel_size=3, padding='same', strides=1)
        """
    return tf.add(output, cost_map)

def image_bias_move(image, disparity_map, batch_size=batch_size):
    """
    new_shape = [batch_size, image.get_shape()[1], image.get_shape()[2], image.get_shape()[3]]
    per_image_length = image.get_shape()[1] * image.get_shape()[2] * image.get_shape()[3]
    arr = tf.reshape(tf.tile(tf.expand_dims(tf.range(image.get_shape()[1]), axis=1), [1, image.get_shape()[-1]]), [-1])
    per_arr = tf.reshape(tf.tile(tf.expand_dims(arr, axis=1), [1, image.get_shape()[1]]), [1, per_image_length])
    index_arr = tf.clip_by_value(tf.to_int32(tf.tile(per_arr, [batch_size, 1])) - tf.to_int32(tf.reshape(disparity_map, [batch_size, per_image_length])), 0, image.get_shape()[1] - 1)
    initial_arr = tf.reshape(tf.tile(tf.expand_dims(tf.range(per_image_length * batch_size, delta=image.get_shape()[2] * image.get_shape()[3]), axis=1), (1, image.get_shape()[2] * image.get_shape()[3])), [batch_size, per_image_length])
    #batch_arr = tf.tile(tf.expand_dims(initial_arr, axis=0), [batch_size, 1]) + tf.tile(tf.expand_dims(tf.range(batch_size * per_image_length, delta=per_image_length), axis=1), [1, per_image_length])
    #batch_arr = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size * per_image_length, delta=per_image_length), axis=1), [1, per_image_length]), [batch_size, per_image_length])
    index_array = tf.reshape(initial_arr + index_arr, [-1])
    index_array = tf.expand_dims(index_array, axis=1)
    return tf.reshape(tf.gather_nd(tf.reshape(image, [-1]), index_array), new_shape)
    """
    return bilinear_sampler_1d_h(image, disparity_map)

# add LCN, window-optimizer and invalidation net later
def loss_func(left_input, right_input, disparity_map):
    right = stn(image_bias_move(right_input, disparity_map))
    left = stn(left_input)
    tf.summary.image('left', left, 2)
    tf.summary.image('right', right, 2)
    tf.summary.image('disparity', disparity_map, 2)
    #tf.summary.image('result', right, 2)
    l = tf.abs(right - left)
    return l
    
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
        W_fc1 = tf.Variable(tf.zeros([H * W * C, n_fc]), name='W_fc1')
        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(tf.zeros([B, H * W * C]), W_fc1) + b_fc1
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
            im_left = np.expand_dims(misc.imresize(image[:, 0: half_len], (length, length)), -1)
            im_right = np.expand_dims(misc.imresize(image[:, half_len: (image.shape)[-1]], (length, length)), -1)
            im_left = (im_left - np.min(im_left)) / (np.max(im_left) - np.min(im_left))
            im_right = (im_right - np.min(im_right)) / (np.max(im_right) - np.min(im_right))
            im_left = (im_left - 0.5) / 0.5
            im_right = (im_right - 0.5) / 0.5
            self.l.append([im_left, im_right])

        print("dataset length:", len(self.l))
        random.shuffle(self.l)
        self.test_dataset = self.l[0: batch_size]
        self.training_dataset = self.l[self.batch_size:]

    def read_next_batch(self, is_training=True):
        if is_training is True:
            if self.current_index + self.batch_size > len(self.training_dataset):
                print("**********all elements relisted**********")
                random.shuffle(self.training_dataset)
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
    
    var_list = tf.trainable_variables()
    print("var list length:", len(var_list))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(loss_func(x_placeholder, y_placeholder, disparity))
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads)

    print("begin running...")
    merged = tf.summary.merge_all()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("/output/board", sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        global_steps += 1
        left, right = reader.read_next_batch()
        summary, Loss, _ = sess.run([merged, loss, train_op], feed_dict={x_placeholder: left, y_placeholder: right})
        if (i + 1) % 25 == 0:
            print("iteration:", i + 1, "------------> training loss:", Loss)
        if (i + 1) % 50 == 0:
            #left_test, right_test = reader.read_next_batch(is_training=False)
            #summary, disp, loss_test, _ = sess.run([merged, disparity, loss, tf.no_op()], feed_dict={x_placeholder: left_test, y_placeholder: right_test})
            #print("iteration:", i + 1, "------------> testing loss:", loss_test)
            summary_writer.add_summary(summary, i + 1)
    return


if __name__ == "__main__":
    tf.app.run()