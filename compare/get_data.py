import os
import tensorflow as tf


def get_input(file_dirs, label_dirs):
    img_list = []
    count = len(os.listdir(file_dirs))
    for i in range(count):
        img_dir = os.path.join(file_dirs, '%d.png' % i)
        img_list.append(img_dir)
    with open(label_dirs, "r") as f:
        text = f.read()
    label_list = text.split(sep=',')
    label_list = list(map(int, label_list))
    print('There are %d images\nThere are %d labels' % (len(img_list), len(label_list)))
    return img_list, label_list


def get_num_batch(image, labels, image_w, image_h, batch_size, capacity):
    image = tf.cast(image, tf.string)
    labels = tf.cast(labels, tf.int32)
    input_queue = tf.train.slice_input_producer([image, labels], shuffle=False)
    labels = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=1)
    image = tf.image.resize_images(image, [image_h, image_w], method=tf.image.ResizeMethod.AREA)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, labels],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


def get_batch2(image_dirs, labels, batch_size, capacity):
    image_dirs = tf.cast(image_dirs, tf.string)
    labels = tf.cast(labels, tf.int32)
    input_queue = tf.train.slice_input_producer([image_dirs, labels], shuffle=False)
    labels = input_queue[1]
    images = input_queue[0]
    image_batch, label_batch = tf.train.batch([images, labels],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    return image_batch, label_batch
