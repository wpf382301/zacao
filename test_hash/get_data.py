import os
import tensorflow as tf


def get_zacao_input(file_dirs, label_dirs):
    img_list = []
    for _ in file_dirs:
        imgs = os.listdir(_)
        for i in imgs:
            img_dir = os.path.join(_, i)
            img_list.append(img_dir)
    with open(label_dirs, "r") as f:
        text = f.read()
    label_list = text.split(sep=',')
    label_list = list(map(int, label_list))
    print('There are %d images\nThere are %d labels' % (len(img_list), len(label_list)))
    return img_list, label_list


def get_batch_jpg(image_dirs, labels, image_w, image_h, batch_size, capacity):
    image_dirs = tf.cast(image_dirs, tf.string)
    labels = tf.cast(labels, tf.int32)
    input_queue = tf.train.slice_input_producer([image_dirs, labels], shuffle=True)
    labels = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image_jpg = tf.image.decode_jpeg(image_contents, channels=3)
    image_jpg = tf.image.resize_images(image_jpg, [image_h, image_w], method=tf.image.ResizeMethod.AREA)
    image_batch, label_batch = tf.train.batch([image_jpg, labels],
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
