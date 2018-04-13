import math
import os

import numpy as np
import tensorflow as tf
from PIL import Image

import get_data
import pre_model

N_CLASSES = 5
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.0001
train_dir = ['F:\\zacao\\train\\cier\\train',
             'F:\\zacao\\train\\huicai\\train',
             'F:\\zacao\\train\\suocao\\train',
             'F:\\zacao\\train\\yumi\\train',
             'F:\\zacao\\train\\zaoshuhe\\train']
label_dir = 'F:\\zacao\\train\\train_label.txt'
logs_train_dir = 'E:\\test'
test_dir = ['F:\\zacao\\val\\cier\\test',
            'F:\\zacao\\val\\huicai\\test',
            'F:\\zacao\\val\\suocao\\test',
            'F:\\zacao\\val\\yumi\\test',
            'F:\\zacao\\val\\zaoshuhe\\test']
test_label_dir = 'F:\\zacao\\val\\test_label.txt'


def run_training():
    train, train_label = get_data.get_zacao_input(train_dir, label_dir)
    train_batch, train_label_batch = get_data.get_batch_jpg(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    train_logits = pre_model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = pre_model.losses(train_logits, train_label_batch)
    train_op = pre_model.trainning(train_loss, learning_rate)
    train__acc = pre_model.evaluation(train_logits, train_label_batch)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()

    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 0
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %d' % step)
        step += 1

    try:
        while step < MAX_STEP:
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0 or step + 1 == MAX_STEP:
                print('Step %d, train loss = %.6f, train accuracy = %.6f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or step + 1 == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model')
                saver.save(sess, checkpoint_path, global_step=step)
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def start_test():
    with tf.Graph().as_default():
        test, test_label = get_data.get_zacao_input(test_dir, test_label_dir)
        test_batch, test_label_batch = get_data.get_batch_jpg(test, test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
        test_logits = pre_model.inference_without_dropout(test_batch, BATCH_SIZE, N_CLASSES)
        test_acc = pre_model.evaluation(test_logits, test_label_batch)

        num_test = 1000
        num_iter = int(math.ceil(num_test / BATCH_SIZE))
        step = 0

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            coord2 = tf.train.Coordinator()
            threads2 = tf.train.start_queue_runners(sess=sess, coord=coord2)
            try:
                count_acc = 0
                while step < num_iter:
                    if coord2.should_stop():
                        break
                    a, t = sess.run([test_batch, test_acc])
                    if step == 0:
                        print('Start test...')
                    count_acc += t
                    step += 1
                print('Test acc = %.2f%%' % (count_acc / step * 100))
            except tf.errors.OutOfRangeError:
                print('Done testing -- epoch limit reached')
            finally:
                coord2.request_stop()
            coord2.join(threads2)


def get_one_image(train):
    img_list = []
    for _ in train:
        imgs = os.listdir(_)
        for i in imgs:
            img_dir = os.path.join(_, i)
            img_list.append(img_dir)
    img_dir = np.random.choice(img_list)
    # image = Image.open(img_dir)
    # image.show()
    return img_dir


def evaluate_one_image():
    image_dir = str(get_one_image(test_dir))

    # image_dir = 'F:\\zacao\\val\\huicai\\test\\IMG_20160626_165905.jpg'
    print(image_dir)
    image_contents = tf.read_file(image_dir)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [IMG_W, IMG_H], method=tf.image.ResizeMethod.AREA)
    image = tf.reshape(tensor=image, shape=[1, IMG_W, IMG_H, 3])

    logit = pre_model.inference_without_dropout(image, 1, N_CLASSES)
    logit = tf.nn.softmax(logit)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        prediction = sess.run(logit)
        max_index = np.argmax(prediction)
        print('This is %d with possibility %.6f' % (max_index, prediction[:, max_index]))


if __name__ == '__main__':
    # start_test()
    run_training()
    # evaluate_one_image()
