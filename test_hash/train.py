import math
import os
from collections import Counter

import numpy as np
import pymysql
import tensorflow as tf
import time
from PIL import Image

from test_hash import get_data
from test_hash import model
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

N_CLASSES = 5
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 5001
learning_rate = 0.0001
train_dir = ['F:\\zacao\\train\\cier\\train',
             'F:\\zacao\\train\\huicai\\train',
             'F:\\zacao\\train\\suocao\\train',
             'F:\\zacao\\train\\yumi\\train',
             'F:\\zacao\\train\\zaoshuhe\\train']
label_dir = 'F:\\zacao\\train\\train_label.txt'
saved_logs_train_dir = 'F:\\zacao\\logs\\pre_train_4conv_256'
logs_train_dir = 'E:\\test'
test_dir = ['F:\\zacao\\val\\cier\\test',
            'F:\\zacao\\val\\huicai\\test',
            'F:\\zacao\\val\\suocao\\test',
            'F:\\zacao\\val\\yumi\\test',
            'F:\\zacao\\val\\zaoshuhe\\test']
test_label_dir = 'F:\\zacao\\val\\test_label.txt'
conn = pymysql.connect(host='localhost', port=3306, user='root', password='wpf382301', database='zacao',
                       charset='utf8')  # autocommit=True
cursor = conn.cursor()


def restore_pre_params():
    ckpt = tf.train.get_checkpoint_state(saved_logs_train_dir)
    reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
    w1 = reader.get_tensor("conv1/weights")
    b1 = reader.get_tensor("conv1/biases")
    w2 = reader.get_tensor("conv2/weights")
    b2 = reader.get_tensor("conv2/biases")
    w3 = reader.get_tensor("conv3/weights")
    b3 = reader.get_tensor("conv3/biases")
    w4 = reader.get_tensor("local3/weights")
    b4 = reader.get_tensor("local3/biases")
    w5 = reader.get_tensor("local4/weights")
    b5 = reader.get_tensor("local4/biases")
    print(w1.shape, b1.shape, w2.shape, b2.shape, w3.shape, b3.shape, w4.shape, b4.shape, w5.shape, b5.shape)
    return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5


def restore_pre_4conv_params():
    ckpt = tf.train.get_checkpoint_state(saved_logs_train_dir)
    reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
    w1 = reader.get_tensor("conv1/weights")
    b1 = reader.get_tensor("conv1/biases")
    w2 = reader.get_tensor("conv2/weights")
    b2 = reader.get_tensor("conv2/biases")
    w3 = reader.get_tensor("conv3/weights")
    b3 = reader.get_tensor("conv3/biases")
    w4 = reader.get_tensor("conv4/weights")
    b4 = reader.get_tensor("conv4/biases")
    w5 = reader.get_tensor("local3/weights")
    b5 = reader.get_tensor("local3/biases")
    w6 = reader.get_tensor("local4/weights")
    b6 = reader.get_tensor("local4/biases")
    print(w1.shape, b1.shape, w2.shape, b2.shape, w3.shape, b3.shape, w4.shape, b4.shape, w5.shape, b5.shape, w6.shape,
          b6.shape)
    return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6


def run_training(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6=None, b6=None, flag=False):
    step_0_time = time.time()
    train, train_label = get_data.get_zacao_input(train_dir, label_dir)
    train_batch, train_label_batch = get_data.get_batch_jpg(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    if not flag:
        print("conv3")
        train_logits = model.inference_3conv(train_batch, BATCH_SIZE, N_CLASSES, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
    else:
        print("conv4")
        train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6,
                                       b6)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
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
            _, tra_loss, tra_acc, summary_str = sess.run([train_op, train_loss, train__acc, summary_op])
            train_writer.add_summary(summary_str, step)

            if step % 50 == 0 or step + 1 == MAX_STEP:
                step__time = time.time()
                print("Step %d time %s" % (step, step__time - step_0_time))
                print('Step %d, train loss = %.6f, train accuracy = %.6f%%' % (step, tra_loss, tra_acc * 100.0))
            if step % 1000 == 0 or step + 1 == MAX_STEP:
                step_1_time = time.time()
                print("Step %d time %s" % (step, step_1_time - step_0_time))
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
        test_logits, _, _ = model.inference_without_dropout(test_batch, BATCH_SIZE, N_CLASSES)
        test_acc = model.evaluation(test_logits, test_label_batch)

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
                return
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


def start_test_and_show_error():
    test, test_label = get_data.get_zacao_input(test_dir, test_label_dir)
    test_batch, test_label_batch = get_data.get_batch2(test, test_label, 1, CAPACITY)

    image_contents = tf.read_file(test_batch[0])
    image_jpg = tf.image.decode_jpeg(image_contents, channels=3)
    image_jpg = tf.image.resize_images(image_jpg, [IMG_W, IMG_H], method=tf.image.ResizeMethod.AREA)
    image_jpg = tf.reshape(tensor=image_jpg, shape=[-1, IMG_W, IMG_H, 3])

    logit, _, _ = model.inference_3conv_without_dropout(image_jpg, 1, N_CLASSES)
    logit = tf.nn.softmax(logit)
    sess = tf.Session()

    saver = tf.train.Saver()
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
        return

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        step = 0
        while step < len(test):
            if coord.should_stop():
                break
            prediction, test_img_dir, real_label = sess.run([logit, test_batch, test_label_batch])
            max_index = np.argmax(prediction)
            if max_index != real_label[0]:
                print('错误路径:%s' % test_img_dir[0].decode())
                print('真实值:%d' % real_label[0])
                print('错误值:%d' % max_index)
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def get_one_image():
    img_list = []
    for _ in test_dir:
        imgs = os.listdir(_)
        for i in imgs:
            img_dir = os.path.join(_, i)
            img_list.append(img_dir)
    img_dir = np.random.choice(img_list)
    print(img_dir)
    image = Image.open(img_dir)
    image.show()
    return img_dir


def get_one_image_without_show():
    img_list = []
    for _ in test_dir:
        imgs = os.listdir(_)
        for i in imgs:
            img_dir = os.path.join(_, i)
            img_list.append(img_dir)
    img_dir = np.random.choice(img_list)
    return img_dir


def evaluate_one_image():
    image_dir = str(get_one_image())

    # image_dir = 'F:\\zacao\\train\\zaoshuhe\\train\\IMG_5369.jpg'
    image_contents = tf.read_file(image_dir)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [IMG_W, IMG_H], method=tf.image.ResizeMethod.AREA)
    image = tf.reshape(tensor=image, shape=[1, IMG_W, IMG_H, 3])
    logit, h_fc1_drop, hashcode = model.inference_without_dropout(image, 1, N_CLASSES)
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
        prediction, h_fc1_drop2, hashcode2 = sess.run([logit, h_fc1_drop, hashcode])
    print(h_fc1_drop2.shape)
    print(h_fc1_drop2)
    print(hashcode2.shape)
    print(hashcode2)
    print(prediction)
    max_index = np.argmax(prediction)
    print('This is %d with possibility %.6f' % (max_index, prediction[:, max_index]))


def generate_hashcode():
    short_hash_list = []
    train, train_label = get_data.get_zacao_input(train_dir, label_dir)
    train_batch, train_label_batch = get_data.get_batch2(train, train_label, 1, CAPACITY)

    image_contents = tf.read_file(train_batch[0])
    image_jpg = tf.image.decode_jpeg(image_contents, channels=3)
    image_jpg = tf.image.resize_images(image_jpg, [IMG_W, IMG_H], method=tf.image.ResizeMethod.AREA)
    image_jpg = tf.reshape(tensor=image_jpg, shape=[1, IMG_W, IMG_H, 3])

    logit, h_fc1_drop, hashcode = model.inference_3conv_without_dropout(image_jpg, 1, N_CLASSES)
    sess = tf.Session()

    saver = tf.train.Saver()
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
        return

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        step = 0
        while step < len(train):
            if coord.should_stop():
                break
            h_fc1_drop2, hashcode2, train_batch2, train_label_batch2 = sess.run(
                [h_fc1_drop, hashcode, train_batch, train_label_batch])
            print(h_fc1_drop2.shape)
            print(h_fc1_drop2)
            print(hashcode2.shape)
            print(hashcode2)
            print(train_label_batch2)
            print(train_batch2)
            img_name = train_batch2[0].decode()
            short_hash_str = ','.join(str(i) for i in hashcode2[0])
            long_hash_str = ','.join(str(round(float(i), 8)) for i in h_fc1_drop2[0])
            train_label_batch_str = str(train_label_batch2[0])
            if short_hash_str not in short_hash_list:
                short_hash_list.append(short_hash_str)
                save_to_mysql2('suoyin_3conv_128_48', short_hash_str)
            save_to_mysql('hash_3conv_128_48', long_hash_str, short_hash_str, img_name, train_label_batch_str)
            print('>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<')
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def save_to_mysql(table, long, short, dirt, label):
    sql = "INSERT INTO " + table + "(long_hash_code,short_hash_code,directory,label)" + " VALUES('" + long + "','" + short + "','" + pymysql.escape_string(
        dirt) + "','" + label + "')"
    try:
        cursor.execute(sql)
    except Exception as e:
        print(sql)
        print(e)


def save_to_mysql2(table, short):
    sql = "INSERT INTO " + table + "(short_hash_code)" + " VALUES('" + short + "')"
    try:
        cursor.execute(sql)
    except Exception as e:
        print(sql)
        print(e)


def find_all_short():
    sql = "SELECT short_hash_code FROM suoyin_4conv_256_128"
    result = None
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
    except Exception as e:
        print(sql)
        print(e)
    return result


def find_img_by_short(short_code):
    sql = "SELECT directory,long_hash_code FROM hash_4conv_256_128 WHERE short_hash_code = '" + short_code + "'"
    result = None
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
    except Exception as e:
        print(sql)
        print(e)
    return result


def find_img_by_short2(short_code):
    sql = "SELECT label,long_hash_code FROM hash_4conv_256_128 WHERE short_hash_code = '" + short_code + "'"
    result = None
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
    except Exception as e:
        print(sql)
        print(e)
    return result


def show_the_images(image_dir, images_dir):
    length = len(images_dir) if len(images_dir) < 6 else 5
    plt.figure(1)
    plt.subplot(100 + (length + 1) * 10 + 1)
    plt.axis("off")
    plt.imshow(Image.open(image_dir), cmap='gray')
    i = 0
    print('Top %d 图:' % length)
    while i < length:
        print('路径: %s 欧氏距离: %.6f' % (images_dir[i][0], images_dir[i][2]))
        img = Image.open(images_dir[i][0])
        plt.subplot(200 + (length + 1) * 10 + i + 2)
        plt.axis("off")
        plt.imshow(img, cmap='gray')
        i += 1
    plt.show()


def sort_by_long_hash(long_code, long_codes):
    i = 0
    while i < len(long_codes):
        dist = np.linalg.norm(long_code - long_codes[i][1])
        long_codes[i].append(dist)
        i += 1
    long_codes.sort(key=lambda x: x[2])
    return long_codes


def search_similar():
    image_dir = str(get_one_image_without_show())

    image_dir = 'F:\\zacao\\val\\cier\\test\\IMG_3955.jpg'
    image_contents = tf.read_file(image_dir)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [IMG_W, IMG_H], method=tf.image.ResizeMethod.AREA)
    image = tf.reshape(tensor=image, shape=[-1, IMG_W, IMG_H, 3])
    logit, h_fc1_drop, hashcode = model.inference_without_dropout(image, 1, N_CLASSES)
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
        long_code, short_code = sess.run([h_fc1_drop, hashcode])
    long_code = long_code.flatten()
    result = find_all_short()
    a = list(list(map(float, i[0].split(','))) for i in result)
    b = np.asarray(a)
    distance = []
    for i in b:
        x = np.vstack([i, short_code])
        distance.append(pdist(x, 'hamming')[0])
    dis_min = min(distance)
    dis_min_count = distance.count(dis_min)
    print('最小汉明距离: %s' % dis_min)
    print('最小距离频率: %s' % dis_min_count)
    print('查询图片Short_hash_code: %s' % ','.join(str(i) for i in short_code[0]))

    temp_distance = distance
    temp_max = 99999999
    images_and_long_codes = []

    while len(images_and_long_codes) < 5:
        temp_counter = 0
        dis_min = min(temp_distance)
        dis_min_count = temp_distance.count(dis_min)
        while temp_counter < dis_min_count:
            _ = temp_distance.index(dis_min)
            short_hash_code = ','.join(str(i) for i in b[_])
            print('相似图片Short_hash_code: %s' % short_hash_code)
            images_name_and_long_code = find_img_by_short(short_hash_code)
            for t in images_name_and_long_code:
                temp_d = t[0]
                temp_x = list(map(np.float32, t[1].split(',')))
                temp_x = np.asarray(temp_x).reshape((1, -1))
                temp_x = temp_x.flatten()
                images_and_long_codes.append([temp_d, temp_x])
            temp_distance[_] = temp_max
            temp_counter += 1
    print('查询图片路径: %s' % image_dir)
    for _ in images_and_long_codes:
        print('相似图片路径: %s' % _[0])
    sorted_img_list = sort_by_long_hash(long_code, images_and_long_codes)
    show_the_images(image_dir, sorted_img_list)


def search_similar_batch():
    batch_size = 50
    result = find_all_short()
    a = list(list(map(float, i[0].split(','))) for i in result)
    b = np.asarray(a)
    test, test_label = get_data.get_zacao_input(test_dir, test_label_dir)
    test_batch, test_label_batch = get_data.get_batch_jpg(test, test_label, IMG_W, IMG_H, batch_size, CAPACITY)
    logit, h_fc1_drop, hashcode = model.inference_without_dropout(test_batch, batch_size, N_CLASSES)
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
            return
        num_test = 1000
        num_iter = int(math.ceil(num_test / batch_size))
        step = 0
        coord2 = tf.train.Coordinator()
        threads2 = tf.train.start_queue_runners(sess=sess, coord=coord2)
        try:
            count_acc = 0
            while step < num_iter:
                temp_counter = 0
                correct_count = 0
                if step == 0:
                    print('Start test...')
                if coord2.should_stop():
                    break
                long_code, short_code, label = sess.run([h_fc1_drop, hashcode, test_label_batch])
                while temp_counter < batch_size:
                    distance = []
                    for i in b:
                        x = np.vstack([i, short_code[temp_counter]])
                        distance.append(pdist(x, 'hamming')[0])
                    inference_label = t_m(long_code[temp_counter], distance, b)
                    print(inference_label, label[temp_counter])
                    if inference_label == label[temp_counter]:
                        correct_count += 1
                    temp_counter += 1
                print("step: ", step)
                print("size: ", batch_size)
                print("correct_count: ", correct_count)
                count_acc += correct_count / batch_size
                step += 1
            print('Test acc = %.2f%%' % (count_acc / step * 100))
        except tf.errors.OutOfRangeError:
            print('Done testing -- epoch limit reached')
        finally:
            coord2.request_stop()
        coord2.join(threads2)


def t_m(long_code, distance, b):
    k = 15
    long_code = long_code.flatten()
    temp_max = 99999999
    images_and_long_codes = []
    while len(images_and_long_codes) < k:
        temp_counter = 0
        dis_min = min(distance)
        dis_min_count = distance.count(dis_min)
        while temp_counter < dis_min_count:
            _ = distance.index(dis_min)
            short_hash_code = ','.join(str(i) for i in b[_])
            images_label_and_long_code = find_img_by_short2(short_hash_code)
            for t in images_label_and_long_code:
                temp_l = t[0]
                temp_x = list(map(np.float32, t[1].split(',')))
                temp_x = np.asarray(temp_x).reshape((1, -1))
                temp_x = temp_x.flatten()
                images_and_long_codes.append([temp_l, temp_x])
                distance[_] = temp_max
            temp_counter += 1
    sorted_img_list = sort_by_long_hash(long_code, images_and_long_codes)
    temp_label = []
    for _ in range(0, k):
        temp_label.append(int(sorted_img_list[_][0]))
    print(temp_label)
    # return max(temp_label, key=temp_label.count)
    return count_min(temp_label)


def count_min(s):
    temp_unique, temp_count = np.unique(s, return_counts=True)
    # print("unique ", temp_unique)
    # print("count ", temp_count)
    temp_count_max = max(temp_count)
    temp_count_max_index = [i for i, j in enumerate(temp_count) if j == temp_count_max]
    # print("count_max index ", temp_count_max_index)
    temp_dict = {}
    for t_ in temp_count_max_index:
        temp_dict[str(temp_unique[t_])] = sum([i for i, j in enumerate(s) if j == temp_unique[t_]])
    # print(temp_dict)
    return int(min(temp_dict.items(), key=lambda x: x[1])[0])


if __name__ == '__main__':
    t1 = time.time()
    # w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = restore_pre_params()
    # run_training(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6 = restore_pre_4conv_params()
    run_training(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, True)
    # evaluate_one_image()
    # restore_pre_params2()
    # start_test()
    # generate_hashcode()
    # search_similar()
    # start_test_and_show_error()
    # search_similar_batch()
    conn.commit()
    cursor.close()
    conn.close()
    t2 = time.time()
    print("Total time: ", t2 - t1)
