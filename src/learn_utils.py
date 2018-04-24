#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @PROJECT : jupyter_Projects
# @Time    : 2018/4/17 10:09
# @Author  : Chen Yuelong
# @Mail    : yuelong.chen@oumeng.com.cn
# @File    : learn_utils.py
# @Software: PyCharm

from __future__ import absolute_import, unicode_literals
import sys, os
import tensorflow as tf
import time
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from sklearn.model_selection import train_test_split
import math

def generate_test_data(n,std):
    '''
    生成随机数据
    :param: n 表示生成的矩阵大小
    :param: 表示随机数的标准差
    :return:
    '''
    ylist = []
    xlist= []

    for i in range(1000):
        yt = [0,0,0]
        yt[i%3]=1
        ylist.append(yt)
        xlist.append(np.random.normal(i%3,std,size = (n,n)))
    return np.array(xlist),np.array(ylist)

def cnn_test(n,std):
    '''

    :param n: 生成矩阵的大小
    :param std: 随机数的标准差
    :return: 准确性
    '''
    start = time.time()
    x_dat, y_dat = generate_test_data(n, std)
    x = tf.placeholder(tf.float32, [None, n, n])
    y_ = tf.placeholder(tf.float32, [None, 3])

    cnn_conv1_out = 7
    cnn_conv2_out = 7

    cnn_x = tf.reshape(x, [-1, n , n,1])

    W_1 = tf.Variable(tf.truncated_normal(shape=[3 , 3, 1,cnn_conv1_out], stddev=0.01))
    b_1 = tf.Variable(tf.truncated_normal(shape=[cnn_conv1_out], stddev=0.01))

    W_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, cnn_conv1_out, cnn_conv2_out], stddev=0.01))
    b_2 = tf.Variable(tf.truncated_normal(shape=[cnn_conv2_out], stddev=0.01))

    fc_1 = tf.Variable(tf.truncated_normal(shape=[cnn_conv2_out*math.ceil(n/4)*math.ceil(n/4),1024], stddev=0.01))
    fc_b1 = tf.Variable(tf.truncated_normal(shape=[1024], stddev=0.01))
    fc_2 = tf.Variable(tf.truncated_normal(shape=[1024,3], stddev=0.01))
    fc_b2 = tf.Variable(tf.truncated_normal(shape=[3], stddev=0.01))

    h_1 = relu(max_pool_2x2(conv2d(cnn_x,W_1)+b_1))
    h_2 = tf.reshape(relu(max_pool_2x2(conv2d(h_1,W_2)+b_2)),[-1,cnn_conv2_out*math.ceil(n/4)*math.ceil(n/4)])


    y = tf.nn.softmax(tf.matmul((tf.matmul(h_2,fc_1)+fc_b1),fc_2)+fc_b2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    ## 准确性计算
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    x_train, x_test, y_train, y_test = train_test_split(x_dat, y_dat, test_size=0.1)
    # print('矩阵大小为：{0}x{0}，标准差为：{1}'.format(n,std))
    result = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            sess.run(train_step, feed_dict={x: x_train, y_: y_train})
            if i % 100 == 0:
                result.append(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
                # print('迭代次数：%d,测试准确性：%.2f，训练准确性：%.2f'
                #       % (i, sess.run(accuracy, feed_dict={x: x_test, y_: y_test}),
                #                                          sess.run(accuracy, feed_dict={x: x_train, y_: y_train})))
    end = time.time()
    mytime  =(end-start)
    print('Use Time: %.2f s and final accuracy is %.2f' % (mytime,result[-1]*100))

    return result

def linear_test(n,std):
    '''

    :param n: 生成矩阵的大小
    :param std: 随机数的标准差
    :return: 准确性
    '''
    x_dat, y_dat = generate_test_data(n,std)
    x = tf.placeholder(tf.float32, [None, n, n])
    y_ = tf.placeholder(tf.float32, [None, 3])

    linear_x = tf.reshape(x, [-1, n * n])

    W = tf.Variable(tf.truncated_normal(shape=[n * n, 3], stddev=0.01))
    b = tf.Variable(tf.truncated_normal(shape=[3], stddev=0.01))

    y = tf.nn.softmax(tf.matmul(linear_x, W) + b)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    ## 准确性计算
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    x_train, x_test, y_train, y_test = train_test_split(x_dat, y_dat, test_size=0.1)
    # print('矩阵大小为：{0}x{0}，标准差为：{1}'.format(n,std))
    result = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            sess.run(train_step, feed_dict={x: x_train, y_: y_train})
            if i % 100 == 0:
                result.append(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
                # print('迭代次数：%d,测试准确性：%.2f，训练准确性：%.2f'
                #       % (i, sess.run(accuracy, feed_dict={x: x_test, y_: y_test}),
                #                                          sess.run(accuracy, feed_dict={x: x_train, y_: y_train})))
    return result

# 权重值的初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 偏好值初始化
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积过程
def conv2d(x,W):
    return tf.nn.conv2d(input=x,filter=W,strides=[1,1,1,1],padding='SAME')

# 池化过程
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                         strides=[1,2,2,1],padding='SAME')

def relu(x):
    return tf.nn.relu(x)

def softmax(x):
    return tf.nn.softmax(x)






def main():
    '''
    测试流程
    '''




if __name__ == '__main__':
    main()