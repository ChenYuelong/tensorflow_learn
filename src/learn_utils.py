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
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_test_data():
    '''
    生成随机数据
    :return:
    '''
    ylist = []
    xlist= []

    for i in range(100):
        yt = [0,0,0]
        yt[i%3]=1
        ylist.append(yt)
        xlist.append((i%3)*np.random.normal(1,3,size = (460,460,1)))
    return np.array(xlist),np.array(ylist)



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
def max_pool_7x7(x):
    return tf.nn.max_pool(x,ksize=[1,7,7,1],
                         strides=[1,7,7,1],padding='SAME')

def relu(x):
    return tf.nn.relu(x)

def softmax(x):
    return tf.nn.softmax(x)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def lv1_cnn(input_x,input_y):

    t_x = tf.placeholder(tf.float32,shape=[None,460,460,1])
    t_y = tf.placeholder(tf.float32,shape=[None,3])
    keep_prob = tf.placeholder(tf.float32)

    #第一个卷积池为3x3的卷积池，输出为7个
    W_1 = weight_variable([3,3,1,7])
    #经过卷积过程
    conv1 = conv2d(t_x,W_1)
    #池化过程
    pool1 = max_pool_7x7(conv1)
    #relu过程
    relu1 = relu(pool1)

    #第二个卷积池，3x3，输出为一个图
    W_2 = weight_variable([3,3,7,1])
    conv2 = conv2d(relu1,W_2)

    #全连接层fc1
    w_fc1 = weight_variable([66*66,1024])
    b_fc1 = bias_variable([1024])
    conv2_flat = tf.reshape(conv2, [-1, 66 * 66 * 1])
    h_fc1 = tf.nn.relu(tf.matmul(conv2_flat, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #全连接层fc2
    W_fc2 = weight_variable([1024, 3])
    b_fc2 = bias_variable([3])
    # 直接经过softmax进行分类
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    cross_entropy = tf.reduce_mean(-tf.reduce_sum(t_y * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    ## 区分训练集及测试集
    x_train, x_test, y_train, y_test = train_test_split(input_x, input_y,
                                                        test_size=0.2,
                                                        random_state=1)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            sess.run(train_step, feed_dict={t_x: x_train, t_y: y_train, keep_prob: 0.5})
            if i % 50 == 0:
                print(compute_accuracy(
                    x_test, y_test))



def main():
    '''
    测试流程
    '''
    data= generate_test_data()
    lv1_cnn(data['features'],data['classification'])



if __name__ == '__main__':
    main()