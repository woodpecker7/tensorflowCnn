#-*- coding:utf-8  -*-

import string, os, sys
import numpy as np
import pandas as pd
import scipy.io
import random
import tensorflow as tf

dir_name = '/data/fer2013'
print '----------- no sub dir'
print ('The folder path: ', dir_name)

files = os.listdir(dir_name)
for f in files:
    print (dir_name + os.sep + f)

file_path = '/data/fer2013/fer2013.csv'

print file_path

data = pd.read_csv(file_path, dtype='a')

label = np.array(data['emotion'])
img_data = np.array(data['pixels'])

N_sample = label.size
data_mat = np.zeros((N_sample, 48*48))
label_mat = np.zeros((N_sample, 7), dtype=int)

for i in range(N_sample):
    x = img_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')
   # print "len(x)",len(x)
    x_max = x.max()
    x = x/(x_max+0.0001)
#    print x_max
#    print x
    
    
    #Face_data[i] = x
    #Face_label[i, label[i]] = 1
        
    data_mat[i] = x
    label_mat[i, label[i]] = 1
    
train_num = 30000
test_num = 5000

train_x = data_mat [0:train_num, :]
train_y = label_mat [0:train_num, :]

test_x = data_mat [train_num : train_num+test_num, :]
test_y = label_mat [train_num : train_num+test_num, :]

print ("All is well")

batch_size = 50
train_batch_num = train_num/batch_size
test_batch_num = test_num/batch_size
train_epoch = 100


learning_rate = 0.001
# Network Parameters
n_input = 2304  # data input (img shape: 48*48)
n_classes = 7   # total classes
dropout = 0.5   # Dropout, probability to keep units

# tf Graph input

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 48, 48, 1])

    
    ## 3x3 conv, 1 input, 128 outputs
    ##'wc1': tf.Variable(tf.random_normal([3, 3, 1, 128]))
    ##'bc1': tf.Variable(tf.random_normal([128]))
     
    
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print "conv1.shape",conv1.get_shape()
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print "conv1.shape",conv1.get_shape()

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print "conv2.shape",conv2.get_shape()
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print "conv2.shape",conv2.get_shape()

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    print "conv3.shape",conv3.get_shape()
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    print "conv3.shape",conv3.get_shape()

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print "fc1.shape",fc1.get_shape()
    fc1 = tf.nn.relu(fc1)
    print "fc1.shape",fc1.get_shape()

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print "out.shape",out.get_shape()

    return out

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 128 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 128]),name="wc1"),
    # 3x3 conv, 128 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 128, 64]),name="wc2"),
    # 3x3 conv, 64 inputs, 32 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32]),name="wc3"),
    # fully connected,
    'wd1': tf.Variable(tf.random_normal([6*6*32, 200]),name="wd1"),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([200, n_classes]),name="outw")
}


biases = {
    'bc1': tf.Variable(tf.random_normal([128]),name="bc1"),

    'bc2': tf.Variable(tf.random_normal([64]),name="bc2"),

    'bc3': tf.Variable(tf.random_normal([32]),name="bc3"),

    'bd1': tf.Variable(tf.random_normal([200]),name="bd1"),

    'out': tf.Variable(tf.random_normal([n_classes]),name="outb")
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

Train_ind = np.arange(train_num)
Test_ind = np.arange(test_num)

with tf.Session() as sess:
    sess.run(init)
    
    #train_num = 30000
    #test_num = 5000
    #batch_size = 50
    #train_batch_num = train_num/batch_size=600
    #test_batch_num = test_num/batch_size=100
    #train_epoch = 100

    for epoch in range(0, train_epoch):

        Total_test_loss = 0
        Total_test_acc = 0

        for train_batch in range (0, train_batch_num):
            sample_ind = Train_ind[train_batch * batch_size:(train_batch + 1) * batch_size]
            batch_x = train_x[sample_ind, :]
            batch_y = train_y[sample_ind, :]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})

            if train_batch % batch_size == 0:
                # Calculate loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})

                print("Epoch: " + str(epoch+1) + ", Batch: "+ str(train_batch) + ", Loss= " + \
                            "{:.3f}".format(loss) + ", Training Accuracy= " + \
                            "{:.3f}".format(acc))

        # Calculate test loss and test accuracy
        for test_batch in range (0, test_batch_num):
            sample_ind = Test_ind[test_batch * batch_size:(test_batch + 1) * batch_size]
            batch_x = test_x[sample_ind, :]
            batch_y = test_y[sample_ind, :]
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        keep_prob: 1.})
            Total_test_lost = Total_test_loss + test_loss
            Total_test_acc =Total_test_acc + test_acc



        Total_test_acc = Total_test_acc/test_batch_num
        Total_test_loss =Total_test_lost/test_batch_num

        print("Epoch: " + str(epoch + 1) + ", Test Loss= " + \
                      "{:.3f}".format(Total_test_loss) + ", Test Accuracy= " + \
                      "{:.3f}".format(Total_test_acc))
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/data/model.ckpt")
    print "Model saved in file: ", save_path
 

print "All is well"


        


