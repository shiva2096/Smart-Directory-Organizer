import featureVectorCreation
from featureVectorCreation import get_x_y
import tensorflow as tf
import datetime
import numpy as np

# Value of Lambda for Regularization
lamda = 0.01;

# Getting featureVector and Labels for data set
[featureVec,labels] = get_x_y('.\\directoryStructure.txt');

#[featureVecTest,labelsTest] = get_x_y('.\\testSet.txt');

# No of Features
noOfFeatures = len(featureVec[0]);

# No of Classes
noOfClasses = len(labels[0]);


# Calculating Mean and Variance for feature Scalling
trainSetVec = tf.placeholder(tf.float32,[None,noOfFeatures])
m, v = tf.nn.moments(trainSetVec, axes=[0]) # 0 axes represents normalize along columns
sess = tf.Session()
meanVec = sess.run(m,{trainSetVec:featureVec})
varVec = sess.run(v,{trainSetVec:featureVec})

# Tensor Holding the Mean and Variance, will be used while saving the model
mean = tf.Variable(meanVec,dtype=tf.float32)
var = tf.Variable(varVec,dtype=tf.float32)

# Generating weights and Biases to be used
W1 = tf.Variable( tf.truncated_normal([noOfFeatures,100], stddev=0.1))
b1 = tf.Variable( tf.constant(0.1, shape=[100]))
W2 = tf.Variable( tf.truncated_normal([100,noOfClasses], stddev=0.1))
b2 = tf.Variable( tf.constant(0.1, shape=[noOfClasses]))

# 3 Layer Neural Network
x = tf.placeholder(tf.float32,[None,noOfFeatures])
x_norm = tf.nn.batch_normalization(x,mean,var,None,None,1e-12,None)
a1 = tf.sigmoid(tf.matmul(x_norm,W1)+b1)
y = tf.sigmoid(tf.matmul(a1,W2)+b2)

# Tensor for holding actual Labels
y_ = tf.placeholder(tf.float32, [None, noOfClasses])

# Regularization Term
reg = lamda*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1)+ tf.nn.l2_loss(b2)) 

# Cost for Training
cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)) + reg

# Training Optimiser (Adam)
train = tf.train.AdamOptimizer(3e-5).minimize(cost)

# Calculating Accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# Start Training
a = str(datetime.datetime.now());
print('training starts at %s'%a)

sess.run(tf.global_variables_initializer())
for i in range(2000):
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy,{x:featureVec,y_:labels})
        train_cost = sess.run(cost,{x:featureVec,y_:labels})
    print('step %d, training accuracy %g,cost %g' % (i, train_accuracy,train_cost))
    sess.run(train,{x: featureVec, y_:labels})


a = str(datetime.datetime.now());
print('training ends at %s'%a)

#test_accuracy = sess.run(accuracy,{x:featureVecTest,y_:labelsTest})
#print('Test Accuracy after 2000 steps : %g' % (test_accuracy))

# Saving the Trained Model
saver = tf.train.Saver({"mean":mean,"var":var,"W1":W1,"W2":W2,"b1":b1,"b2":b2})
save_path = saver.save(sess, "./TrainedModel/model.ckpt")
print('Model saved in ./TrainedModel/model.ckpt ')
