import os
import tensorflow as tf;
from featureVectorCreation import get_x;
from dataSetManipulator import processFileName;

tf.reset_default_graph()


# Generating Class Map  (Class No.  ->  Directory Name)
classMap = {};
words = [];
key = [];
with open('.\\classes.txt') as f:
    for lines in f:
        [a,b] = lines.split(' ');
        words.append(a);
        key.append(b);    
for i in range(0,len(words)):
    classMap[int(key[i])] = words[i];


# Taking Input
inputFilepath = input('Enter file name with Directory and extension (D:/Music/myfile.mp3)')

# Generating Feature Vector
filename_w_ext = os.path.basename(inputFilepath)
filename, file_extension = os.path.splitext(filename_w_ext)
fileSize = str(os.stat(inputFilepath).st_size)
processedFileName = processFileName(filename)
fVec = get_x(processedFileName,file_extension,sileSize)

# No. of Features
noOfFeatures = len(fVec)

# No. of Classes
noOfClasses = len(classMap)

# Finding the predicted Class(Directory)

# 1) Restoring the Trained Model
W1 = tf.get_variable("W1", shape=[noOfFeatures,100])
b1 = tf.get_variable("b1",shape=[100])
W2 = tf.get_variable("W2", shape=[100,noOfClasses])
b2 = tf.get_variable("b2",shape=[noOfClasses])
mean = tf.get_variable("mean", shape=[noOfFeatures])
var = tf.get_variable("var",shape=[noOfFeatures])

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./TrainedModel/model.ckpt")

# 2) Feed forward the feature vector through the trained Model
x = tf.placeholder(tf.float32,[None,noOfFeatures])
x_norm = tf.nn.batch_normalization(x,mean,var,None,None,1e-12,None)
a1 = tf.sigmoid(tf.matmul(x_norm,W1)+b1)
y = tf.sigmoid(tf.matmul(a1,W2)+b2)

output = sess.run(y,{x:fVec})

predictedClass = tf.argmax(output, 1) + 1;

print('The predicted Directory for Input File is : %s' % (classMap[predictedClass]))


