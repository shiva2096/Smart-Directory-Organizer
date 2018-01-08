from dataSetManipulator import dataExtractor
import numpy as np

# getting vocab from vocab.txt
vocab = {};
words = [];
key = [];
with open('.\\vocab.txt') as f:
    for lines in f:
        [a,b] = lines.split(' ');
        words.append(a);
        key.append(b);
    
for i in range(0,len(words)):
    vocab[words[i]] = int(key[i]);
    
# getting extension vocab from extVocab.txt
extVocab = {};
words = [];
key = [];
with open('.\\extVocab.txt') as f:
    for lines in f:
        [a,b] = lines.split(' ');
        words.append(a);
        key.append(b);    
for i in range(0,len(words)):
    extVocab[words[i]] = int(key[i]);

# getting classes and their no. from classes.txt
classMap = {};
words = [];
key = [];
with open('.\\classes.txt') as f:
    for lines in f:
        [a,b] = lines.split(' ');
        words.append(a);
        key.append(b);
# Mapping (Directory Name  ->  Class no)
for i in range(0,len(words)):
    classMap[words[i]] = int(key[i]);

# size of feature vector
fvSize = len(vocab) + len(extVocab) + 1;

# no of classes
noOfClass = len(classMap);

def get_x(FileName,FileExtension,FileSize):
    list = [0]*fvSize;
    str = FileName.split(' ');
    for x in str:
        if x in vocab:
            list[vocab[x]-1] = 1;
    p = len(vocab) + extVocab[FileExt] - 1;
    list[p] = 1;
    p = len(vocab) + len(extVocab);
    list[p] = int(FileSize);
    return list;

def get_y(Directory):
    classList = [0]*noOfClass;
    classList[classMap[Directory]-1] = 1;
    return classList;


def get_x_y(input):
    # creating feature vector
    [FileName,FileExt,FileSize,Directory] = dataExtractor(input);
       
    trainingSet = [];
    trainingSetClass = [];
    
    for i in range(0,len(FileName)):
        l1 = get_x(FileName[i],FileExt[i],FileSize[i])
        l2 = get_y(Directory[i])
        trainingSet.append(l1)
        trainingSetClass.append(l2)
    
    return trainingSet,trainingSetClass;   
