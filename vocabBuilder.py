import numpy as np;
from dataSetManipulator import dataExtractor;

processedFileNames,FileExt,FileSize,Directory = dataExtractor('.\\directoryStructure.txt');

unfilteredLib = []
for x in processedFileNames:
    token = x.split(' ')
    for y in token:
        unfilteredLib.append(y)

vocab = np.unique(unfilteredLib);

extVocab = np.unique(FileExt);

classes = np.unique(Directory);

# Creating file for Vocab

i = 1;

for x in vocab:
    with open('.\\vocab.txt','a') as f:
        print(x+' '+str(i),file=f);
        i = i+1;

# Creating file for Vocab of file Extensions

i = 1;

for x in extVocab:
    with open('.\\extVocab.txt','a') as f:
        print(x+' '+str(i),file=f);
        i = i+1;

# Creating file for Vocab for Classes (Directory)

i = 1;

for x in classes:
    with open('.\\classes.txt','a') as f:
        print(x+' '+str(i),file=f);
        i = i+1;

