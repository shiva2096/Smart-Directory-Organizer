import re;
import io;
from stemming.porter2 import stem;
import numpy as np;


def processFileName(x):
    LowerFileName = x.lower()
    
    # replacing  _ . ( ) [ ] , &  with <single space>
    regex1 = re.compile('[_.\[\]\(\),&$@%!~\+]')
    cleared = regex1.sub(' ',LowerFileName)

    # replacing - with <single space>
    regex2 = re.compile('[-]')
    cleared = regex2.sub(' ',cleared)

    # replacing multiple spaces with a <single space>
    regex3 = re.compile('[ ]+')
    cleared = regex3.sub(' ',cleared)

    # replacing alphanumerics with word 'alphanumeric'
    # for abc123
    regex4 = re.compile('[a-zA-Z]+[0-9]+[a-zA-Z]+')
    cleared = regex4.sub('alphanumeric',cleared)
    # for 123abc
    regex5 = re.compile('[0-9]+[a-zA-Z]+')
    cleared = regex5.sub('alphanumeric',cleared)
    #for abc123abc
    regex6 = re.compile('[a-zA-Z]+[0-9]+')
    cleared = regex6.sub('alphanumeric',cleared)

    # replacing numbers with word 'number'
    regex7 = re.compile('[0-9]+')
    cleared = regex7.sub('number',cleared)

    # stemming words using porter2 steammer
    tmp = ''
    token = cleared.split(' ')
    for t in token:
            tmp = tmp + ' ' + stem(t)
    stemmedName = tmp
    
    # removing leading white spaces
    regex8 = re.compile('^\s+')
    cleared = regex8.sub('',stemmedName)

    # removing trailing white spaces
    regex9 = re.compile('\s+$')
    processedFileName = regex9.sub('',cleared)

    return processedFileName;
    

def dataExtractor(input):
    Directory = [];
    FileName = [];
    FileExt = [];
    FileSize = [];
    with open(input,'r') as f:
        for lines in f:
            [a,b,c,d,e] = lines.split('\t')
            Directory.append(a);
            FileName.append(c);
            FileExt.append(d);
            FileSize.append(e);

    processedFileNames = [processFileName(x) for x in FileName]
    
    return processedFileNames,FileExt,FileSize,Directory;


