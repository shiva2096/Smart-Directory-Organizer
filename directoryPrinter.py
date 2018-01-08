import os

searchDir = input("Input the directory or Drive for which you want Smart Paste Work (like  D: or D:\Music):")

for dirpath, dirnames, files in os.walk(searchDir):
    for name in files:
        filename, file_extension = os.path.splitext(name)
        filePath = os.path.join(dirpath,name)
        fileSize = str(os.stat(filePath).st_size)
        with open('.\\directoryStructure.txt', 'a') as f:
            print(str(dirpath)+'\t'+str(name)+'\t'+str(filename)+'\t'+str(file_extension)+'\t'+fileSize,file=f)

