# Smart-Directory-Organizer
This project organizes your files in Directories in Personalized Order using Machine Learning.

## Features
- First this program learns how you organize files on your Desktop (This makes it Personalized)
- Then it can predicts the Directory for the new files with **75 % Accuracy** 

## Prerequisites
- Tensorflow
- Numpy

## How to run ?
- To Train the Model Run --> **runMeForTraining.py**
- To Predict the directory Run --> **runMeForClassification.py**

## Working

### 1) Dataset Creation
Collecting the data such that it contains:
- Directory Name
- File Name
- File Extension 
- File Size

### 2) Feature Creation
I used 3 features to predict the correct directory of a File; its Name,Extension and Size
- I processed the File Names and remove special symbols,extra white space,and other unwanted things. 
(Precisely described in **dataSetManipulator.py**)
- Then I used **porter2 steamer** for Steeming the file names. ( Google Steeming Keyword if you don't know about it )
- Then from these File Names, I created a vocabulary containing all the unique words in the File Name after processsing it.
- Each word in the dictonary was a feature. (Thus a word is present in file name then that feature value is set 1 otherwise 0)
- Similarly a Vocabulary of all unique Extensions was also created.

**Thus Feature Vector will be of size :  no of words in Vocab + no of words in extension Vocab + 1 (For file size)**

### 3) Training
- Used Tensorflow to create a 3 Layered Neural Network
- Trained the model for **2000 Iterations to achieve 81% training accuracy**
- Training took about **30 Minutes**

![alt text](https://github.com/shiva2096/Smart-Directory-Organizer/blob/master/TrainingImage.JPG "Training Image")

### 4) Prediction
- Give the full directory path for your file to **runMeForClassification.py**
- It will Output the Predicted Directory Name
