import cv2
import glob
import random
import numpy as np
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from skimage import io,color
import matplotlib.pyplot as plt
from random import randint
A = np.zeros([6 * 123, 300 * 300])
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
fishface = cv2.face.createFisherFaceRecognizer() #Initialize fisher face classifier
#fishface = cv2.face.FisherFaceRecognizer_create()
lda = LinearDiscriminantAnalysis()
#Use cross validation to check performance
k_fold = model_selection.KFold(3, shuffle=True)
print(type(k_fold))
for (trn, tst) in k_fold.split(glob.glob("dataset\neutral\*")) :
    #Use PCA to transform from dimension F to dimension N-m
    print(trn)
    print(tst)

    '''pca = PCA(n_components=(len(trn) - numSbj))
    pca.fit(A[trn])
    #Compute LDA of reduced data
    lda.fit(pca.transform(A[trn]), y[trn])
    yHat = lda.predict(pca.transform(A[tst]))
    #Compute classification error
    outVal = accuracy_score(y[tst], yHat)
    print('Score: ' + str(outVal))
data = {}
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print ("training fisher face classifier")
    print ("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print ("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
#Now run it
metascore = []
for i in range(0,10):
    correct = run_recognizer()
    print ("got", correct, "percent correct!")
    metascore.append(correct)
print ("\n\nend score:", np.mean(metascore), "percent correct!")'''