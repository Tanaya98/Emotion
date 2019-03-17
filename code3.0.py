import cv2
import glob
import numpy as np
data=glob.glob('datasets/neutral/*')
#print(data)
c=data[0]


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

d=0
for i in data:
    print(i)

    img1 = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

    img2 = cv2.imread(c, cv2.IMREAD_GRAYSCALE)

    print(mse(img1,img2))
    if(mse(img1,img2)>1000):
        cv2.imwrite('C:\\Users\DELL_PC\PycharmProjects\IPProject\datasets\new\\'+i[-6:] ,img1)
        d+=1
        print(d)
    c=i
    #if (cv2.imread(data[i])==cv2.imread(data[i+1])):
    #    print(str(i)+' SAME')