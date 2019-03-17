import glob
from shutil import copyfile
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
#participants = glob.glob("/Users/jahinmajumdar/Desktop/Emotion_Recognition/source_emotions//*") #Returns a list of all folders with participant numbers")
participants = glob.glob("source_emotion,*")

for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s//*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s//*" %sessions):
            current_session = sessions[-3:]
            #current_session = files
            file = open(files, 'r')

            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            print(len(glob.glob("source_images\\%s\\%s\\*" %(part, current_session))))
            sourcefile_emotion = glob.glob("source_images\\%s\\%s\\*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion

           # sourcefile_emotion = glob.glob("source_images//%s//%s//*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            #print sourcefile_emotion
            #print len(glob.glob("/Users/jahinmajumdar/Desktop/Emotion_Recognition/source_images//%s//%s//*" %(part, current_session)))
            #print len(glob.glob("source_images//%s//%s//*" %(part, current_session)))
            sourcefile_neutral = glob.glob("source_images//%s//%s//*" %(part, current_session))[0] #do same for neutral image
            #print sourcefile_neutral

            dest_neut = "sorted_setn//neutral//%s" %sourcefile_neutral[25:] #Generate path to put neutral image
            #print dest_neut
            dest_emot = "sorted_sete//%s//%s" %(emotions[emotion], sourcefile_emotion[25:]) #Do same for emotion containing image

            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file