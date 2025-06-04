import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/All"
splitRatio = {"train":0.7,"validation":0.2,"test":0.1}
classes = ["fake","real"]
 
try:
    shutil.rmtree(outputFolderPath)
    print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)


#Directory to create -----------------------------------
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)

os.makedirs(f"{outputFolderPath}/validation/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/validation/labels",exist_ok=True)

os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)


#get the names----------------------------------------------

listNames = os.listdir(inputFolderPath)

uniqueNames =[]
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))
    



#Shuffle-----------------------------------------------------
random.shuffle(uniqueNames)

#Find the unmber of images for each folder-------------------
lenData = len(uniqueNames)

lenTrain = int(lenData*splitRatio['train'])
lenValidation = int(lenData*splitRatio['validation'])
lenTest = int(lenData*splitRatio['test'])



#Put remaining images in Training---------------------------------------------

if lenData != lenValidation+lenTest+lenTrain:
    remaining = lenData - (lenValidation+lenTest+lenTrain)
    lenData += remaining

#split the list ---------------------------------------------
lenghtToSplit = [lenTrain, lenValidation, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input,elem)) for elem in lenghtToSplit]


#copy the files ---------------------------------------------
sequence = ['train','validation','test']

for i,out in enumerate(Output):
    for filename in out:
        shutil.copy(f"{inputFolderPath}/{filename}.jpg", f"{outputFolderPath}/{sequence[i]}/images/{filename}.jpg" )
        shutil.copy(f"{inputFolderPath}/{filename}.txt", f"{outputFolderPath}/{sequence[i]}/labels/{filename}.txt" )


#creating Data.yaml file--------------------------------------
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")
