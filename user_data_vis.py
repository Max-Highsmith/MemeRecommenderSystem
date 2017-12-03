import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
#data importing
df                     = pd.read_csv('memeDataRound1.csv');
userData               = df.as_matrix();
#data cleaning
numOfMemes             = 400;
numOfUsers             = 30;
dirtyMemeMatrix        = userData[0:numOfMemes, 1:numOfUsers];
memeMatrix             = dirtyMemeMatrix[dirtyMemeMatrix[:,0]>1];
numCM, numCU           = memeMatrix.shape;
memeApproval           = memeMatrix[:,0:2];
memePriors             = memeApproval[:,0]/memeApproval[:,1];
print(memePriors.sort());

#plot priors
fig = plt.figure();
ax = fig.add_subplot(111);
ax.set_title("Meme Priors");

ax.scatter(np.linspace(1, memePriors.size, memePriors.size), memePriors);
plt.show();


#collaborative filtering
#training and testing data split
#we split half the users as pure training data
#the other half we train on half the memes and test on the remaining memes
#Item-Item collaborative filtering
UserTrainers  = 30;
TrainData     = memeMatrix[:, 2:UserTrainers];
print(TrainData.shape);
TestData      = memeMatrix[:, UserTrainers:numOfUsers];


#########################
#########################
#May have to change right now treating un observed as voted down
#########################
#########################
itemSimil     = np.full((numCM, numCM),-9);

tsa, tsb= TrainData.shape;
for i in range(0,tsa):
    for j in range(0,tsb):
        if(np.isnan(TrainData[i,j])):
           TrainData[i,j]=0;

print(TrainData);
for i in range(0,numCM-2):
    for j in range(0, numCM-1):
        holder = cosine(TrainData[i,:], TrainData[j,:]);
        if((math.isnan(holder))==False):
            itemSimil[i,j] = 1-holder;  
        else:
            itemSimil[i,j] =-9;
            print(i);
            print(TrainData[i,1:]);
            print(j);
            print(TrainData[j,1:]);

print(TrainData[0,1:]);
print(1-cosine(TrainData[0,:],TrainData[0,:]));
print(TrainData[1,1:]);
print(1-cosine(TrainData[1,:],TrainData[1,:]));
print(TrainData[2,1:]);
print(1-cosine(TrainData[2,:],TrainData[2,:]));
print(TrainData[3,1:]);
print(1-cosine(TrainData[3,:],TrainData[3,:]));

print(TrainData[0:20,0:20]);
