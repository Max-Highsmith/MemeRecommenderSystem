import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
#data importing
df                     = pd.read_csv('MemeSurveyData.csv');
userData               = df.as_matrix();
#data cleaning
numOfMemes             = 800;
numOfUsers             = 44;
dirtyMemeMatrix        = userData[0:numOfMemes, 1:numOfUsers];
memeURL                = userData[0:numOfMemes, 0];
dMIndex                = np.arange(numOfMemes);
cMIndex                = dMIndex[dirtyMemeMatrix[:,0]>1];
memeMatrix             = dirtyMemeMatrix[dirtyMemeMatrix[:,0]>1];
numCM, numCU           = memeMatrix.shape;
memeApproval           = memeMatrix[:,0:2];
memePriors             = memeApproval[:,0]/memeApproval[:,1];


#plot priors
fig = plt.figure();
ax = fig.add_subplot(111);
ax.set_title("Meme Priors");

ax.scatter(np.linspace(1, memePriors.size, memePriors.size), memePriors);
#plt.show();


#collaborative filtering
#training and testing data split
#we split half the users as pure training data
#the other half we train on half the memes and test on the remaining memes
#Item-Item collaborative filtering
UserTrainers  = 35;
TrainData     = np.copy(memeMatrix[:, 2:UserTrainers]);
TestData      = np.copy(memeMatrix[:, UserTrainers:numOfUsers]);
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


#account for nan
for i in range(0,tsa):
    for j in range(0,numOfUsers- UserTrainers-1):
        if(np.isnan(TestData[i,j])):
           TestData[i,j]=0.5;

similMat = np.full((tsa,tsa),-9.1);
for i in range(0,tsa):
    for j in range(0,tsa):
        similMat[i,j] = cosine(TrainData[i,0:UserTrainers], TrainData[j,0:UserTrainers]);
        if(np.isnan(similMat[i,j])):
            similMat[i,j] = 0;
        similMat[i,j] = 1 - similMat[i,j];

mostSimItems    = np.copy(similMat);
mostSimItemVal  = np.copy(similMat);
for i in range(0, tsa):
    mostSimItems[i]   =  np.argsort(similMat[i,:]);
    mostSimItemVal[i] =  np.sort(similMat[i,:]);

mostSimItems   = np.flip(mostSimItems,1);
mostSimItemVal = np.flip(mostSimItemVal,1);




memesSeen= 0;
memesWithCorrectPred =0;
memesCorrectExist = 0;
for tester in range(0,numOfUsers-UserTrainers-1):
    recomendedToUser = np.array([-9]);
    for meme in range(0, tsa):
        if(TestData[meme,tester]==1): #tester liked base meme
            memesCorrectExist = memesCorrectExist +1;
            for simInd in range(meme+1,tsa):
                SeenAlready = (np.any(recomendedToUser==mostSimItems[meme,simInd]));
                if(mostSimItemVal[meme,simInd] > .5 and SeenAlready==False):  #similar enough
                    recomendedToUser = np.append(recomendedToUser,mostSimItems[meme,simInd] );
                    if(TestData[int(mostSimItems[meme,simInd]),tester] ==1 or TestData[int(mostSimItems[meme,simInd]),tester] == 0):#if tester
#				#observed similar meme
                        memesSeen = memesSeen+1;
                        if(TestData[int(mostSimItems[meme,simInd]),tester]==1):
                            memesWithCorrectPred = memesWithCorrectPred+1; 
         

print("precision");
precision = memesWithCorrectPred/memesSeen;
print(precision);
print("recall");
recall = memesWithCorrectPred/memesCorrectExist;
print(recall);
print("f1");
f1 = 2* (precision*recall)/(precision+recall);
print(f1);




#popularity Recommendation
totalTrainingLikes = np.sum(TrainData,axis=1);
TrainData     = memeMatrix[:, 2:UserTrainers];
totalObs = np.copy(totalTrainingLikes);
for meme in range(0, totalObs.shape[0]):
    total = 0;
    for user in range(2,UserTrainers):
        if (memeMatrix[meme,user]==1 or memeMatrix[meme,user]==0):
            total = total+1;
    totalObs[meme] = total;
#print(TrainData);
print(totalTrainingLikes);
print(totalObs);

rMPRob =  np.copy(totalObs);
for i in range(totalObs.size):
    if(totalObs[i]!=0):
        rMPRob[i] = totalTrainingLikes[i]/totalObs[i];
    else:
        rMPRob[i] = 0;

#test

corRet =0;
retrieved = 0;
threshold=0.5;
for tester in range(0,numOfUsers-UserTrainers-1):
    for meme in range(0, tsa):
        if(TestData[meme,tester] ==1 or TestData[meme,tester] == 0):
             if(rMPRob[meme] >threshold):
                 retrieved = retrieved+1;
                 if(TestData[meme, tester]==1):
                     corRet = corRet +1;
print("correctRetrieved");
print(corRet);
print("total Retreived");
print(retrieved);
print("total in existence");
print(memesCorrectExist);
pres = corRet/ retrieved;
rec  = corRet/memesCorrectExist;
print("precision");
print(pres);
print("recall");
print(rec);
print("f1");
f1 = 2* (pres*rec)/(pres+rec);
print(f1);
