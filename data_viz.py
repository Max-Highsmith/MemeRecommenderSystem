from math import log
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import webbrowser

df = pd.read_csv('MemeData.csv', sep=',', header=None)
memeData= df.as_matrix()
numOfObservations =200;
urlList = memeData[2:numOfObservations, 0];
nominalMemeData = memeData[2:numOfObservations, 2:22];
[numOfObs, numOfComponents] = nominalMemeData.shape

numOfRedComp =3;
pca = PCA(n_components =numOfRedComp)
reducedMemeData = pca.fit_transform(nominalMemeData);

#3d data vizualization
fig = plt.figure();
ax  = fig.add_subplot(111, projection='3d');
#ax   = fig.add_subplot(111);
ax.set_title('PCA reduced Meme data');

xs = reducedMemeData[:,0];
ys = reducedMemeData[:,1];
zs = reducedMemeData[:,2];
line = ax.scatter(xs, ys, zs, picker=5);
#line = ax.scatter(xs,ys, picker=5);
#line = ax.plot(np.random.rand(100), 'o', picker=5) 



ax.set_xlabel('X axis');
ax.set_ylabel('Y axis');
ax.set_zlabel('Z axis');

def onpick(event):
    print("something happened");
    thisline = event.artist;
  #  xdata    = thisline.get_xdata();
   # ydata    = thisline.get_ydata();
 #   zdata    = thisline.get_zdata();
    #points = tuple(zip(xdata[ind], ydata[ind]))
    ind      = event.ind
    print(ind);
    print(urlList[ind[0]]);
    webbrowser.open(urlList[ind[0]]);
   # print("you picked", ind);

fig.canvas.mpl_connect('pick_event', onpick);
 

plt.show();



#GMM
#initialization
numOfGMMs = 1;
weights = np.full([numOfGMMs],1/numOfGMMs);
mus     = np.empty([numOfRedComp, numOfGMMs]);
sigmas  = np.empty([numOfRedComp, numOfRedComp, numOfGMMs]);

chunkSize = int(numOfObservations/numOfGMMs);

for i in range(numOfGMMs):
    #mus[0:numOfRedComp,i]   = np.mean(reducedMemeData[(chunkSize*(i)):(chunkSize*(i+1)), 0:numOfRedComp], axis=0);
    sigmas[:,:,i] = np.identity(numOfRedComp);

mus[:,0] = np.array([1,0,0]);
#mus[:,1] = np.array([-1,0,0]); 
#mus[:,2] = np.array([0,1,0]);
#mus[:,3] = np.array([0,-1,0]);
#mus[:,4] = np.array([0,0,1]);

print(sigmas);
def expectation(weights, sigmas, mus):
    responsibility = np.full([numOfObservations, numOfGMMs],-1,dtype=np.float);
    lowSum = np.full([numOfObservations], 0, dtype=np.float);
    print("oj");
    for n in range(numOfObservations-2):
        for j in range(numOfGMMs):
            print("normie");
            print(n);
            print("iter");
            print(i);
            print("mu");
            print(mus[:,j]);
            print("sig");
            print(sigmas[:,:,j]);
            normie = st.multivariate_normal(mus[:,j],sigmas[:,:,j]);
            print("kk");
            xx = normie.pdf(reducedMemeData[n,:]);
            print("shut");
            responsibility[n,j] = xx;
            print("oh");
            lowSum[n] = lowSum[n] + responsibility[n,j];
    for n in range(numOfObservations-2):
        for j in range(numOfGMMs):
            responsibility[n,j] = responsibility[n,j]/ lowSum[n];
    return responsibility;

def maximization(responsibility):
    unNormWeights = np.full([numOfGMMs], 0, np.float);
    for n in range(numOfObservations-2):
        for k in range(numOfGMMs):
            unNormWeights[k] = unNormWeights[k] + responsibility[n,k];
            mus[:,k] = mus[:,k] + responsibility[n,k]*reducedMemeData[n,:];
    sumWeights = np.sum(weights[k]);
    for k in range(numOfGMMs):
        weights[k] = unNormWeights[k]/sumWeights;
        for n in range(numOfObservations-2):
            mus[:,k]       = (1/unNormWeights[k])*mus[:,k];
            difFromMean    = reducedMemeData[n,:]-mus[:,k];
            mainVec = np.expand_dims(difFromMean,axis=0);
            tranVec = np.transpose(mainVec);
            print("big meanie");
            print(difFromMean);
            print("prod");
            print(np.matmul(tranVec, mainVec));
            sigmas[:,:,k]  = (1/unNormWeights[k])*responsibility[n,k]*np.matmul(tranVec, mainVec);                       
    return weights, mus, sigmas;    

respp = expectation(weights, sigmas, mus);            
print("Responsibility");
print(respp);
[weights, mus, sigmas] = maximization(respp);
print("weights");
print(weights);
print("mus");
print(mus);
print("sigmas");
print(sigmas);
def logLikelihood(weights, mus, sigmas):
    outerSum =0;
    for n in range(numOfObservations-2):
        innerSum = 0;
        for k in range(numOfGMMs):
            normie   = st.multivariate_normal(mus[:,k], sigmas[:,:,k]);
            xx       = normie.pdf(reducedMemeData[n,:]);
            innerSum = innerSum + weights[k]*xx;
        outerSum = outerSum + math.log(innerSum);
    return outerSum;         
#print(mus);
#print(sigmas);
#print("sigma");
#print(sigmas[:,:,1]);
for i in range(10):
   respp = expectation(weights, sigmas, mus);
   print("resp");
   print(respp);
   [weights, mus, sigmas] = maximization(respp);
   print("this far");
   print(logLikelihood(weights, mus, sigmas));
   print("nah");



