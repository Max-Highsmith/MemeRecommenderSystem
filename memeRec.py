# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 01:45:10 2017

@author: Jon-Michael
"""

import numpy as np
from openpyxl import load_workbook
import random as rnd

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def pickFirstMeme(options):
    reduce = options[0]
    for i in range(0, len(options)):
        if options[i][1] != None:
            if options[i][1] > reduce[1]:
                reduce = options[i]
    return reduce

def nextRecommend(users, index, allMemes):
    recommend = [0, 0]
    for i in range(1, len(users[0])):
        hits = 0
        unpicked = True
        for j in range(0, len(index)):
            if i == index[j]:
                unpicked = False
        for j in range(0, len(users)):
            if users[j][i] == True:
                hits += 1
        if hits > recommend[1] and unpicked:
            recommend = [i, hits]
    giveback = allMemes[recommend[0]]
    return giveback

def getImage(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
    
def getMemeNum(url, basememes):
    for i in range(0, len(basememes)):
        if basememes[i][0] == url:
            return i
    return -1
    
def userLikes(users, index):
    incrowd = []
    for i in range(0, len(users)):
        if users[i][index] == True:
            incrowd.append(users[i])
    return incrowd

def userHates(users, index):
    incrowd = []
    for i in range(0, len(users)):
        if users[i][index] != True:
            incrowd.append(users[i])
    return incrowd

workbook = load_workbook("MemeSurveyData.xlsx", data_only=True)
worksheet = workbook.worksheets[0]

row_count = worksheet.max_row

row = 0

data = []

for row in worksheet.iter_rows():
    holder = []
    for cell in row:
        holder.append(cell.internal_value)
    data.append(holder)

memeBaseData = []

for i in range(1, len(data)):
    memeBaseData.append([data[i][0], data[i][1], data[i][2]])

buildUsers = []

for i in range(3, len(data[0])):
    buildUsers.append([data[0][i]]);

for i in range(3, len(data[0])):
    for j in range(1, len(data)):
        buildUsers[i-3].append(data[j][i])
        
meme = pickFirstMeme(memeBaseData)

url = meme[0]
img = getImage(url)

imgplot = plt.imshow(img)
plt.show()

picks = [getMemeNum(url, memeBaseData)]

like = input("Type yes if you liked, no otherwise:")

for i in range(0, 4):
    if like == "yes" or like == "Yes":
        indexer = getMemeNum(url, memeBaseData)
        friends = userLikes(buildUsers, indexer)
        stuff = nextRecommend(friends, picks, memeBaseData)
        url = stuff[0]
        img = getImage(url)
        
        imgplot = plt.imshow(img)
        plt.show()
        
        like = input("Type yes if you liked, no otherwise:")
        picks.append(getMemeNum(url, memeBaseData))
    else:
        #do other stuff
        indexer = getMemeNum(url, memeBaseData)
        friends = userHates(buildUsers, indexer)
        stuff = nextRecommend(friends, picks, memeBaseData)
        url = stuff[0]
        img = getImage(url)
        
        imgplot = plt.imshow(img)
        plt.show()
        
        like = input("Type yes if you liked, no otherwise:")
        picks.append(getMemeNum(url, memeBaseData))

"""
Note to Self:

- create a ban list
- create a no option
"""

