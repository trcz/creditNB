
#EXAMPLE APPLICATION OF NAIVE BAYES FOR FEATURES WITH MIXED DISTRIBUTIONS

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore") #Scaling values can raise warning about int->float conversion, it suits me so I want to ignore this

#Functional part
#Can be expanded to make script more universal


#Loading data from .csv file, delimiter is semicolon due to language settings in my Windows
def loadCsv(filename):
    with open(filename, "rt"):
        preset = np.genfromtxt(open(filename, "rt"), delimiter=';', dtype=None)
    return preset


#Example function to make script utiilities more flexible
#It will split features based on their distribution
#Args are dataset and lists of indexes representing features' numbers
def splitByFeatures(set, gauNo, beNo, mnNo, clNo):
    gau = set[:, gauNo]
    be = set[:, beNo]
    mn = set[:, mnNo]
    cl = set[:, clNo]
    return [gau, be, mn, cl]

#Loading file
name = "credit_dataset2.csv"
dataset = loadCsv(name)
print('Loaded data file {0} with {1} rows'.format(name, np.size(dataset, 0)))


gauList = [0,4,11,12,13,14,15,16,17,18,19,20,21,22] #columns with gauss distribution
beList = [1] #columns with bernoulli distribution
mnList = [2,3,5,6,7,8,9,10] #columns with multinomial distribution
clList = [23] #columns with class variables
preSets = splitByFeatures(dataset, gauList, beList, mnList, clList)


#Grouping columns by their distribution
gauSet = preSets[0]
beSet = preSets[1]
mnSet = preSets[2]
scaler = MinMaxScaler(copy=True, feature_range=(1, 10)) #scaling values will remove negative ones to make MultinomialNB possible
scaler.fit(mnSet)
mnSet = scaler.transform(mnSet)
clSet = preSets[3] #class variables

#Splitting each group of features for training and test, I choose standard ratio of 66,(6)%
splitRatio = int(np.size(dataset, 0)*0.67)

#Training groups
gauTrPrep = gauSet[0:splitRatio]
beTrPrep = beSet[0:splitRatio]
mnTrPrep = mnSet[0:splitRatio]
clTrain = clSet[0:splitRatio]
#Test groups
gauTsPrep = gauSet[splitRatio:]
beTsPrep = beSet[splitRatio:]
mnTsPrep = mnSet[splitRatio:]
clTest = clSet[splitRatio:]


#Machine learning part
#To deal with mix of discrete and continous features I've decided to base my machine learning on probabilities instead of values

#Creating training models
gauModel = GaussianNB()
gauModel.fit(gauTrPrep, clTrain.ravel())
beModel = BernoulliNB()
beModel.fit(beTrPrep, clTrain.ravel())
mnModel = MultinomialNB()
mnModel.fit(mnTrPrep, clTrain.ravel())

#Getting training probabilities
gauTrain = gauModel.predict_proba(gauTrPrep)
beTrain = beModel.predict_proba(beTrPrep)
mnTrain = mnModel.predict_proba(mnTrPrep)


#Getting test probabilities
#These are my testing variables in fact
gauTest = gauModel.predict_proba(gauTsPrep)
beTest = beModel.predict_proba(beTsPrep)
mnTest = mnModel.predict_proba(mnTsPrep)

#In this moment splitted groups are getting stacked again
wholeTrain = np.hstack((gauTrain, beTrain, mnTrain))
wholeTest = np.hstack((gauTest, beTest, mnTest))

#Creating final prediction model
#I'm using rather big portion of data here (30000 records)
#I can assume (based on central limit theorem) my probabilities got normal distribution (treated as new features)
finalModel = GaussianNB()
finalModel.fit(wholeTrain, clTrain.ravel())

#Making predictions based on test probabilities
predictions = finalModel.predict(wholeTest)

#Changing type of my testing class values from np.ndarray to python list
realValues = clTest.ravel()

#Counting precision
matchCount = 0
for i in range(len(predictions)):
    if predictions[i] == realValues[i]:
        matchCount += 1

print("{0} percent of test values predicted well".format(float(matchCount/len(predictions))*100))