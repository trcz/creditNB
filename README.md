# creditNB
Naive Bayes based on credit card dataset  

Dataset source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#  
Language: python3  
External libraries: numpy and sklearn  

I processed data by myself (removal of ID and names)  

There are 23 features (X1-X23) and 1 class variable (Y)  
Features X1, X5 and X12-X23 can be described by normal distribution  
X2 is Bernoulli's distribution  
In X3, X4, X6-X11 there are discrete values so I've assumed they have multinomial distribution  
  
Achieved precision of prediction: over 80% (70-76% is standard)  
  
To-do list:  
-generalisation of script for more flexible use  
-making preprocessing data automatic instead of manual  
