# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 19:03:07 2019

@author: Gosia
"""

import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, tree
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import sklearn
import time

#read file
with open('C:/Users/Gosia/Desktop/MLIP/train_sentiment/a63f6c6a1.json', 'r') as f:
    diction = json.load(f)
    
#function: average score of sentences / average magnitude of sentences
def avg_var(obs,var_type):
    sum_mag=0
    num = len(obs['sentences'])
    for i in obs['sentences']:
        sum_mag+=i['sentiment'][var_type]
    return sum_mag / num
        
avg_var(diction,'magnitude')
avg_var(diction,'score')
    
    
#key-sentences -> sentiment -> magnitude, score
#key-documentSentiment -> magnitude, score
#key-entities -> Type

folder="C:/Users/Gosia/Desktop/MLIP/train_sentiment/"
os.chdir("C:/Users/Gosia/Desktop/MLIP")
train=pd.read_csv('train.csv')
    
#get all the file names
files = [f for f in os.listdir(folder)]
record=[]
start = time.time()   
for j in files:
    file = folder+j
    
    with open (file, 'r',encoding="utf8") as f:
        obs = json.load(f)
    if type(obs)==list: data=obs[0] 
    else: data=obs
    j=j[:-5]#ID without ".json" part
    doc_mag=data['documentSentiment']['magnitude']
    doc_score=data['documentSentiment']['score']
    sent_count = len(data['sentences']) #how many sentences?
    sen1_mag=data['sentences'][0]['sentiment']['magnitude'] #magnitude of the 1st sentence
    sen1_score=data['sentences'][0]['sentiment']['score'] #score of the 1st sentence
    avg_mag=avg_var(data,'magnitude')
    avg_score=avg_var(data,'score')
    record.append([j,doc_mag,doc_score,sent_count,sen1_mag,sen1_score, avg_mag, avg_score])

end = time.time()
print(end-start)

rec_df = pd.DataFrame(record,columns=['PetID','docMagnitude','docScore','sent_count', 'sen1_magnitude','sen1_score','avg_mag', 'avg_score'])   

#plot below shows that doc magnitude is way higher for longer descriptions: 
rec_df.plot.scatter('sent_count', 'docMagnitude')
#therefore, lets correct magnitude, dividing it by sentences number

rec_df['docMag_corr']= rec_df.docMagnitude / rec_df.sent_count


#returns the count of values of a given column
count = pd.value_counts(rec_df.sent_count)
count_df = pd.DataFrame(count)
count_df.sent_count.plot.bar()
#rec_df['sent_count']=np.where(rec_df.sent_count>=21, 20, rec_df.sent_count) #get rid of superlong descriptions
#rec_df.PetID[rec_df['sent_count']==63] #find out ID with a certain count value (outliers!)

    
#magnitude: strength of emotion (0.0 -> inf)
#score: -1 negative, 1 positive
#salience: relevance to the overall text


rec_df.plot.scatter('docMagnitude', 'avg_mag')
rec_df.plot.scatter('docMag_corr', 'avg_mag')
rec_df.plot.scatter('docScore', 'avg_score')

#document score vs 1st sentence score:
plt.hist(rec_df.docScore, label='Doc Score')#, alpha=0.5)
plt.hist(rec_df.sen1_score, label='1st sentence Score', alpha=0.3)
plt.legend(loc='upper left')
plt.show()

#document magnitude vs 1st sentence magnitude:
plt.hist(rec_df.docMagnitude, label='Doc Magnitude')
plt.hist(rec_df.sen1_magnitude, label='1st sentence Magnitude', alpha=0.3)
plt.legend(loc='upper left')
plt.show()

#document Corrected magnitude vs 1st sentence magnitude:
plt.hist(rec_df.docMag_corr, label='Doc Corrected Magnitude')
plt.hist(rec_df.sen1_magnitude, label='1st sentence Magnitude', alpha=0.3)
plt.legend(loc='upper left')
plt.show()

#document Corrected magnitude vs avg sentence magnitude:
plt.hist(rec_df.docMag_corr, label='Doc Corrected Magnitude')
plt.hist(rec_df.avg_mag, label='Average Magnitude', alpha=0.3)
plt.legend(loc='upper right')
plt.show()

#document score vs avg sentence score:
plt.hist(rec_df.docScore, label='Doc Score')#, alpha=0.5)
plt.hist(rec_df.avg_score, label='Average Score', alpha=0.3)
plt.legend(loc='upper left')
plt.show()

#AdoptionSpeed - our label
plt.hist(train.AdoptionSpeed)
plt.show()

#join sentiment and adoption speed:
train.set_index('PetID')
rec_df.set_index('PetID')
everything = train.join(rec_df.set_index('PetID'), on='PetID')
sentiment = everything[['PetID','docMagnitude','docScore','sent_count','sen1_magnitude','sen1_score','AdoptionSpeed', 'avg_mag','avg_score']]

#Some pets dont have description at all
sentiment['Has_description']=np.where(sentiment.docMagnitude.isnull()==True, 0,1)
has_desc=pd.value_counts(sentiment.Has_description)
has_desc.plot.bar()   
#show magnitudes & scores per AdoptionSpeed score
for i in range(5):
    plt.hist(sentiment[sentiment.AdoptionSpeed==i].docMagnitude.dropna(), color=np.random.rand(3))
    plt.title('Adoption speed: ' + str(i))
    plt.show()
    
for i in range(5):
    plt.hist(sentiment[sentiment.AdoptionSpeed==i].docScore.dropna(), color=np.random.rand(3))
    plt.title('Adoption speed: ' + str(i))
    plt.show()
    
list(sentiment)    
    
'''
To analysis:
    1. Number of sentences
    2. Does it have a description at all
    3. docScore
    4. docMagnitude
    
Notices:
    avg is linear to overall (for score and corrected magnitude)
'''
     
#train/test separation
sentiment=sentiment.dropna()    
sent_train, sent_test = train_test_split(sentiment, test_size=0.25)#,random_state=6542)
x_train=sent_train[['docMagnitude',
 'docScore',
 'sent_count',
 'sen1_magnitude',
 'sen1_score',
 'Has_description']]
y_train=sent_train[['AdoptionSpeed']]
x_test=sent_test[['docMagnitude',
 'docScore',
 'sent_count',
 'sen1_magnitude',
 'sen1_score',
 'Has_description']]
y_test=sent_test[['AdoptionSpeed']]

# REG LOG: TERRIBLE. Score: 0.29

regr = linear_model.LogisticRegression()#solver='lbfgs', multi_class='multinomial')
#df.isnull().values.any() <- does it have any NAN
# Train the model using the training sets
regr.fit(x_train,y_train)

# Make predictions using the testing set
sent_y_pred = regr.predict(x_test)

sklearn.metrics.confusion_matrix(y_test,pd.DataFrame(sent_y_pred))

score=regr.score(x_test,y_test)
    
    
#TREE: TERRIBLE. Score: 0.28
    
dt = tree.DecisionTreeClassifier()
dt.fit(x_train,y_train)

# Make predictions using the testing set
sent_y_pred = dt.predict(x_test)

sklearn.metrics.confusion_matrix(y_test,pd.DataFrame(sent_y_pred))

score=dt.score(x_test,y_test)

#KERAS: TERRIBLE. Score: 0.30

import tensorflow as tf
mnist = tf.keras.datasets.mnist

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0  #normalizacja!

x_train=x_train.values
y_train=y_train.values
x_test=x_test.values
y_test=y_test.values

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  #dense layer - jak full. warstwa ukryta 512: liczba neuronów. 
  tf.keras.layers.Dense(100, activation=tf.nn.relu), #rectified linear unit (prostownik?)! 512->100
  #smoother ReLu: f(x) = ln(1 + exp(x))
  tf.keras.layers.Dropout(0.2), #fraction of units to drop
  tf.keras.layers.Dense(5, activation=tf.nn.softmax) #ost. warstwa: 5 klas 0,1,2,3 etc
])
model.compile(optimizer='adam', #how accurate is the training. AdamOptimizer()
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)


#PCA - nothing matches
p_comp, variance = myPCA(x_train)
principalDf = pd.DataFrame(data=p_comp, columns = ['PC_1', 'PC_2'])
y_train=y_train.reset_index()
finalDf = pd.concat([principalDf, y_train], axis = 1)
finalDf.plot.scatter('PC_1', 'PC_2',c='AdoptionSpeed',colormap='viridis')


