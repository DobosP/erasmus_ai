# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:06:50 2019

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


#-1.Set the directory (change it to your ofc), read the train.csv for later :
os.chdir("C:/Users/Gosia/Desktop/MLIP")
folder="train_sentiment/"
folder2="train_metadata/"
train=pd.read_csv('train.csv')

#0. Bonus: read the dominant colour for image metadata.

with open(folder2 + '000a290e4-1.json', 'r') as f:
    diction0 = json.load(f)

#returns a dictionary with red, green, blue values for the color
diction0['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']


files = [f for f in os.listdir(folder2)]
record_img=[]
start = time.time()  # it can take around a minute
for j in files:
    file = folder2+j
    
    with open (file, 'r',encoding="utf8") as f: #read all the files
        obs = json.load(f)
    if type(obs)==list: data=obs[0] #sometimes json's are as list, sometimes dictionary,
    else: data=obs    #here it is handled
    
    if j[-6]=="1":         #WE ONLY TAKE 1ST PICTURE DATA
        j=j[:-7]    #We want PetID without "-picture number.json" part
    #returns a dictionary with red, green, blue values for the dominant color on 1st picture
        dom_color = obs['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']
        record_img.append([j, dom_color])

end = time.time()
print(end-start)

rec_img_df = pd.DataFrame(record_img, columns = ['PetID', 'dom_color' ])



#1. Read example file
with open(folder + 'a63f6c6a1.json', 'r') as f:
    diction = json.load(f)
    
 #File structure (dictionary) - keys that we'll use:
#key 1: sentences -> sentiment -> magnitude, score
#key 2: documentSentiment -> magnitude, score   
 
#magnitude: strength of emotion (0.0 -> inf)
#score: -1 negative, 1 positive
    
#2. Function: average score of sentences or average magnitude of sentences
def avg_var(obs,var_type):
    ''' obs: a dictionary read from json file. 
        var_type: either 'magnitude' or 'score'. '''
    sum_var=0
    num = len(obs['sentences'])
    for i in obs['sentences']:
        sum_var+=i['sentiment'][var_type]
    return sum_var / num
        
avg_var(diction,'magnitude')
avg_var(diction,'score')
    
    
#3. Get all the TRAINING files:

    
#get all the file names
files = [f for f in os.listdir(folder)]
record=[]
start = time.time()  # it takes around 10-30 seconds
for j in files:
    file = folder+j
    
    with open (file, 'r',encoding="utf8") as f: #read all the files
        obs = json.load(f)
    if type(obs)==list: data=obs[0] #sometimes json's are as list, sometimes dictionary,
    else: data=obs                 #here it is handled
    j=j[:-5]    #We want PetID without ".json" part
    doc_mag=data['documentSentiment']['magnitude'] #whole description magnitude!
    doc_score=data['documentSentiment']['score']  #whole desription score!
    sent_count = len(data['sentences']) #how many sentences?
    sen1_mag=data['sentences'][0]['sentiment']['magnitude'] #magnitude of the 1st sentence
    sen1_score=data['sentences'][0]['sentiment']['score'] #score of the 1st sentence
    avg_mag=avg_var(data,'magnitude') #average magnitude of all sentences
    avg_score=avg_var(data,'score') #average score of all sentences
    record.append([j,doc_mag,doc_score,sent_count,sen1_mag,sen1_score, avg_mag, avg_score])

end = time.time()
print(end-start)

#4. Our DataFrame with columns as below:
rec_df = pd.DataFrame(record,columns=['PetID','docMagnitude','doc_score','sent_count', 'sen1_magnitude','sen1_score','avg_mag', 'avg_score'])   

#5. The Plot below shows that doc magnitude is way higher for longer descriptions: 
rec_df.plot.scatter('sent_count', 'docMagnitude')

#6. Therefore, lets correct the magnitude, dividing it by sentences number

rec_df['doc_mag_corr']= rec_df.docMagnitude / rec_df.sent_count


#7. Code below returns the count of values of a given column
count = pd.value_counts(rec_df.sent_count)
count_df = pd.DataFrame(count)
count_df.sent_count.plot.bar()

#rec_df['sent_count']=np.where(rec_df.sent_count>=21, 20, rec_df.sent_count) #get rid of superlong descriptions
#rec_df.PetID[rec_df['sent_count']==63] #find out ID with a certain count value (outliers!)

#8. Plots below show linear relation between overall and average score and magnitude.
# Therefore, let's not include the averages.

rec_df.plot.scatter('doc_mag_corr', 'avg_mag')
rec_df.plot.scatter('doc_score', 'avg_score')

#9. Pots comparing overall scores/magnitudes to 1st sentence scores/magnitudes

#document score vs 1st sentence score:
plt.hist(rec_df.doc_score, label='Doc Score')#, alpha=0.5)
plt.hist(rec_df.sen1_score, label='1st sentence Score', alpha=0.3)
plt.legend(loc='upper left')
plt.show()

#document Corrected magnitude vs 1st sentence magnitude:
plt.hist(rec_df.doc_mag_corr, label='Doc Corrected Magnitude')
plt.hist(rec_df.sen1_magnitude, label='1st sentence Magnitude', alpha=0.3)
plt.legend(loc='upper left')
plt.show()

#AdoptionSpeed - our label
plt.hist(train.AdoptionSpeed)
plt.show()

#10. Join sentiment, dominant color adoption speed:
train.set_index('PetID')
rec_df.set_index('PetID')
rec_img_df.set_index('PetID')
everything = train.join(rec_df.set_index('PetID'), on='PetID')
everything = everything.join(rec_img_df.set_index('PetID'), on='PetID')
sentiment = everything[['PetID','doc_mag_corr','doc_score','sent_count','sen1_magnitude','sen1_score', 'dom_color','AdoptionSpeed']]

#11. Some pets dont have ENGLISH description - can be meaningful
#0 - no overall magnitude, therefore - no description. 1 - otherwise.

sentiment['has_eng_description']=np.where(sentiment.doc_mag_corr.isnull()==True, 0,1)
has_desc=pd.value_counts(sentiment.has_eng_description)
has_desc.plot.bar()   
has_desc


#12. Lots of histograms: show corr. magnitudes & scores PER AdoptionSpeed score
# No actual differences between the categories
for i in range(5):
    plt.hist(sentiment[sentiment.AdoptionSpeed==i].doc_mag_corr.dropna(), color=np.random.rand(3))
    plt.title('Corrected magnitude of Adoption speed: ' + str(i))
    plt.show()
    
for i in range(5):
    plt.hist(sentiment[sentiment.AdoptionSpeed==i].doc_score.dropna(), color=np.random.rand(3))
    plt.title('Score of Adoption speed: ' + str(i))
    plt.show()
    
#13. List of columns of dataset (PetID & AdoptionSpeed included)
list(sentiment)    
    
'''
Features to add to analysis:
    1. doc_mag_corr : corrected overall magnitude per description
    2. doc_score : overall score (positive/negative)
    3. sent_count: sentence count, how many does the description have
    4. sent1_magnitude : magnitude of 1st sentence
    5. sent1_score : score of 1st sentence
    6. has_eng_description : 0 it doesn't, 1 it does
    7. dom_color : a dictionary with RGB values for a dominant color for 1st picture

Remarks:

Dominant color is hard to analyse for now; I did not separate the green, red, blue
values, because they only work well together (as a color they create).
    
'''


