# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:13:33 2019

@author: Gosia
"""

import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, tree
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
import sklearn
import time
import numpy as np
import numbers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

#helper function: get sum of var
def summ_var(obs,var_type):
    ''' obs: a dictionary read from json file. 
        var_type: either 'magnitude' or 'score'. '''    
    sum_var=0
    for i in obs['sentences']:
        sum_var+=i['sentiment'][var_type]
    return sum_var

#helper function: get entities (labels, tags) from description as string
def get_tags(obs):
    final_tags = []
    ent = obs['entities'] # list of entities
    for i in ent:
        final_tags.append(i['name'])
    joined = ' '.join(final_tags)
    return joined 

def merge_tags(array):
    try:
        res = ' '.join(array)
    except TypeError:
        res = ' '.join(str(array))
    return res

def get_all_data(ds_type, directory, pics):
    ''' Extracts features: text sentiment and metadata images.
        Merges it with CSV and returns the final file.
        ds_type - train or test, which file are you separating
        directory - dir to place with test.csv, train.csv and folders with metadata
    '''
    
    #-1.Set the directory
    os.chdir(directory)
    folder= ds_type + "_sentiment/" 
    folder2= ds_type + "_metadata/"
    dataset = pd.read_csv(ds_type + '.csv')
    
    
    # 2. GET DESCRIPTION (TEXT) SENTIMENT DATA
    
    files = [f for f in os.listdir(folder)]
    record=[]
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
        sum_mag=summ_var(data,'magnitude') #sum magnitude of all sentences
        sum_score=summ_var(data,'score') #sum score of all sentences
        tags = get_tags(data) #returns a string with comma-separated tags
        record.append([j,doc_mag,doc_score,sent_count,sen1_mag,sen1_score, sum_mag, sum_score, tags])

    
    #Our DataFrame with columns as below:
    rec_df = pd.DataFrame(record,columns=['PetID','docMagnitude','doc_score','sent_count', 'sen1_magnitude','sen1_score', 'sum_mag', 'sum_score','sent_tags'])   
    # lets correct the magnitude and sum of magnitude, dividing it by sentences number
    
    rec_df['doc_mag_corr']= rec_df.docMagnitude / rec_df.sent_count
    rec_df['sum_mag_corr']= rec_df.sum_mag / rec_df.sent_count
    
    del rec_df['docMagnitude']
    # does pet have english description
    sentiment = rec_df #we ignore Adoption Speed merging
    sentiment['has_eng_description']=np.where(sentiment.doc_mag_corr.isnull()==True, 0,1)
    #list(sentiment)   
    
    # 3. GET IMAGES METADATA & MERGE IT WITH THE REST
    
    files = [f for f in os.listdir(folder2)]
    record_img=[]
    for j in files:
        file = folder2+j
      
        with open (file, 'r',encoding="utf8") as f: #read all the files
            obs = json.load(f)
        if type(obs)==list: data=obs[0] #sometimes json's are as list, sometimes dictionary,
        else: data=obs    #here it is handled        
        
        if pics != "ALL": condition = "j[-6] == '1'" #do we want just 1st pic or all
        else: condition = 'True'
        flag=eval(condition)
            
        if flag==True:
        
        #1. CropHintsAnnotation:
    
    #boundingPoly 	: The bounding polygon for the crop region. 
    #The coordinates of the bounding box are in the original image's scale.
    #confidence 	:Confidence of this being a salient region. Range [0, 1].
    #importanceFraction 	: Fraction of importance of this salient region with respect to the original image. 
        
            img_bound_polygon_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            img_bound_polygon_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            img_confidence = data['cropHintsAnnotation'] ['cropHints'] [0] ['confidence']
            try: 
                img_imp_fract = data['cropHintsAnnotation'] ['cropHints'] [0] ['importanceFraction']
            except KeyError:
                img_imp_fract = 0
                
        #2. imagePropertiesAnnotation:
        
            try:
                domcol_r = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
                domcol_g = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
                domcol_b = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            except KeyError:
                domcol_r,domcol_g,domcol_b = 0,0,0
        
        # 3. labelAnnotations: tags, like 'dog', 'puppy' , with topicality score.
            file_keys = list(data.keys())
        
            if 'labelAnnotations' in file_keys:
                file_annots = data['labelAnnotations'][:int(len(data['labelAnnotations']) * 0.3)]
                file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
                file_top_desc = [x['description'] for x in file_annots]
            else:
                file_top_score = np.nan
                file_top_desc = ['']
            meta_tags = ' '.join(file_top_desc)  
            
            file_colors = data['imagePropertiesAnnotation']['dominantColors']['colors']
            file_crops = data['cropHintsAnnotation']['cropHints']

            file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
            file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

            file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()
        
            if 'importanceFraction' in file_crops[0].keys():
                file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
            else:
                file_crop_importance = np.nan
  
            PetID = j[:-7] #PetID
            PetID=PetID.replace('-','') #just in case
            if pics == "ALL": 
                PetID_pic=j[:-5]    #We want PetID with "-picture number.json" part
                pic_no = int(j[-6])
            row = [PetID, img_bound_polygon_x,img_bound_polygon_y, img_confidence, 
                   img_imp_fract, domcol_r, domcol_g, domcol_b, file_top_score, meta_tags,
                   file_color_score, file_color_pixelfrac, file_crop_conf, file_crop_importance]
            if pics == "ALL": 
                row.append(PetID_pic) 
                row.append(pic_no) 
            record_img.append(row)
            row=[]  #clear
    
    columns =  ['PetID', 'img_bound_polygon_x','img_bound_polygon_y','img_confidence',
               'img_imp_fract','domcol_r','domcol_g','domcol_b','file_top_score', 'img_tags',
               'file_color_score', 'file_color_pixelfrac', 'file_crop_conf', 'file_crop_importance']
    if pics == "ALL":  
        columns.append('PetID_pic')
        columns.append('pic_no')
    
    rec_img_df = pd.DataFrame(record_img, columns = columns)
    
    rec_img_df.set_index('PetID')
    # 4. MERGE ALL FILES
    everything = pd.merge( dataset, rec_img_df,  how="outer", left_on='PetID' , right_on='PetID', suffixes=('_img','_dataset'))
    
    all_cols = pd.merge(everything, sentiment, how="left", left_on='PetID' , right_on='PetID', suffixes=('_every','_sent'))
      
    # 5. CLEAR FINAL DATASET
    df = all_cols
    return df

def count_rescuer(df):
    rescuer_count = df.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']
    df = df.merge(rescuer_count, how='left', on='RescuerID')
    del df["RescuerID"]
    return df

def breed_mapping(df):
    labels_breed = pd.read_csv('breed_labels.csv')
    breed_main = df[['Breed1']].merge(labels_breed, how='left',
    left_on='Breed1', right_on='BreedID', suffixes=('', '_main_breed'))

    breed_main = breed_main.iloc[:, 2:]
    breed_main = breed_main.add_prefix('main_breed_')

    breed_second = df[['Breed2']].merge(labels_breed, how='left',
    left_on='Breed2', right_on='BreedID',   suffixes=('', '_second_breed'))

    breed_second = breed_second.iloc[:, 2:]
    breed_second = breed_second.add_prefix('second_breed_')

    df_breed = pd.concat([df, breed_main, breed_second], axis=1)
    categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']
    for i in categorical_columns:
        df_breed.loc[:, i] = pd.factorize(df_breed.loc[:, i])[0]
        
    return df_breed

def aggregate_features(data):
    ''' returns features per pet, not per picture '''
    #we separate columns connected to image: only those will be summed. Rest: only averaged
    img_cols = ['img_bound_polygon_x','img_bound_polygon_y','img_confidence',
               'img_imp_fract','domcol_r','domcol_g','domcol_b','file_top_score', 'img_tags',
               'file_color_score', 'file_color_pixelfrac', 'file_crop_conf', 'file_crop_importance']
    sent_cols = [ 'docMagnitude', 'doc_score', 'sent_count', 'sen1_magnitude', 'sen1_score', 'sum_mag',
                 'sum_score', 'sent_tags', 'doc_mag_corr', 'sum_mag_corr', 'has_eng_description']
    added_cols = img_cols + sent_cols
    final_df = pd.DataFrame()
    cols = list(data)
    for col in cols:
        if isinstance(data[col][0], numbers.Number): #if numeric, we aggregate:
            column = data.groupby(['PetID'])[col].mean()
            if col in added_cols: #name: either 'normal' or with suffix
                final_df[col+'_Mean'] = column
            else:
                final_df[col] = column
            if col in img_cols: #also sum
                column2 = data.groupby(['PetID'])[col].sum()
                final_df[col+'_Sum'] = column2
        
        else:  # text object   
            column = data.groupby(['PetID'])[col].unique()
            if col=='PetID': column = [x[0] for x in column]
            #elif col == 'img_tags' or col == 'sent_tags' or col == 'Description': 
            else:   #column=column.reset_index()
                column = column.map(merge_tags)
            final_df[col]= column
            
    return final_df


def interpret_text(df): 
    X_temp = df.copy()
    text_columns = ['Description', 'sent_tags', 'img_tags']
    X_text = X_temp[text_columns]

    for i in X_text.columns:
        X_text.loc[:, i] = X_text.loc[:, i].fillna('<MISSING>')
        X_text[i] = X_text[i].replace(['[ n a n ]'],'<MISSING>')
    n_components = 5
    text_features = []    
    # Generate text features:
    for i in X_text.columns:        
        # Initialize decomposition methods:
        print('generating features from: {}'.format(i))
        svd_ = TruncatedSVD(
            n_components=n_components, random_state=1337)
        nmf_ = NMF(
            n_components=n_components, random_state=1337)
        
        tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)
        svd_col = svd_.fit_transform(tfidf_col)
        svd_col = pd.DataFrame(svd_col)
        svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
        
        nmf_col = nmf_.fit_transform(tfidf_col)
        nmf_col = pd.DataFrame(nmf_col)
        nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))
        
        text_features.append(svd_col)
        text_features.append(nmf_col)    
        
    # Combine all extracted features:
    text_features = pd.concat(text_features, axis=1)
    
    # Concatenate with main DF:
    X_temp = X_temp.reset_index(drop=True)
    X_temp = pd.concat([X_temp, text_features], axis=1)
    
    # Remove raw text columns:
    for i in X_text.columns:
        X_temp = X_temp.drop(i, axis=1)
    return X_temp

def merge_PetID(ID_source, y):
    ID_source=ID_source.reset_index()
    try:
        y=y.reset_index(drop=True)
    except AttributeError:
        pass
    y_df = pd.DataFrame(y)
    y_final = pd.concat([ID_source['PetID'],y_df],ignore_index=True, axis=1)
    y_final.columns=["PetID","AdoptionSpeed"]
    return y_final

# TRAIN V4

df=get_all_data(ds_type="train",directory="../input/pet_adoption_prediction/",pics="ALL")

df_v2 = aggregate_features(df)
df_v3 = interpret_text(df_v2)
df_v4 = count_rescuer(df_v3)
df_v5 = breed_mapping(df_v4)

#all_train= pd.read_csv('all_train_data_V3.csv')
#df_v5[['dullness', 'average_pixel_width', 'blurrness']] = all_train[['dullness', 'average_pixel_width', 'blurrness']]

del df_v5["PetID_pic"]
del df_v5["pic_no"]

df_v5.to_csv("all_train_data_V4.csv", index=False)

#TRAIN V5 - V4 without some features

df_v6 = df_v5.copy()
df_v6 =  df_v6.drop([
        #'dullness','blurrness','average_pixel_width',
         'has_eng_description_Mean','sen1_magnitude_Mean','sen1_score_Mean',
          'domcol_r_Mean',  'domcol_r_Sum',  'domcol_g_Mean','domcol_g_Sum',
 'domcol_b_Mean','domcol_b_Sum','img_bound_polygon_x_Mean',
 'img_bound_polygon_x_Sum',  'img_bound_polygon_y_Mean','img_bound_polygon_y_Sum'],
axis=1 )

df_v6.to_csv("all_train_data_V5_baseline.csv", index=False)

#TEST V4

df_tst = get_all_data(ds_type="test",directory="../input/pet_adoption_prediction/",pics="ALL")
df_tst_v2 = aggregate_features(df_tst)
df_tst_v3 = interpret_text(df_tst_v2)
df_tst_v4 = count_rescuer(df_tst_v3)
df_tst_v5 = breed_mapping(df_tst_v4)

#all_test = pd.read_csv('all_test_data_V3.csv')
#df_tst_v5[['dullness', 'average_pixel_width', 'blurrness']] = all_test[['dullness', 'average_pixel_width', 'blurrness']]

del df_tst_v5["PetID_pic"]
del df_tst_v5["pic_no"]

df_tst_v5.to_csv("all_test_data_V4.csv", index=False)

# TEST V5 - V4 without some features

df_tst_v6 = df_tst_v5.copy()
df_tst_v6 =  df_tst_v6.drop([
        #'dullness','blurrness','average_pixel_width',
         'has_eng_description_Mean','sen1_magnitude_Mean','sen1_score_Mean',
          'domcol_r_Mean',  'domcol_r_Sum',  'domcol_g_Mean','domcol_g_Sum',
 'domcol_b_Mean','domcol_b_Sum','img_bound_polygon_x_Mean',
 'img_bound_polygon_x_Sum',  'img_bound_polygon_y_Mean','img_bound_polygon_y_Sum'],
axis=1 )

df_tst_v6.to_csv("all_test_data_V5_baseline.csv", index=False)



