# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:20:54 2020

This program was made to label missing data so that analysis could be done
on school districts lunch program adaptations to COVID-19.

Input: This program requires a folder full of text files that have been web-scraped
from school district websites, a csv that contains training data, and a blank csv
containing the same features with the new text file names as the training data.

Output: A csv that has filled in the blank data csv.

@author: nicholasjernigan@gmail.com
"""

import csv
import os 
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import numpy as np
from inspect import signature
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

#Test out the different classifiers by changing this with a different classifier:
dtree = KNeighborsClassifier(n_neighbors=6, leaf_size=1)
#dtree = DecisionTreeClassifier(max_depth=3)

def create_district_dataframe(txt_folder):
    '''
    This function counts the files in the folder, then makes a dataframe called
    'district_data' with a column that includes each files name in the folder
    and a column that includes the text
    '''
    os.chdir(txt_folder) 
    district_num = len([name for name in os.listdir('.') if os.path.isfile(name)])
    file_names = glob.glob("*.txt")
    dist_data = pd.DataFrame(np.array(file_names).reshape(district_num,1), columns = feature_names)
    
    text = []
    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            #text.append(open(filename, "r", encoding="utf-8").read())
            try:
                text.append(open(filename, "r", encoding="utf-8").read())
            except:    
                text.append(open(filename, "r", encoding="ISO-8859–1").read())
                


    dist_data['text'] = text
    dist_data['text'] = dist_data['text'].astype(str)
    
    
        #Deletes the .txt from district names
    for jj in range(0,len(dist_data['district'])):
        dist_data['district'][jj] = dist_data['district'][jj].replace('.txt','')
    
    return dist_data
#count files in the folder, then make a dataframe that size with "X" features (we choose X)
#Then run each file through each classifier, which adds another row to the dataframe
#Finishes by creating csv of the finished 
    
#NEED TO DO: delete the .txt from each name so that later it can match to training data by name


    
    
def combine_on_name(x_variable_dataframe, y_variable_dataframe,combine_name):
    
    '''
    This function combines the features dataframes and district dataframe on district
    '''
    combined_df = x_variable_dataframe.merge(y_variable_dataframe, left_on=combine_name, right_on=combine_name)
    #, indicator="TRUE" adding this shows that all merged items must appear in both locations
    return combined_df
    ##############TEXT TO FEATURES AFTER COMBINING, MAKES IT MUCH SIMPLER
    
    
def text_to_features(text_column_name, dataframe):
    '''
    This function takes a folder full of txt files
    and reads each one, creating the numeric features so that if can be
    classified. 
    '''
    
    vectorizer = CountVectorizer(stop_words='english')
    text_features = vectorizer.fit_transform(dataframe[text_column_name])
    
    return text_features, vectorizer    



def ml_for_variable(y_variable, x_data, y_data):
    '''
    This function takes the x dataset, y dataset, and the chosen variable from the
    y dataset, and then returns a machine learning algorithem that can be used on
    future unlabled data
    '''
    X = x_data
    y = y_data.filter(items=[y_variable])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
    reg = DecisionTreeClassifier(max_depth=4)
    #can try RandomForestClassifier too

    test_fit = reg.fit(X_train, y_train)

    
    acc_results = print("\n",y_variable,"\n","Accuracy on training set: {:.3f}".format(test_fit.score(X_train, y_train))), print("Accuracy on test set: {:.3f}".format(test_fit.score(X_test, y_test)))
    

    return test_fit, acc_results



def test_variable_accuracy(combined_train_df, text_features):
    '''
    This function can be used to list the accuracy percents of the tested features
    '''
    tlist = []
    for col in combined_train_df:
        tlist.append(col)
        
    dfff = pd.DataFrame(tlist, columns = ['Variable'])
    
    dlist = []
    for col in df:
        alg, results = ml_for_variable(col, text_features, df)
        dlist.append(results)
        
    dfff['Results'] = dlist
    return dfff


'''
Tests out a variety of classification parameters using cross validation
'''


def cross_validate_test(txt_folder,test_df, train_df, item):
    #Creates dataframe from all text files
    #feature_names = ['district']
    dist_data = create_district_dataframe(txt_folder)
    
    #This is to correct the name times for uniformaty
    #Only important if name from the test_df differs from the chosen feature name
    test_df['district'] = test_df['agencyname']

    #Combined dataframe in order to run ML algorithem
    combined_test_df = combine_on_name(dist_data,test_df,'district')
    ctdf = combined_test_df

    #Correcting train_df names
    ###########################BUT SHOULDNT IT BE FROM THE TRAINING DATA? NOT PULLING IN NAMES FROM TEST DATA? TO MAKE SURE IT LINES UP?
    train_df['district'] = train_df['agencyname']
    #Old
    #train_df['district'] = test_df['agencyname']
    combined_train_df = combine_on_name(dist_data,train_df,'district')


    #Below is ML algorithem
    training_features, vectorizern = text_to_features('text', combined_train_df)

    column_list = []  
    for col in combined_test_df:
        column_list.append(col)
        
    #Loops through each column to make a new 'prediction' column for it
    #Will need to delete unnceseary columns
    

    X_train, X_test, y_train, y_test = train_test_split(training_features,combined_train_df[item])
    
    ###Cross validation guide:
    # max_features
    # n_classes
    #n_features
    # n_outputs
    
    
    #Actual cross validation part:
    #Cross validation for EVERYTHING (takes a long time, inserted the found parameters below)

    # define grid for max_depth
    depth_grid = {'n_neighbors': [1,2,4,6,8,10,12,14,16,18,20]}
    grid_search = GridSearchCV(KNeighborsClassifier(),depth_grid,cv=5,return_train_score=True)

    best_model=grid_search.fit(X_train,y_train)
    print("Best n neighbors: ",best_model.best_estimator_.get_params()['n_neighbors'])

    # define grid for max_features
    depth_grid = {'leaf_size': [1,2,4,6,8,10,12,14,16,18,20,40,60 ]}
    grid_search = GridSearchCV(KNeighborsClassifier(),depth_grid,cv=5,return_train_score=True)

    best_model=grid_search.fit(X_train,y_train)
    print("Best leaf size: ",best_model.best_estimator_.get_params()['leaf_size'])
    
    
    dtree.fit(training_features, combined_train_df[item])

    acc_results = print("\n",item,"\n","Accuracy on training set: {:.3f}".format(dtree.score(X_train, y_train))), print("Accuracy on test set: {:.3f}".format(dtree.score(X_test, y_test)))


    
    return df, column_list, acc_results, ctdf

'''
Tests out the above definitions using user input

First creates the dataframe using a folder with just the .txt files

Then creates the 'features' or x variables using those files text

Then reads your training/test csv data, fills NA w zeros

Finally does the machine learning train/test showing results
'''


def ML_Text_First(txt_folder, test_df, train_df, item):
    
    #Creates dataframe from all text files
    #feature_names = ['district']
    dist_data = create_district_dataframe(txt_folder)
    
    
    ###DELETE THIS LATER MAYBE: This is to turn the number id names into numeric data type for leftover code
    dist_data['district'] = pd.to_numeric(dist_data['district'])

    
    #This is to correct the name times for uniformaty
    #Only important if name from the test_df differs from the chosen feature name
    test_df['district'] = test_df['agencyname']

    #Combined dataframe in order to run ML algorithem
    combined_test_df = combine_on_name(dist_data,test_df,'district')
    ctdf = combined_test_df

    #Correcting train_df names
    ###########################BUT SHOULDNT IT BE FROM THE TRAINING DATA? NOT PULLING IN NAMES FROM TEST DATA? TO MAKE SURE IT LINES UP?
    #train_df['district'] = train_df['agencyname']
    #Old
    #train_df['district'] = test_df['agencyname']
    combined_train_df = combine_on_name(dist_data,train_df,'district')


    #Below is ML algorithem
    training_features, vectorizern = text_to_features('text', combined_train_df)

    column_list = []  
    for col in combined_test_df:
        column_list.append(col)
        
    #Loops through each column to make a new 'prediction' column for it
    #Will need to delete unnceseary columns
    

    X_train, X_test, y_train, y_test = train_test_split(training_features,combined_train_df[item])
    
    ###Cross validation guide:
    # max_features
    # n_classes
    #n_features
    # n_outputs
    dtree.fit(training_features, combined_train_df[item])

    acc_results = print("\n",item,"\n","Accuracy on training set: {:.3f}".format(dtree.score(X_train, y_train))), print("Accuracy on test set: {:.3f}".format(dtree.score(X_test, y_test)))


    pickup_list = []
    any_text_indicator_list = []



    #ADDED CODE, Delete later since this is unique to this case
    combined_test_df['text'] = combined_test_df['text_x']

    for k in range(0,len(combined_test_df)):
        newt = combined_test_df['text'][k]
        newf = vectorizern.transform([newt])
        #if print(dtree.predict(newf)) == "[1]":
        #    pickup_list.append(1)
        #else:
        #    pickup_list.append(0)
        pickup_list.append(dtree.predict(newf))
        
        #Part checks the text to see if there is text in it and then adds indicator to list
        word_count = len(combined_test_df['text'][k].split())
        
        if (word_count < 10):
        
            any_text_indicator_list.append(1)
        else:
            any_text_indicator_list.append(0)
             
        
    #------------------------------------------------------
    for i in pickup_list:
        i = i.tostring()

    #1st time to make df *#*#*#*#*#*#*#*# need to figure out how to do########################################
    df = pd.DataFrame(pickup_list,columns=[item +'_predicted'])
    df['district'] = combined_test_df['district']
    df['no_words'] = any_text_indicator_list
    #Delete brackets
    #df[item +'_predicted'] = df[item +'_predicted'].str[0]
    
    #Once done for all important variables, export df to csv
    #df.to_csv(r"C:\Users\nicho\Downloads\USDA_Predictions.csv")
    return df, column_list, acc_results, ctdf
    
def ML_Text_Second(txt_folder, test_df, train_df, item, output_df):
    
   
    #Creates dataframe from all text files
    feature_names = ['district']
    dist_data = create_district_dataframe(txt_folder)
    ###DELETE THIS LATER MAYBE: This is to turn the number id names into numeric data type for leftover code
    dist_data['district'] = pd.to_numeric(dist_data['district'])
    #This is to correct the name times for uniformaty
    #Only important if name from the test_df differs from the chosen feature name
    test_df['district'] = test_df['agencyname']

    #Combined dataframe in order to run ML algorithem
    combined_test_df = combine_on_name(dist_data,test_df,'district')
    ctdf = combined_test_df

    #Correcting train_df names
    ###########################BUT SHOULDNT IT BE FROM THE TRAINING DATA? NOT PULLING IN NAMES FROM TEST DATA? TO MAKE SURE IT LINES UP?
    #train_df['district'] = train_df['agencyname']
    #Old
    #train_df['district'] = test_df['agencyname']
    combined_train_df = combine_on_name(dist_data,train_df,'district')


    #Below is ML algorithem
    training_features, vectorizern = text_to_features('text', combined_train_df)

    column_list = []  
    for col in combined_test_df:
        column_list.append(col)
        
    #Loops through each column to make a new 'prediction' column for it
    #Will need to delete unnceseary columns
    

    X_train, X_test, y_train, y_test = train_test_split(training_features,combined_train_df[item])

    dtree.fit(training_features, combined_train_df[item])


    acc_results = print("\n",item,"\n","Accuracy on training set: {:.3f}".format(dtree.score(X_train, y_train))), print("Accuracy on test set: {:.3f}".format(dtree.score(X_test, y_test)))


    #ADDED CODE, Delete later since this is unique to this case
    combined_test_df['text'] = combined_test_df['text_x']
    
    pickup_list = []
    for k in range(0,len(combined_test_df)):
        newt = combined_test_df['text'][k]
        newf = vectorizern.transform([newt])
        #if print(dtree.predict(newf)) == "[1]":
        #    pickup_list.append(1)
        #else:
        #    pickup_list.append(0)
        pickup_list.append(dtree.predict(newf))
                
    for i in pickup_list:
        i = i.tostring()
        
    #1st time to make df *#*#*#*#*#*#*#*# need to figure out how to do########################################
    output_df[item + '_predicted'] = pickup_list

    #Once done for all important variables, export df to csv
    #df.to_csv(r"C:\Users\nicho\Downloads\USDA_Predictions.csv")
    return output_df

#Need to add indicator for if the text file has more text beyond the first | 




    
#################################################################################################
###DEFINITION ENDS HERE
#################################################################################################
    






choose_directory = r"C:\Users\nicho\Downloads\USDA_Covid_Project"
os.chdir(choose_directory) 
txt_folder = r"C:\Users\nicho\Downloads\USDA_Covid_Project\all_text"




feature_names = ['district']

'''
process to make blank DF by hand, requires some hand editing to match it to the training template
for some reason exporting to csv is not working, but exporting to xlsx does


dist_dataframe = create_district_dataframe(txt_folder)
dist_dataframe.to_csv(r"C:\Users\nicho\Downloads\full_template.csv")

Saving this excel as a csv (after exporting as excel) seems to work
dist_dataframe.to_excel(r"C:\Users\nicho\Downloads\full_template.xlsx")


'''




choose_directory = r"C:\Users\nicho\Downloads\USDA_Covid_Project"
os.chdir(choose_directory) 
test_df = pd.read_excel("full_temp.xlsx")
train_df = pd.read_csv("training_data1.csv")

#dtree = RandomForestClassifier()
dtree = LinearSVC()
item ='meals_provided'
df, column_list, acc_results, ctdf = ML_Text_First(txt_folder, test_df, train_df, item)

dtree = LogisticRegression()
item = 'P-EBT'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, df)

dtree = LinearSVC()
item = 'all_schools'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, df)

dtree = LogisticRegression()
item = 'community_sites'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LogisticRegression()
item = 'pickup'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LogisticRegression()
item = 'school_bus_deliv'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LogisticRegression()
item = 'home_deliv'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LogisticRegression()
item = 'drive_through_pickup'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LogisticRegression()
item = 'daily_pickup'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LinearSVC()
item = 'parent_pickup'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LinearSVC()
item = 'school_id_required'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LogisticRegression()
item = 'meals_for_all_children'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LinearSVC()
item = 'weekend_meals'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = RandomForestClassifier()
item = 'free_meals'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)

dtree = LogisticRegression()
item = 'not_daily'
dfout = ML_Text_Second(txt_folder, test_df, train_df, item, dfout)


comb_df = combine_on_name(dfout,test_df,'district')
comb_df = comb_df.drop(columns=['text'])
comb_df.to_csv(r"C:\Users\nicho\Downloads\USDA_Predictions.csv")
choose_directory = r"C:\Users\nicho\Downloads"
os.chdir(choose_directory) 
with open("USDA_Predictions.csv", encoding="utf8", errors='ignore') as f:
    text = f.read()
text = ''.join([i for i in text]) \
    .replace("[", "")
text = ''.join([i for i in text]) \
    .replace("]", "")
x = open("output.csv","w", encoding="utf8")
x.writelines(text)
x.close()




