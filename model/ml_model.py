"""
filename: ml_model
Machine Learning Model
Feeder/ML
This script will combine all features from various function scripts and feed them into the ML pipeline. 
"""

# Author: Chelsea Momoh
# Date: 2026-04-06
# Version: 1.0

##################SETUP##########################
#UPDATE PYCACHE BEFORE EACH RUN: 
import subprocess
subprocess.run([
    "find", "/workspaces/Scholarly-Impact---Automation-Pipeline-Work/src",
    "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"
])

import json
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath("/workspaces/Scholarly-Impact---Automation-Pipeline-Work/src/model/ml_model")))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from feature import scraped_features #contains the functions: field_similarity_feature, 
                                         #institution_similarity, & coauthorship_similarity  
                                         #field_similarity_rapidfuzz 
   
from feature import disambiguation_feature #name_similarity, disambiguate_names

from data import raw_data_converter #This file contains main dunction, zipper function, and other related functinos. 

from data import training_output_generator


##################SELECTING DATA##########################
#@Brief: this section creates the lists of pairs that will be fed into the FEATURE functions.
print("Creating global data pairs...")
name_pairs, field_pairs, institution_pairs, shared_ids = raw_data_converter.main()
print("Completed global data pairs...")

##################FEATURE EXTRACTION##########################
#@Brief: This function creates all feature scores and converts the resulting lists into 
#        a format the ML model can use.


def feature_extraction():
    print("\n\nRunning feature_extraction...")
    
    features = []
    name_sim = disambiguation_feature.name_similarity(name_pairs)
    name_disambig = disambiguation_feature.disambiguate_names(name_pairs)
    field_sim = scraped_features.field_similarity_feature(field_pairs)
    field_fuzz_sim = scraped_features.field_similarity_rapidfuzz(field_pairs)
    institution_sim = scraped_features.institution_similarity(institution_pairs)

    features = list(zip(name_sim, name_disambig, field_sim, field_fuzz_sim, institution_sim))
    print("\nCompleted Feature Score Generation")
    return features



##################MACHINE LEARNING MODEL##########################
#@Brief: This section implements the ML model using the features extracted in this file. 
#        I will start with a simple logistic regression model and then explore more complex models 

#Rough draft:

#In terminal: pip install scikit-learn
print("\n\nRunning Model...\n\n")

features = list(feature_extraction())
verified_y= list(training_output_generator.decision_vectors(shared_ids))
verified_y = [int(x) if not pd.isna(x) else 0 for x in verified_y]

#TRAINING the data on 80% of my set
print(f"Total features: {len(features)}")
print(f"Total labels: {len(verified_y)}")

split = int(len(features) * 0.8)
X_train = features[:split]
X_test  = features[split:]

y_train = verified_y[:split]
y_test  = verified_y[split:]



#In my final output, I will run both logistic regression AND gradient boosting and 
# export them both in the output dict csv file. I will make a function for this to contain the code and make it easier to run both models.
def run_models(X_train, y_train, X_test, y_test):
    #Regression model using training data
    print("LOGISTIC REGRESSION")
    model = LogisticRegression(class_weight='balanced') 
    model.fit(X_train, y_train)

    #try Gradient Boosting (doesn't support class_weight, so use this instead)
    print("GRADIENT BOOSTING")
    model_GB = GradientBoostingClassifier(random_state=42, n_estimators=1000, 
                                       learning_rate=0.08,
                                       validation_fraction=0.1,  # 10% of training data used for validation
                                        n_iter_no_change=150,      # Stop if no improvement for 10 iterations
                                        tol=1e-4, ) 
    model_GB.fit(X_train, y_train) #No longer balancing due to high type 2 error: sample_weight=sample_weights
    
    lr_y_pred = model.predict(X_test)
    gb_y_pred = model_GB.predict(X_test)
    lr_proba = model.predict_proba(X_test)
    gb_proba = model_GB.predict_proba(X_test)
    lr_level = np.max(lr_proba, axis = 1)
    gb_level = np.max(gb_proba, axis = 1)


    print("Logistic Regression Classification Report: ", classification_report(y_test, lr_y_pred, labels=[0, 1, 2], target_names=["No Match", "Match", "Inconclusive"], zero_division=0))
    print("Gradient Boosting Classification Report", classification_report(y_test, gb_y_pred, labels=[0, 1, 2], target_names=["No Match", "Match", "Inconclusive"], zero_division=0))

    test_names = [name_pairs[i][1] for i in range(split, len(name_pairs))]
    test_features = [features[i][0:5] for i in range(split, len(features))]
    test_features = [str(feature_tuple) for feature_tuple in test_features]
    output_df = pd.DataFrame(np.column_stack([test_names, lr_y_pred, gb_y_pred, y_test, lr_level, gb_level,  test_features]), index = shared_ids[split:], columns = ["UCPMS Author Name", "LR Predicted", "GB Predicted", "True value", "LR Probability", "GB Probability", "Feature Scores (name, name, field, field, institution)"])
    output_df.to_csv('test_mixed_output_1_may30.csv')
    print("\n Model Output can be found in test_mixed_output.csv")

#------------------MODEL TYPE TESTS---------------------#
#Regression model using training data
'''print("LOGISTIC REGRESSION")
model = LogisticRegression(class_weight='balanced') 
model.fit(X_train, y_train)

#try Gradient Boosting (doesn't support class_weight, so use this instead)
print("GRADIENT BOOSTING")
model_GB = GradientBoostingClassifier(random_state=42, n_estimators=1000, 
                                   learning_rate=0.08,
                                   validation_fraction=0.1,  # 10% of training data used for validation
                                    n_iter_no_change=150,      # Stop if no improvement for 10 iterations
                                    tol=1e-4, ) 
model_GB.fit(X_train, y_train)'''#No longer balancing due to high type 2 error: sample_weight=sample_weights
#---------------------------------------#

'''
#Testing model using the test data 
y_pred = model.predict(X_test)

##print(f"Model Parameters: \nLearningRate: {model.learning_rate}\nNumberofTrees: {model.n_estimators}")

print("\n\nModel results:...")
# Breakdown by class (0, 1, 2)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["No Match", "Match", "Inconclusive"], zero_division=0))

#generating confidence/probability
proba = model.predict_proba(X_test)
print(f"Full Probabilities: {proba[:10]}")  # Print probabilities for the first 10 test samples
level = np.max(proba, axis = 1)

#I need the second name (the UCPMS Name) of each pair for the output dictionary
test_names = [name_pairs[i][1] for i in range(split, len(name_pairs))]

test_features = [features[i][0:5] for i in range(split, len(features))]


#I need each tuple in this list to occupy a row in the output dictionary
#I'll convert each tuple into a string maybe so that it can be easily read in the output dictionary.
test_features = [str(feature_tuple) for feature_tuple in test_features]
print(f"Test Features: {test_features[:10]}")
#test_features_array = np.array(test_features)



##################OUTPUT DICTIONARY##########################
#@Brief: This section generates a dictionary with SCOPUS IDs as keys. The paired
# values will inlcude the model's decision and its prediction scores (Value assigned to each class). 
print("\n\nGenerating Model Output...")

output_df = pd.DataFrame(np.column_stack([y_pred, y_test, level, test_names, test_features]), index = shared_ids[split:], columns = ["Predicted", "True value", "Probability", "UCPMS Author Name", "Feature Scores (name, name, field, field, institution)"])
output_df.to_csv('test_output_9_may27.csv')
print("\n Model Output can be found in test_output.csv")

'''


run_models(X_train, y_train, X_test, y_test)