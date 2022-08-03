# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:48:00 2022

!pip install shap
!pip install shapash
!pip install xgboost
!pip install datatable
!pip install category_encoders 
!pip install -U scikit-learn
!pip install explainerdashboard
!pip install Jinja2==3.0.3

Jinja 3.10 + will give error


@author: manish
"""

import xgboost as xgb
import datatable as dt # data table factory
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from markupsafe import escape
import explainerdashboard as expdb
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard import InlineExplainer
from explainerdashboard.custom import (ImportancesComposite,
                                       IndividualPredictionsComposite,
                                       WhatIfComposite,
                                       ShapDependenceComposite,
                                       ShapInteractionsComposite,
                                       DecisionTreesComposite)

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb



from sklearn.datasets import load_wine

# Read the DataFrame, first using the feature data

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add a target column, and fill it with the target data
df['target'] = data.target

# Show the first five rows
df.head()


# Set up the data for modelling 
y=df['target'].to_frame() # define Y
X=df[df.columns.difference(['target'])] # define X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) # create train and test

# build model - Xgboost
xgb_mod=xgb.XGBClassifier(random_state=101) # build classifier
xgb_mod=xgb_mod.fit(X_train,y_train.values.ravel()) 

# make prediction and check model accuracy 
y_pred = xgb_mod.predict(X_test)

# Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#########################################
##  EXPLAINERDASHBOARD ##############
######################################



# Create the explainer object
explainer = ClassifierExplainer(xgb_mod, X_test, y_test,model_output='logodds')

# Create individual component plants using Inexplainer

ie = InlineExplainer(explainer)

# SHAP overview
ie.shap.overview()

# SHAP interactions
ie.shap.interaction_dependence()

# Model Stats
ie.classifier.model_stats()

# SHAP contribution
ie.shap.contributions_graph()

# SHAP dependence
ie.shap.dependence()



db = ExplainerDashboard(explainer, 
                        title="Wine Data Explainer", # defaults to "Model Explainer"
                        shap_interaction=False, # you can switch off tabs with bools
                        )
db.run(port=8054)