# -*- coding: utf-8 -*-
# ---
# title: "Home"
# site: distill::distill_website
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [raw] tags=[] language="R"
# library(reticulate)
# use_condaenv('ethique_env')

# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import seaborn as sns
sns.set_theme(style='darkgrid')

seed = 2021

# %% [markdown]
# # Présentation de la mission du stage et de la chaire Good in Tech

# %% [markdown]
# # Présentation de la base german credit scoring / cas d'usage (prédiction risque de défaut)
#
# Pour étudier les différentes catégories de l'éthique de l'intelligence artificielle définies plus haut nous allons nous utiliser une version nettoyer de la base de données : [German Credit Risk](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
# Cette base est mise à disposition par le professeur Hofmann et contient 1000 entrées avec 22 variables. Chaque entrée représente une personne qui a contracté un emprunt à une banque. Chaque personne est classée par 0 ou 1 si elle a remboursé ou non son crédit.

# %%
credit = pd.read_csv("data/german_credit_prepared.csv", sep=",", engine="python")
credit.head()

# %% [markdown]
#  - default : 
#      - 0 : a remboursé
#      - 1 : a fait défaut
#  - account_check_status : 
#      - ... < 0 DM
#      - ... < 0 DM
#      - 0 <= ... < 200 DM
#      - ... >= 200 DM / salary assignments for at least 1 year
#      - no checking account 
#  - duration_in_month :
#      
#  - credit_history :
#      - no credits taken/ all credits paid back duly
#      - all credits at this bank paid back duly
#      - existing credits paid back duly till now
#      - delay in paying off in the past
#      - critical account/ other credits existing (not at this bank) 
#  - purpose :
#      - car (new)
#      - car (used)
#      - furniture/equipment
#      - radio/television
#      - domestic appliances
#      - repairs
#      - education
#      - (vacation - does not exist?)
#      - retraining
#      - business
#      - others
#  - credit_amount :
#    
#  - savings :
#      - ... < 100 DM
#      - 100 <= ... < 500 DM
#      - 500 <= ... < 1000 DM
#      - .. >= 1000 DM
#      - unknown/ no savings account 
#  - present_emp_since :
#      - unemployed
#      - ... < 1 year
#      - 1 <= ... < 4 years
#      - 4 <= ... < 7 years
#      - .. >= 7 years
#
#  - installment_as_income_perc :
#  
#  - sex :
#      - male
#      - female
#  - personal_status :
#      - single
#      - divorced
#      - married 
#  - other_debtors : 
#      - none
#      - co-applicant
#      - guarantor 
#  - present_res_since :
#  
#  - property :
#      - real estate
#      - if not A121 : building society savings agreement/ life insurance
#      - if not A121/A122 : car or other, not in attribute 6
#      - unknown / no property 
#  - age :
#  
#  - other_installment_plans :
#      - bank
#      - stores
#      - none
#  - housing : 
#      - rent
#      - own
#      - for free
#  - credits_this_bank :
#  
#  - job :
#      - unemployed/ unskilled - non-resident
#      - unskilled - resident
#      - skilled employee / official
#      - management / self-employed / highly qualified employee / officer 
#  - people_under_maintenance : 
#  
#  - telephone :
#      - none
#      - yes
#  - foreign_worker :
#      - yes
#      - no

# %% [markdown]
# # Création du modèle

# %% [markdown]
# En plus d'une base de données il faut un modèle sur lequel appliquer les différentes méthodes. On choisi ici une régression linéaire pour sa simplicité et ses bons résutats sur notre base de données. 
# Il peut être également intéressant de noter que même sur des modèles simple comme une régression logistique une grande partie des problématiques de l'éthique de l'IA sont pertinantes. La discipline est plutôt récente mais elle ne provient pas de la compléxification des modèles statistiques bien que ce phenomène aggrave en générale les problèmes.

# %%
y = credit.default 
X = credit.drop(columns=["default"])

# class the variable between categorical and ordinal 
cat_variables = [col for col in X.columns if credit[col].dtype==object]
ord_variables = [col for col in X.columns if credit[col].dtype==int]

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_variables),
        ('ord', StandardScaler(), ord_variables)
    ])

model = Pipeline(
        [
            ('prepro', preprocess),
            ('logreg', LogisticRegression())
        ]
)

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.20,
                                                                    random_state=seed)

logreg = model.fit(X_train, y_train)


# %% [markdown]
# # Présentation de la performance du modèle

# %%
def mesure_clas(model, X_test, y_test):
    """ Return some mesures on the classifier 'model'
    """
    
    y_test_pred = model.predict(X_test)
    results = classification_report(y_test, 
                                    y_test_pred,
                                    output_dict=True)

    disp = plot_confusion_matrix(model,
                                 X_test,
                                 y_test,
                                 cmap=plt.cm.Blues)
    _ = disp.ax_.set_title('Confusion matrix')
    plt.show()
    
    
    print("{:^12} {:^12} {:^12} {:^12}".format('Sensitivity', 
                                               'Specificity', 
                                               'Precision', 
                                               'Accuracy'))
    print("{:^12.2f} {:^12.2f} {:^12.2f} {:^12.2f}".format(results['0']['recall'], 
                                                           results['1']['recall'], 
                                                           results['1']['precision'],
                                                           results['accuracy']))

# %% [markdown]
# On regarde quatre mesures différentes : 
#  - *Sensitivity* : la fraction de label 0 prédit correctement, ici la fraction de personnes qui ont remboursé qui sont prédit comme capable de rembourser sur la totalité des personnes qui sont prédit comme tel.
#  - *Specificity* : la fraction de label 1 prédit correctement, ici la fraction de personnes qui ont fait défaut qui sont prédit comme incapable de rembourser sur la totalité des personnes prédites comme tel.
#  - *Precision* : la fraction de label prédit comme 1 correct, ici la fraction de personnes prédites comme incapable de rembourser qui fait défaut sur la totalité des personnes ayant fait défaut.
#  - *Accuracy* : la fraction de label correct sur la totalité des personnes, ici la fraction de personne donc le label est correct sur la totalité des personnes.

# %%
mesure_clas(logreg, X_test, y_test)

# %% [markdown]
# On a donc un modèle qui est bon dans les mesures présentés ci-dessus. On va utiliser ce modèle dans les trois notebooks.
