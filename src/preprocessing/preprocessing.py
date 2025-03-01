#!/usr/bin/env python
# coding: utf-8
# @author: konain

import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from alive_progress import alive_bar

def load_dataset():
    """
    This function allows to load the datasets
    """
    print("Loading data")
    path = r"C:\Jean Eudes Folder\_Projects\Financial_Fraud_Detection\src\dataset\dataset.xlsx"
    data = pd.read_excel(path)

    with alive_bar(100) as bar:
        for i in range(100):
            time.sleep(0.1)
            bar()

    return data

def feature_extraction(_options="financial_factor"):
    """
    This function allows to extract features
    """
    output = {
        "dt" : [],
        "label": []
    }

    # charger le dataset
    data = load_dataset()

    # suppression des colonnes doubles
    data.drop(columns=['Type de plainte reue', 'Pays','Province/tat', 
                       'Catgories thmatiques sur la fraude et la cybercriminalit',
                       'Mthode de sollicitation','Genre', 'Langue de correspondance', 
                       'Type de plainte'], inplace=True)
    
    # extraction de la colonne cible
    fraud = data['Fraud and Cybercrime Thematic Categories']
    _fraud = fraud.drop_duplicates(ignore_index=True)
    _fraud_arr = _fraud.to_numpy()

    for ind, val in enumerate(_fraud_arr):
        fraud.replace(to_replace=val, value=ind, inplace=True)

    # ajout dans le dictionnaire
    output['label'] = fraud

    # options
    if _options == "financial_factor":
        dollarLoss = data['Dollar Loss']

        # ajout dans le dictionnaire
        output['dt'] = dollarLoss

        return output

    elif _options == "method_used_for_fraud":
        method = data['Solicitation Method']
        _method = method.drop_duplicates(ignore_index=True)
        _method_arr = _method.to_numpy()

        for ind, val in enumerate(_method_arr):
            method.replace(to_replace=val, value=ind, inplace=True)

        # ajout dans le dictionnaire
        output['dt'] = method

        return output

def preprocessing(_options="financial_factor"):
    """
    This function allows to a carry out a data cleaning and data preprocessing before using in the model
    """
    output = {
        "train" : {},
        "val" : {},
        "test" : {}
    }

    # extraction des caractéristiques cibles
    features = feature_extraction(_options)

    feature = features['dt'].to_numpy()
    feature_reshaped = feature.reshape(-1, 1)
    label = features['label'].to_numpy()

    # normalisation des données
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_reshaped)
    
    # séparation des données training, validation and testing dataset
    feature_train, x, label_train, y = train_test_split(feature_scaled, label, test_size=0.2, random_state=15)
    feature_val, feature_test, label_val, label_test = train_test_split(x, y, test_size=0.5, random_state=42)

    print("Séparation des données : ")
    print(f"\tTraining dataset size : {feature_train.size}")
    print(f"\tValidation dataset size : {feature_val.size}")
    print(f"\tTesting dataset size : {feature_test.size}")

    # Visulaisation
    train_size = feature_train.size
    val_size = feature_val.size
    test_size = feature_test.size
    total_size = train_size + val_size + test_size

    train_percent = (train_size / total_size) * 100
    val_percent = (val_size / total_size) * 100
    test_percent = (test_size / total_size) * 100

    plt.figure(figsize=(8, 6))
    labels = ['Training set', 'Validation set', 'Test set']
    sizes = [train_percent, val_percent, test_percent]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    explode = (0.08, 0.05, 0.05)  # Highlight "Training set"

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.f%%',
            shadow=True, startangle=90)
    plt.title("Split of samples between Training, Validation and Test set", fontsize=12)
    # plt.show()

    # ajout dans le dictionnaire
    output['train']['feature'] = feature_train
    output['train']['label'] = label_train
    output['val']['feature'] = feature_val
    output['val']['label'] = label_val
    output['test']['feature'] = feature_test
    output['test']['label'] = label_test

    return output



if __name__ == "__main__":
    out = preprocessing(_options="method_used_for_fraud")
    print(out)