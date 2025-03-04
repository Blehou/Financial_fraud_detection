#!/usr/bin/env python
# coding: utf-8
# @author: konain

import matplotlib.pyplot as plt
from preprocessing.preprocessing import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# pre-processing
data = preprocessing(_options="method_used_for_fraud")

X_train = data['train']['feature']
y_train = data['train']['label']

X_val = data['val']['feature']
y_val = data['val']['label']

X_test = data['test']['feature']
y_test = data['test']['label']

# Random Forest Classifier
print("Entrainement du modèle")
rfc_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=16, n_jobs=-1)

# Entraîner le modèle sur les données d'entraînement
rfc_model.fit(X_train, y_train)
print("Fin entrainement")

# Evaluer le modèle
print("Evaluation du modèle")
y_pred_rfc = rfc_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_rfc)
print(f"Accuracy val score : {accuracy}")

# Confusion matrix
label_pred = rfc_model.predict(X_test)
plt.figure(figsize=(6,5))
conf_matrix = confusion_matrix(y_test, label_pred)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, label_pred, digits=4))