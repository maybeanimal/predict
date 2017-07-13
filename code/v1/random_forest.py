import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

cd C:\Users\YJ\Desktop
data = pd.read_csv("important_features.csv")
del data["Unnamed: 0"]
X = data.iloc[:,:-1]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42, stratify = y)
rfc = RandomForestClassifier(n_jobs = -1, max_features = 4, n_estimators = 500, random_state = 42, oob_score = True, min_samples_leaf = 10, class_weight={0:0.9, 1:0.1})
param_grid = {
	'max_features':[4,5,6]
}
CV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 10, scoring = 'roc_auc')
CV_rfc.fit(X_train, y_train)
CV_rfc.cv_results_
CV_rfc.best_params_
CV_rfc.best_score_
y_true, y_pred = y_test, CV_rfc.predict(X_test)
print(classification_report(y_true, y_pred))
//获取最重要的特征
importances = forest.feature_importances_

predict = pd.read_csv("08_data.csv")
predict_x = predict[X_train.columns]
predict["label"] = rfc.predict(predict_x)
predict["label"].value_counts()
time = predict[predict["label"] == 0]

import matplotlib.pyplot as plt

//尝试不同的class_weight
for w in [5,10,15,20]:
    print('---Weight of {} for minority class---'.format(w))
	rfc = RandomForestClassifier(n_jobs = -1, max_features = 4, n_estimators = 500, random_state = 42, oob_score = True, min_samples_leaf = 10, class_weight={1:1,0:w})
    rfc.fit(X_train,y_train)
    pred = rfc.predict(X_test)
    PlotConfusionMatrix(y_test, pred, y_test_majority, y_test_minority)