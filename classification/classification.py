import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
import warnings

'''
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''

file = '../data/obesity_dataset_clean.csv'
obesity = pd.read_csv(file)
obesity = obesity.iloc[:, 1:]

lb = LabelEncoder()

obesity['Gender'] = lb.fit_transform(obesity['Gender'])
obesity['Transportation Used'] = lb.fit_transform(obesity['Transportation Used'])
obesity['Nutritional Status'] = lb.fit_transform(obesity['Nutritional Status'])

X = obesity.drop(['Nutritional Status'], axis=1)
Y = obesity['Nutritional Status']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=13)

scaler = preprocessing.StandardScaler().fit(X_train)

# Random Forest Classifier
RForest = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rf = cross_val_score(RForest, X_train, y_train, cv=5)
print(f'Random Forest - Cross-Validated Accuracy: {scores_rf.mean()}')

RForest.fit(X_train, y_train)
y_pred_rf = RForest.predict(X_test)
print(f'Random Forest - Test Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Random Forest - Classification Report:\n{classification_report(y_test, y_pred_rf)}')

# Naive Bayes Classifier
naive_bayes = GaussianNB()
scores_nb = cross_val_score(naive_bayes, X_train, y_train, cv=5)
print(f'Naive Bayes - Cross-Validated Accuracy: {scores_nb.mean()}')

naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
print(f'Naive Bayes - Test Accuracy: {accuracy_score(y_test, y_pred_nb)}')
print(f'Naive Bayes - Classification Report:\n{classification_report(y_test, y_pred_nb)}')

# XGBoost Classifier
xgb = XGBClassifier()
scores_xgb = cross_val_score(xgb, X_train, y_train, cv=5)
print(f'XGBoost - Cross-Validated Accuracy: {scores_xgb.mean()}')

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f'XGBoost - Test Accuracy: {accuracy_score(y_test, y_pred_xgb)}')
print(f'XGBoost - Classification Report:\n{classification_report(y_test, y_pred_xgb)}')

# Support Vector Machine Classifier
svc = SVC()
scores_svc = cross_val_score(svc, X_train, y_train, cv=5)
print(f'SVC - Cross-Validated Accuracy: {scores_svc.mean()}')

svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(f'SVC - Test Accuracy: {accuracy_score(y_test, y_pred_svc)}')
print(f'SVC - Classification Report:\n{classification_report(y_test, y_pred_svc)}')

# Logistic Regression Classifier
logistic_regression = LogisticRegression(max_iter=100, random_state=42)
scores_lr = cross_val_score(logistic_regression, X_train, y_train, cv=5)
print(f'Logistic Regression - Cross-Validated Accuracy: {scores_lr.mean()}')

logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
print(f'Logistic Regression - Test Accuracy: {accuracy_score(y_test, y_pred_lr)}')
print(f'Logistic Regression - Classification Report:\n{classification_report(y_test, y_pred_lr)}')

# AdaBoost Classifier
adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
scores_ab = cross_val_score(adaboost, X_train, y_train, cv=5)
print(f'AdaBoost Classifier - Cross-Validated Accuracy: {scores_ab.mean()}')

adaboost.fit(X_train, y_train)
y_pred_ab = adaboost.predict(X_test)
print(f'AdaBoost Classifier - Test Accuracy: {accuracy_score(y_test, y_pred_ab)}')
print(f'AdaBoost Classifier - Classification Report:\n{classification_report(y_test, y_pred_ab)}')

# Decision Tree Classifier
# Gradient Boosting Classifier
# Linear Discriminant Analysis

# Confusion Matrix Plot
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
conf_matrix_svc = confusion_matrix(y_test, y_pred_svc)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=RForest.classes_,
            yticklabels=RForest.classes_, ax=axes[0, 0])
axes[0, 0].set_title('Random Forest - Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=naive_bayes.classes_,
            yticklabels=naive_bayes.classes_, ax=axes[0, 1])
axes[0, 1].set_title('Naive Bayes - Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=xgb.classes_, yticklabels=xgb.classes_,
            ax=axes[1, 0])
axes[1, 0].set_title('XGBoost - Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svc, annot=True, fmt='d', cmap='Blues', xticklabels=svc.classes_, yticklabels=svc.classes_,
            ax=axes[1, 1])
axes[1, 1].set_title('Support Vector Machine - Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
