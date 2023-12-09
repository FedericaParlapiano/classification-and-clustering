import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

# Random Forest Classifier
RForest = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(RForest, X_train, y_train, cv=5)
print(f'Random Forest - Cross-Validated Accuracy: {scores.mean()}')

RForest.fit(X_train, y_train)
y_pred_rf = RForest.predict(X_test)
print(f'Random Forest - Test Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Random Forest - Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}')
print(f'Random Forest - Classification Report:\n{classification_report(y_test, y_pred_rf)}')

# Naive Bayes Classifier
naive_bayes = GaussianNB()
scores_nb = cross_val_score(naive_bayes, X_train, y_train, cv=5)
print(f'Naive Bayes - Cross-Validated Accuracy: {scores_nb.mean()}')

naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
print(f'Naive Bayes - Test Accuracy: {accuracy_score(y_test, y_pred_nb)}')
print(f'Naive Bayes - Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nb)}')
print(f'Naive Bayes - Classification Report:\n{classification_report(y_test, y_pred_nb)}')

# Confusion Matrix Plot
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=RForest.classes_, yticklabels=RForest.
            classes_, ax=axes[0])
axes[0].set_title('Random Forest - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=naive_bayes.classes_, yticklabels=naive_bayes
            .classes_, ax=axes[1])
axes[1].set_title('Naive Bayes - Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print(f'Accuracy: {accuracy}')

# Support Vector Machine Classifier
svc = SVC()
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, pred)*100
print(f'Accuracy: {accuracy}')
