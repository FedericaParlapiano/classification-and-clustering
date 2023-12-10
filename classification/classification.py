import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
import collections


file = '../data/obesity_dataset_clean.csv'
obesity = pd.read_csv(file)
obesity = obesity.iloc[:, 1:]
obesity['Nutritional Status'] \
    = obesity['Nutritional Status'].replace('Insufficient_Weight', 'Insufficient Weight') \
    .replace('Normal_Weight', 'Normal Weight').replace('Overweight_Level_I', 'Overweight Level I') \
    .replace('Overweight_Level_II', 'Overweight Level II').replace('Obesity_Type_I', 'Obesity Type I') \
    .replace('Obesity_Type_II', 'Obesity Type II').replace('Obesity_Type_III', 'Obesity Type III')

output_file_path = '../classification/result.txt'

lb = LabelEncoder()

obesity['Gender'] = lb.fit_transform(obesity['Gender'])
obesity['Transportation Used'] = lb.fit_transform(obesity['Transportation Used'])
label_names = obesity['Nutritional Status'].unique()
obesity['Nutritional Status'] = lb.fit_transform(obesity['Nutritional Status'])


label_dict = dict(zip(list(obesity['Nutritional Status'].unique()), list(label_names)))
label_dict_ordered = dict(collections.OrderedDict(sorted(label_dict.items())))


X = obesity.drop(['Nutritional Status'], axis=1)
Y = obesity['Nutritional Status']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=13)


def save_confusion_matrix(y_test, y_pred, file_name, title):
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_dict_ordered.values()),
                yticklabels=list(label_dict_ordered.values()))
    plt.title(title + ' Confusion Matrix')
    plt.savefig('confusion_matrix/' + file_name, bbox_inches='tight')
    plt.show()



# scaler = preprocessing.StandardScaler().fit(X_train)

def decision_tree():
    # Decision Tree Classifier
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    y_pred_dt = decision_tree.predict(X_test)
    save_confusion_matrix(y_test, y_pred_dt, 'DecisionTree.png', 'Decision Tree Classifier')
    return y_pred_dt

def random_forest():
    # Random Forest Classifier
    RForest = RandomForestClassifier(n_estimators=100, random_state=42)
    RForest.fit(X_train, y_train)
    y_pred_rf = RForest.predict(X_test)
    save_confusion_matrix(y_test, y_pred_rf, 'RandomForest.png', 'Random Forest Classifier')
    return y_pred_rf

def SVC_nogs():
    # Support Vector Machine Classifier without GridSearch
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    save_confusion_matrix(y_test, y_pred_svc, 'SVC.png', 'Support Vector Machine Classifier without GridSearch')
    return y_pred_svc

def SVC_gs():
    # Support Vector Machine Classifier with GRID
    param_grid_svc = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto'],
                  'class_weight': ['balanced', 'None'],
                  'kernel': ['rbf', 'linear', 'poly']}
    grid_svc = GridSearchCV(SVC(), param_grid_svc, refit=True, verbose=3)
    grid_svc.fit(X_train, y_train)
    best_params_svc = grid_svc.best_params_
    y_pred_grid_svc = grid_svc.predict(X_test)
    save_confusion_matrix(y_test, y_pred_grid_svc, 'SVC_gs.png', 'Support Vector Machine Classifier with GridSearch')
    return y_pred_grid_svc

def logistic_regression():
    # Logistic Regression Classifier
    logistic_regression = LogisticRegression(max_iter=100, random_state=42)
    logistic_regression.fit(X_train, y_train)
    y_pred_lr = logistic_regression.predict(X_test)
    save_confusion_matrix(y_test, y_pred_lr, 'LogisticRegression.png', 'Logistic Regression Classifier')
    return y_pred_lr


def XGboost():
    # XGBoost Classifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    save_confusion_matrix(y_test, y_pred_xgb, 'XGBoost.png', 'XGBoost Classifier')
    return y_pred_xgb

def AdaBoost():
    # AdaBoost Classifier
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
    adaboost.fit(X_train, y_train)
    y_pred_ab = adaboost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ab, 'AdaBoost.png', 'AdaBoost Classifier')
    return y_pred_ab

def GradientBoosting():
    # Gradient Boosting Classifier
    gradient_boost = GradientBoostingClassifier(random_state=42)
    gradient_boost.fit(X_train, y_train)
    y_pred_gb = gradient_boost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_gb, 'GradientBoosting.png', 'Gradient Boosting Classifier')
    return y_pred_gb

y_pred_dt = decision_tree()
y_pred_rf = random_forest()
y_pred_svc = SVC_nogs()
y_pred_grid_svc = SVC_gs()
y_pred_lr = logistic_regression()
y_pred_xgb = XGboost()
y_pred_ab = AdaBoost()
y_pred_gb = GradientBoosting()



# Writing result in output file
with open(output_file_path, 'w') as f:
    f.write('\n<------------------------------------- Test Accuracy ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier: {accuracy_score(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier: {accuracy_score(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier without GridSearch: {accuracy_score(y_test, y_pred_svc)}\n')
    f.write(f'Support Vector Machine Classifier with GridSearch: {accuracy_score(y_test, y_pred_grid_svc)}\n')
    f.write(f'Logistic Regression: {accuracy_score(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier: {accuracy_score(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier: {accuracy_score(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier: {accuracy_score(y_test, y_pred_gb)}\n')

    f.write('\n<------------------------------------- Classification Report ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier:\n{classification_report(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier:\n{classification_report(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier without GridSearch:\n{classification_report(y_test, y_pred_svc)}\n')
    f.write(f'Support Vector Machine Classifier with GridSearch:\n{classification_report(y_test, y_pred_grid_svc)}\n')
    f.write(f'Logistic Regression:\n{classification_report(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier:\n{classification_report(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier:\n{classification_report(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier:\n{classification_report(y_test, y_pred_gb)}')
