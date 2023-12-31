import numpy as np
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
import collections
import time

from yellowbrick.model_selection import LearningCurve

# Data
file = '../data/obesity_dataset_clean.csv'
obesity = pd.read_csv(file)
obesity = obesity.iloc[:, 1:]
obesity['Nutritional Status'] \
    = obesity['Nutritional Status'].replace('Insufficient_Weight', 'Insufficient Weight') \
    .replace('Normal_Weight', 'Normal Weight').replace('Overweight_Level_I', 'Overweight Level I') \
    .replace('Overweight_Level_II', 'Overweight Level II').replace('Obesity_Type_I', 'Obesity Type I') \
    .replace('Obesity_Type_II', 'Obesity Type II').replace('Obesity_Type_III', 'Obesity Type III')

output_file_path = '../classification/result.txt'

# Data encoding
lb = LabelEncoder()

obesity['Gender'] = lb.fit_transform(obesity['Gender'])
obesity['Transportation Used'] = lb.fit_transform(obesity['Transportation Used'])
label_names = obesity['Nutritional Status'].unique()
obesity['Nutritional Status'] = lb.fit_transform(obesity['Nutritional Status'])

label_dict = dict(zip(list(obesity['Nutritional Status'].unique()), list(label_names)))
label_dict_ordered = dict(collections.OrderedDict(sorted(label_dict.items())))

# Train-Test split
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


# Models

def decision_tree():
    # Decision Tree Classifier
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    y_pred_dt = decision_tree.predict(X_test)
    save_confusion_matrix(y_test, y_pred_dt, 'DecisionTree.png', 'Decision Tree Classifier')
    dt_dict = dict(zip(decision_tree.feature_names_in_, decision_tree.feature_importances_))
    return y_pred_dt, dt_dict


def random_forest():
    # Random Forest Classifier
    RForest = RandomForestClassifier(n_estimators=100, random_state=42)
    RForest.fit(X_train, y_train)
    y_pred_rf = RForest.predict(X_test)
    save_confusion_matrix(y_test, y_pred_rf, 'RandomForest.png', 'Random Forest Classifier')
    rf_dict = dict(zip(RForest.feature_names_in_, RForest.feature_importances_))
    return y_pred_rf, rf_dict


def SVC_ngs():
    # Support Vector Machine Classifier without GridSearch
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    save_confusion_matrix(y_test, y_pred_svc, 'SVC_ngs.png', 'Support Vector Machine Classifier without GridSearch')
    return y_pred_svc


def SVC():
    # Support Vector Machine Classifier with GRID
    param_grid_svc = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 'auto'],
                      'class_weight': ['balanced', 'None'],
                      'kernel': ['rbf', 'linear', 'poly']}

    grid_svc = GridSearchCV(SVC(), param_grid_svc, refit=True, verbose=3)
    grid_svc.fit(X_train, y_train)
    best_params_svc = str(grid_svc.best_params_)
    y_pred_grid_svc = grid_svc.predict(X_test)
    save_confusion_matrix(y_test, y_pred_grid_svc, 'SVC.png', 'Support Vector Machine Classifier with GridSearch')
    return y_pred_grid_svc, best_params_svc

def logistic_regression_ngs():
    # Logistic Regression Classifier
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred_lr = logistic_regression.predict(X_test)
    save_confusion_matrix(y_test, y_pred_lr, 'LogisticRegression_ngs.png', 'Logistic Regression Classifier')
    return y_pred_lr


def logistic_regression():
    # Logistic Regression Classifier
    logistic_regression = LogisticRegression()

    parameters = [{'penalty': ['l2', 'none']},
                  {'C': [1, 10, 100, 1000]},
                  {'max_iter': [100, 150, 200]},
                  {'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}]
    grid_lr = GridSearchCV(estimator=logistic_regression,
                           param_grid=parameters,
                           scoring='accuracy',
                           verbose=3)
    grid_lr.fit(X_train, y_train)
    y_pred_lr = grid_lr.predict(X_test)
    best_params = str(grid_lr.best_params_)
    save_confusion_matrix(y_test, y_pred_lr, 'LogisticRegression.png', 'Logistic Regression Classifier')

    sizes = np.linspace(0.3, 1.0, 10)
    visualizer = LearningCurve(logistic_regression, scoring='accuracy', train_sizes=sizes)
    visualizer.fit(X_train, y_train)
    visualizer.show()

    return y_pred_lr, best_params


def XGboost():
    # XGBoost Classifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    save_confusion_matrix(y_test, y_pred_xgb, 'XGBoost.png', 'XGBoost Classifier')
    XGboost = dict(zip(xgb.feature_names_in_, xgb.feature_importances_))
    return y_pred_xgb, XGboost


def AdaBoost_ngs():
    # AdaBoost Classifier
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=6)
    adaboost.fit(X_train, y_train)
    y_pred_ab = adaboost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ab, 'AdaBoost_ngs.png', 'AdaBoost Classifier')
    return y_pred_ab


def AdaBoost():
    # AdaBoost Classifier
    parameters = {
        'estimator__criterion': ['gini', 'entropy'],
        'estimator__max_depth': [1, 5, 9],
        'learning_rate': [0.01, 0.1, 0.5, 1],
    }
    dtc = DecisionTreeClassifier()
    adaboost = AdaBoostClassifier(estimator=dtc, n_estimators=100, random_state=6)
    gs = GridSearchCV(adaboost, param_grid=parameters, refit=True, verbose=3)
    gs.fit(X_train, y_train)
    y_pred_ab = gs.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ab, 'AdaBoost.png', 'AdaBoost Classifier')
    ada = dict(zip(gs.best_estimator_.feature_names_in_, gs.best_estimator_.feature_importances_))

    sizes = np.linspace(0.3, 1.0, 10)
    visualizer = LearningCurve(adaboost, scoring='accuracy', train_sizes=sizes)
    visualizer.fit(X_train, y_train)
    visualizer.show()

    return y_pred_ab, gs.best_params_, ada


def GradientBoosting():
    # Gradient Boosting Classifier
    gradient_boost = GradientBoostingClassifier(random_state=42)
    gradient_boost.fit(X_train, y_train)
    y_pred_gb = gradient_boost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_gb, 'GradientBoosting.png', 'Gradient Boosting Classifier')
    gradient = dict(zip(gradient_boost.feature_names_in_, gradient_boost.feature_importances_))
    return y_pred_gb, gradient


# Overfitting and Time complexity
def decision_tree_overfitting():
    train_scores = []
    test_scores = []
    values = [i for i in range(1, 21)]
    # Decision Tree Classifier
    for i in values:
        decision_tree = DecisionTreeClassifier(max_depth=i)
        decision_tree.fit(X_train, y_train)
        train_eval = decision_tree.predict(X_train)
        train_acc = accuracy_score(y_train, train_eval)
        train_scores.append(train_acc)
        y_pred_dt = decision_tree.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_dt)
        test_scores.append(test_acc)
        # save_confusion_matrix(y_test, y_pred_dt, 'DecisionTree.png', 'Decision Tree Classifier')

    plt.plot(values, train_scores, '-o', label='Train')
    plt.plot(values, test_scores, '-o', label='Test')
    plt.legend()
    plt.title('Overfitting in Decision Tree')
    plt.savefig('plot/DToverfitting.png',  bbox_inches='tight')
    plt.show()


def random_forest_timecomplexity():
    train_scores = []
    test_scores = []
    values = [i for i in range(1, 51)]
    times = []
    # Random Forest Classifier
    for i in values:
        RForest = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=i)
        start = time.time()
        RForest.fit(X_train, y_train)
        train_eval = RForest.predict(X_train)
        train_acc = accuracy_score(y_train, train_eval)
        train_scores.append(train_acc)
        y_pred_rf = RForest.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_rf)
        test_scores.append(test_acc)
        end = time.time()
        times.append(end - start)
        # save_confusion_matrix(y_test, y_pred_rf, 'RandomForest.png', 'Random Forest Classifier')
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.plot(values, train_scores, '-o', label='Train')
    plt.plot(values, test_scores, '-o', label='Test')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(values, times, '-o', label='Time taken')
    plt.suptitle('Time Complexity by Decision Tree Depth in Random Forest', )
    plt.savefig('plot/RFTimeComplexity.png',  bbox_inches='tight')
    plt.show()


# save confusion matrix


y_pred_dt, dt_dict = decision_tree()
y_pred_rf, rf_dict = random_forest()
y_pred_grid_svc, optimal_svc = SVC()
y_pred_svc = SVC_ngs()
y_pred_lr, optimal_lr = logistic_regression()
y_pred_lr_ngs = logistic_regression_ngs()
y_pred_xgb, xg_dict = XGboost()
y_pred_ab, optimal_ab, ab_dict = AdaBoost()
y_pred_ab_ngs = AdaBoost_ngs()
y_pred_gb, gb_dict = GradientBoosting()


# heatmap feature importances

d = {'Random Forest': pd.Series(rf_dict.values(),
                                index=rf_dict.keys()),
     'Decision Tree': pd.Series(dt_dict.values(),
                                index=dt_dict.keys()),
     'AdaBoost': pd.Series(ab_dict.values(),
                            index=ab_dict.keys()),
     'Gradient Boosting': pd.Series(gb_dict.values(),
                                    index=gb_dict.keys()),
     'XGBoost': pd.Series(xg_dict.values(),
                          index=xg_dict.keys())
     }

feature_importance = pd.DataFrame(d)
sns.heatmap(feature_importance, cmap="crest")
plt.title('Feature importance by model')
plt.savefig('plots/Heatmap', bbox_inches='tight')
plt.show()

# comparisons
fig, ax = plt.subplots(figsize=(11, 3))
ax = sns.lineplot(x=y_test, y=y_pred_xgb,
                  label='XGBoost')
ax1 = sns.lineplot(x=y_test, y=y_pred_gb,
                   label='GradientBoosting')
ax2 = sns.lineplot(x=y_test, y=y_pred_ab,
                   label='AdaBoost')

ax.set_xlabel('y_test', color='g')
ax.set_ylabel('y_pred', color='g')
plt.title('Comparison between models')
plt.savefig('plots/Comparison', bbox_inches='tight')
plt.show()


# Plot accuracy
def plot_accuracy_bar():
    models = ['Decision Tree', 'Random Forest', 'Support Vector Machine GS', 'Logistic Regression', 'XGBoost',
              'AbaBoost', 'Gradient Boosting']
    accuracies = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf),
                  accuracy_score(y_test, y_pred_grid_svc), accuracy_score(y_test, y_pred_lr),
                  accuracy_score(y_test, y_pred_xgb), accuracy_score(y_test, y_pred_ab),
                  accuracy_score(y_test, y_pred_gb)]

    color = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f']
    plt.bar(models, accuracies, color=color)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.title('Classifier Accuracy Comparison')
    plt.ylim([0, 1])  # Assuming accuracy values are between 0 and 1
    plt.savefig('plots/Accuracy', bbox_inches='tight')
    plt.show()


plot_accuracy_bar()


# Writing result in output file
with open(output_file_path, 'w') as f:
    f.write('\n<------------------------------------- Test Accuracy ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier: {accuracy_score(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier: {accuracy_score(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier without GridSearch: {accuracy_score(y_test, y_pred_svc)}\n')
    f.write(f'Support Vector Machine Classifier with GridSearch: {accuracy_score(y_test, y_pred_grid_svc)}\n')
    f.write(f'Logistic Regression without GridSearch: {accuracy_score(y_test, y_pred_lr_ngs)}\n')
    f.write(f'Logistic Regression with GridSearch: {accuracy_score(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier: {accuracy_score(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier without GridSearch: {accuracy_score(y_test, y_pred_ab_ngs)}\n')
    f.write(f'AdaBoost Classifier  with GridSearch: {accuracy_score(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier: {accuracy_score(y_test, y_pred_gb)}\n')

    f.write('\n<------------------------------------- Classification Report ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier:\n{classification_report(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier:\n{classification_report(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier without GridSearch:\n{classification_report(y_test, y_pred_svc)}\n')
    f.write(f'Support Vector Machine Classifier with GridSearch:\n{classification_report(y_test, y_pred_grid_svc)}\n')
    f.write(f'Logistic Regression without GridSearch: {classification_report(y_test, y_pred_lr_ngs)}\n')
    f.write(f'Logistic Regression with GridSearch:\n{classification_report(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier:\n{classification_report(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier without GridSearch: {classification_report(y_test, y_pred_ab_ngs)}\n')
    f.write(f'AdaBoost Classifier with GridSearch:\n{classification_report(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier:\n{classification_report(y_test, y_pred_gb)}')

    f.write('\n<------------------------------ Optimal Parameters for Grid Search ------------------------------->\n\n')
    f.write(f'Support Vector Machine Classifier with GridSearch: {optimal_svc}\n')
    f.write(f'Logistic Regression Classifier with GridSearch: {optimal_lr}\n')
    f.write(f'AdaBoost Classifier with GridSearch: {optimal_ab}\n')
