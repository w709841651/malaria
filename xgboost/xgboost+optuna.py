

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,StratifiedKFold
import matplotlib.ticker as mticker
from matplotlib.pylab import rcParams
import pickle
import pydotplus
from sklearn.tree import export_graphviz
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import accuracy_score,recall_score,f1_score
from numpy import loadtxt
from xgboost import plot_importance
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import shap
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import cross_val_score

shap.initjs()

trial_dir = datetime.today().strftime('%Y%m%d_%H%M%S')
if not os.path.exists(trial_dir):
    os.makedirs(trial_dir)


data = pd.read_csv(
    '2classtest2_0118.csv',
)

target = data[['class01']]
features = data.drop('class01', axis=1) 
features = features.drop('Tail.Length', axis=1)
features = features.drop('Wing.Length', axis=1)
features = features.drop('Prevalence', axis=1) 
features.head()


x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=0)
x_train["PHYLO"].astype("category")
x_train.shape
x_train.ndim
print(features)


def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3,10),
        'lambda': trial.suggest_uniform('lambda', 0,6),
        'alpha': trial.suggest_uniform('alpha', 0,6),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
        'gamma': trial.suggest_uniform('gamma', 0,10),
        'learning_rate': trial.suggest_loguniform('learning_rate',0.01,0.15),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 0.9),
        'subsample': trial.suggest_uniform('subsample', 0.5, 0.9),
        'n_estimators': trial.suggest_int('n_estimators', 500,4000),
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 1, 2),
        'nthread': 32
    } 

##Tuning again with more narrower values to get the sweet spot
# def objective(trial):
#     param = {
#         'max_depth': trial.suggest_categorical('max_depth', [3,7,8,9]),
#         'lambda': trial.suggest_uniform('lambda', 4,6),
#         'alpha': trial.suggest_uniform('alpha', 0,1),
#         'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
#         'gamma': trial.suggest_uniform('gamma', 7.5,10),
#         'learning_rate': trial.suggest_loguniform('learning_rate',0.009,0.025),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 0.37),
#         'subsample': trial.suggest_uniform('subsample', 0.85, 0.9),
#         'n_estimators': trial.suggest_int('n_estimators', 500,2200),
#         'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 2,2),
#         'random_state':trial.suggest_int('random_state',1,40),
#         'nthread': 32
#     } 
    model = xgb.XGBClassifier(**param)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)],  verbose=False)
    preds = model.predict(x_test)

    stratifiedkf=StratifiedKFold(n_splits=5)
    scorer = make_scorer(f1_score)
    f1 = cross_val_score(model, x_train, y_train, cv=stratifiedkf, scoring=scorer).mean()
    return f1





import sys
from optuna.samplers import TPESampler
# file = open('output0120.txt','w')
# sys.stdout = file
study1 = optuna.create_study(direction='maximize',sampler = TPESampler())
n_trials=500
study1.optimize(objective, n_trials=n_trials, show_progress_bar = True)
print('Number of finished trials:', len(study1.trials))
print("------------------------------------------------")
print('Best trial:', study1.best_trial.params)
print("------------------------------------------------")
print(study1.trials_dataframe())
print("------------------------------------------------")


fig1 = optuna.visualization.plot_optimization_history(study1)
fig1.write_image(trial_dir + '/fig1.pdf')

fig2 = optuna.visualization.plot_slice(study1)
fig2.write_image(trial_dir + '/fig2.pdf')


params_7=study1.best_params
print(params_7)

eval_set = [(x_train, y_train), (x_test, y_test)]
#final model
model=xgb.XGBClassifier(**params_7
                        
    )
#early_stopping_rounds=20,
model.fit(x_train,y_train,eval_set=eval_set, verbose=True,eval_metric=["error", "logloss"])

from matplotlib import pyplot

results = model.evals_result()
print(results)
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
# pyplot.show()
pyplot.savefig(trial_dir+'/log_loss.pdf')

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
clf_roc_auc = roc_auc_score(y_test, model.predict(x_test))
clf_fpr, clf_tpr, clf_thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(clf_fpr, clf_tpr, label='Decision Tree (area = %0.2f)' % clf_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
# plt.show()
plt.savefig(trial_dir+'/roc.pdf')
print (" AUC = %2.2f" % clf_roc_auc)
Y_pred = model.predict(x_test)
print(accuracy_score(y_test, Y_pred))
print(classification_report(y_test, Y_pred))

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
# pyplot.show()
pyplot.savefig(trial_dir+'/classification_error.pdf')
