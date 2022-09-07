import os
import sys
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Binarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from itertools import cycle
from sklearn.metrics import precision_recall_curve

data=pd.read_excel('finaldata.xlsx', engine='openpyxl')

data_shuffled=data.sample(frac=1).reset_index(drop=True)
data_shuffled.head(10)

train_data=data_shuffled.iloc[:,2:-1]
target_data=data_shuffled.iloc[:,-1]

train_data['head_x']/=train_data['width']
train_data['neck_x']/=train_data['width']
train_data['rshoulder_x']/=train_data['width']
train_data['relbow_x']/=train_data['width']
train_data['rwrist_x']/=train_data['width']
train_data['lshoulder_x']/=train_data['width']
train_data['lelbow_x']/=train_data['width']
train_data['lwrist_x']/=train_data['width']
train_data['rhip_x']/=train_data['width']
train_data['rknee_x']/=train_data['width']
train_data['rankle_x']/=train_data['width']
train_data['lhip_x']/=train_data['width']
train_data['lknee_x']/=train_data['width']
train_data['lankle_x']/=train_data['width']
train_data['chest_x']/=train_data['width']

train_data['head_y']/=train_data['height']
train_data['nexk_y']/=train_data['height']
train_data['rshoulder_y']/=train_data['height']
train_data['relbow_y']/=train_data['height']
train_data['rwrist_y']/=train_data['height']
train_data['lshoulder_y']/=train_data['height']
train_data['lelbow_y']/=train_data['height']
train_data['lwrist_y']/=train_data['height']
train_data['rhip_y']/=train_data['height']
train_data['rknee_y']/=train_data['height']
train_data['rankle_y']/=train_data['height']
train_data['lhip_y']/=train_data['height']
train_data['lknee_y']/=train_data['height']
train_data['lankle_y']/=train_data['height']
train_data['chest_y']/=train_data['height']

train_data=train_data.iloc[:,2:]
train_data.describe()
target_data.describe()
def print_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index = ['ALABESQUE','PASSE','PLIE'], 
                         columns = ['ALABESQUE','PASSE','PLIE'])
    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('RandomForest')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()
    plt.clf()

def print_auc_roc(model, x_test):
    global y_test
    #Calculate the y_score
    y_score = model.predict_proba(x_test)
    #Binarize the output
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    n_classes = y_test_bin.shape[1]

    sum=0
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr_sum=[]
    tpr_sum=[]


    colors = ['blue', 'red', 'green']
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        fpr_sum.append(fpr[i])
        tpr_sum.append(tpr[i])
        #plt.plot(fpr[i], tpr[i], color=colors[i], lw=2)
        print('AUC for Class {}: {}'.format(i, auc(fpr[i], tpr[i])))
        sum+=auc(fpr[i], tpr[i])
        
    print("average sum:", sum/3)
    fpr_avg=[]
    tpr_avg=[]
    for i in range(max(fpr_sum[0].shape[0],fpr_sum[1].shape[0],fpr_sum[2].shape[0])):
        num=0
        sum2=0
        if i< fpr_sum[0].shape[0]:
            num+=1
            sum2+=fpr_sum[0][i]
        if i< fpr_sum[1].shape[0]:
            num+=1
            sum2+=fpr_sum[1][i]
        if i< fpr_sum[2].shape[0]:
            num+=1
            sum2+=fpr_sum[2][i]

        fpr_avg.append(sum2/num)

    for i in range(max(tpr_sum[0].shape[0],tpr_sum[1].shape[0],tpr_sum[2].shape[0])):
        num=0
        sum2=0
        if i< tpr_sum[0].shape[0]:
            num+=1
            sum2+=tpr_sum[0][i]
        if i< tpr_sum[1].shape[0]:
            num+=1
            sum2+=tpr_sum[1][i]
        if i< tpr_sum[2].shape[0]:
            num+=1
            sum2+=tpr_sum[2][i]

        tpr_avg.append(sum2/num)
        
    return fpr_avg, tpr_avg
    #plt.plot(fpr_avg, tpr_avg, color='blue', lw=2)

    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic Curves')
    #plt.show()

def print_pr_curve(model, x_test):
    #Calculate the y_score
    y_score = model.predict_proba(x_test)
    #Binarize the output
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    n_classes = y_test_bin.shape[1]

    sum=0
    pr = dict()
    rc = dict()
    #roc_auc = dict()
    pr_sum=[]
    rc_sum=[]

    pr_avg=[]
    rc_avg=[]


    colors = ['blue', 'red', 'green']
    for i in range(n_classes):
        pr[i], rc[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        base_rate=y_score[:, i].mean()
        pr_sum.append(pr[i])
        rc_sum.append(rc[i])
        #plt.plot(rc[i], pr[i], color=colors[i], lw=2)

    for i in range(max(pr_sum[0].shape[0],pr_sum[1].shape[0],pr_sum[2].shape[0])):
        num=0
        sum2=0
        if i< pr_sum[0].shape[0]:
            num+=1
            sum2+=pr_sum[0][i]
        if i< pr_sum[1].shape[0]:
            num+=1
            sum2+=pr_sum[1][i]
        if i< pr_sum[2].shape[0]:
            num+=1
            sum2+=pr_sum[2][i]

        pr_avg.append(sum2/num)

    for i in range(max(rc_sum[0].shape[0],rc_sum[1].shape[0],rc_sum[2].shape[0])):
        num=0
        sum2=0
        if i< rc_sum[0].shape[0]:
            num+=1
            sum2+=rc_sum[0][i]
        if i< rc_sum[1].shape[0]:
            num+=1
            sum2+=rc_sum[1][i]
        if i< rc_sum[2].shape[0]:
            num+=1
            sum2+=rc_sum[2][i]

        rc_avg.append(sum2/num)
        
    return pr_avg, rc_avg


    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.title('Precision-Recall Curve')
    #plt.show()

def print_feature_importances(model, train_data):
    importances=model.feature_importances_
    indices=np.argsort(importances)[::-1]

    print('Feature ranking:')

    for f in range(train_data.shape[1]):
        print('{}. feature {} ({:.3f})'.format(f+1, train_data.columns[indices][f], importances[indices[f]]))
    plt.figure(figsize=(10,8))
    plt.title('feature importances')
    plt.bar(range(train_data.shape[1]), importances[indices],
            color='r', align='center')
    for i,v in enumerate(range(train_data.shape[1])):
        plt.text(v, importances[indices][i],round(importances[indices][i],2), fontsize=9, color='black', horizontalalignment='center', verticalalignment='bottom')
    plt.xticks(range(train_data.shape[1]), train_data.columns[indices], rotation=45)
    plt.xlim([-1,train_data.shape[1]])
    plt.show()


x_train, x_test, y_train, y_test=train_test_split(train_data, target_data, test_size=0.2,  stratify=target_data)

#########################
# RandomForest Classifier
#########################
x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y']]
x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y','lknee_x','rknee_x','rshoulder_x','neck_x','lshoulder_x']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y','lknee_x','rknee_x','rshoulder_x','neck_x','lshoulder_x']]
x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y','rknee_x','rshoulder_x','lelbow_x','lknee_x','lshoulder_x','neck_x','rhip_y','lhip_y','head_x','rwrist_x']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y','rknee_x','rshoulder_x','lelbow_x','lknee_x','lshoulder_x','neck_x','rhip_y','lhip_y','head_x','rwrist_x']]

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(x_train_new, y_train)
y_pred=model.predict(x_test_new)
print('precision:', precision_score(y_test, y_pred, average='macro'))
print('recall:', recall_score(y_test, y_pred, average='macro'))
print('f1:', f1_score(y_test, y_pred, average='macro'))

print_confusion_matrix(y_test,y_pred)
a1,b1=print_auc_roc(model, x_test_new)
c1,d1=print_pr_curve(model, x_test_new)

#########################
# XGBClassifier
#########################

x_train_new=x_train[['head_x', 'lelbow_x','rankle_x','rshoulder_y','lankle_y']]
x_test_new=x_test[['head_x', 'lelbow_x','rankle_x','rshoulder_y','lankle_y']]
x_train_new=x_train[['head_x', 'lelbow_x','rankle_x','rshoulder_y','lankle_y','relbow_x','lankle_x','lhip_y','rankle_y','lshoulder_x']]
x_test_new=x_test[['head_x', 'lelbow_x','rankle_x','rshoulder_y','lankle_y','relbow_x','lankle_x','lhip_y','rankle_y','lshoulder_x']]
x_train_new=x_train[['relbow_x', 'lelbow_x','lhip_y','rankle_x','lankle_x','head_x','lknee_x','lshoulder_x','lankle_y','rankle_y','rshoulder_y','rhip_y','head_y','rknee_x','neck_x']]
x_test_new=x_test[['relbow_x', 'lelbow_x','lhip_y','rankle_x','lankle_x','head_x','lknee_x','lshoulder_x','lankle_y','rankle_y','rshoulder_y','rhip_y','head_y','rknee_x','neck_x']]

from xgboost import XGBClassifier

model=XGBClassifier()
model.fit(x_train_new, y_train)
y_pred=model.predict(x_test_new)
print('precision:', precision_score(y_test, y_pred, average='macro'))
print('recall:', recall_score(y_test, y_pred, average='macro'))
print('f1:', f1_score(y_test, y_pred, average='macro'))

print_confusion_matrix(y_test,y_pred)
a2,b2=print_auc_roc(model, x_test_new)
c2,d2=print_pr_curve(model, x_test_new)

#########################
# Gradient Boosting Classifier
#########################

x_train_new=x_train[['lankle_x', 'rankle_y','rankle_x','lankle_y','lelbow_x']]
x_test_new=x_test[['lankle_x', 'rankle_y','rankle_x','lankle_y','lelbow_x']]
x_train_new=x_train[['lankle_x', 'rankle_y','rankle_x','lankle_y','lelbow_x','relbow_x','lknee_y','lhip_y','rknee_x','rhip_y']]
x_test_new=x_test[['lankle_x', 'rankle_y','rankle_x','lankle_y','lelbow_x','relbow_x','lknee_y','lhip_y','rknee_x','rhip_y']]
x_train_new=x_train[['lankle_x', 'rankle_x','rankle_y','lankle_y','relbow_x','lelbow_x','lhip_y','rhip_y','lknee_x','rknee_x','rshoulder_y','lshoulder_x','head_y','head_x','rknee_y']]
x_test_new=x_test[['lankle_x', 'rankle_x','rankle_y','lankle_y','relbow_x','lelbow_x','lhip_y','rhip_y','lknee_x','rknee_x','rshoulder_y','lshoulder_x','head_y','head_x','rknee_y']]

from sklearn.ensemble import GradientBoostingClassifier

model=GradientBoostingClassifier()
model.fit(x_train_new, y_train)
y_pred=model.predict(x_test_new)
print('precision:', precision_score(y_test, y_pred, average='macro'))
print('recall:', recall_score(y_test, y_pred, average='macro'))
print('f1:', f1_score(y_test, y_pred, average='macro'))

print_confusion_matrix(y_test,y_pred)
a3,b3=print_auc_roc(model, x_test_new)
c3,d3=print_pr_curve(model, x_test_new)

#########################
# AdaGradient Boosting Classifier
#########################

x_train_new=x_train[['lankle_y', 'rankle_y','rankle_x','lankle_x','rknee_x']]
x_test_new=x_test[['lankle_y', 'rankle_y','rankle_x','lankle_x','rknee_x']]
x_train_new=x_train[['lankle_y', 'rankle_y','rankle_x','lankle_x','rknee_x','relbow_x','lknee_x','lshoulder_y','head_y','rhip_y']]
x_test_new=x_test[['lankle_y', 'rankle_y','rankle_x','lankle_x','rknee_x','relbow_x','lknee_x','lshoulder_y','head_y','rhip_y']]
x_train_new=x_train[['lankle_y', 'rankle_y','lankle_x','head_y','lknee_x','rankle_x','relbow_x','lknee_y','rknee_x','rhip_y','rshoulder_y','lhip_y','lhip_x','lshoulder_y','rwrist_x']]
x_test_new=x_test[['lankle_y', 'rankle_y','lankle_x','head_y','lknee_x','rankle_x','relbow_x','lknee_y','rknee_x','rhip_y','rshoulder_y','lhip_y','lhip_x','lshoulder_y','rwrist_x']]

from sklearn.ensemble import AdaBoostClassifier

model=AdaBoostClassifier()
model.fit(x_train_new, y_train)
y_pred=model.predict(x_test_new)
print('precision:', precision_score(y_test, y_pred, average='macro'))
print('recall:', recall_score(y_test, y_pred, average='macro'))
print('f1:', f1_score(y_test, y_pred, average='macro'))

print_confusion_matrix(y_test,y_pred)
a4,b4=print_auc_roc(model, x_test_new)
c4,d4=print_pr_curve(model, x_test_new)

#########################
# Bagging Classifier
#########################

x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x']]
x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x','rshoulder_x','lknee_x','rknee_x','lshoulder_x','lhip_y']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x','rshoulder_x','lknee_x','rknee_x','lshoulder_x','lhip_y']]
x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y','rknee_x','rshoulder_x','lelbow_x','lknee_x','lshoulder_x','neck_x','rhip_y','lhip_y','head_x','rwrist_x']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','relbow_x','lankle_y','rknee_x','rshoulder_x','lelbow_x','lknee_x','lshoulder_x','neck_x','rhip_y','lhip_y','head_x','rwrist_x']]

from sklearn.ensemble import BaggingClassifier

model=BaggingClassifier()
model.fit(x_train_new, y_train)
y_pred=model.predict(x_test_new)
print('precision:', precision_score(y_test, y_pred, average='macro'))
print('recall:', recall_score(y_test, y_pred, average='macro'))
print('f1:', f1_score(y_test, y_pred, average='macro'))

print_confusion_matrix(y_test,y_pred)
a5,b5=print_auc_roc(model, x_test_new)
c5,d5=print_pr_curve(model, x_test_new)

#########################
# Extra Tree Classifier
#########################

x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x']]
x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x','rknee_x','lknee_x','rwrist_x','rshoulder_x','rhip_y']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','lankle_y','relbow_x','rknee_x','lknee_x','rwrist_x','rshoulder_x','rhip_y']]
x_train_new=x_train[['rankle_x', 'lankle_x','rankle_y','lankle_y','rknee_x','relbow_x','lknee_x','rwrist_x','rshoulder_x','rhip_y','lhip_y','lshoulder_x','neck_x','rknee_y','head_x']]
x_test_new=x_test[['rankle_x', 'lankle_x','rankle_y','lankle_y','rknee_x','relbow_x','lknee_x','rwrist_x','rshoulder_x','rhip_y','lhip_y','lshoulder_x','neck_x','rknee_y','head_x']]

from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()
model.fit(x_train_new, y_train)
y_pred=model.predict(x_test_new)
print('precision:', precision_score(y_test, y_pred, average='macro'))
print('recall:', recall_score(y_test, y_pred, average='macro'))
print('f1:', f1_score(y_test, y_pred, average='macro'))

print_confusion_matrix(y_test,y_pred)
a6,b6=print_auc_roc(model, x_test_new)
c6,d6=print_pr_curve(model, x_test_new)

#### 각종 그래프 ####
plt.plot(a1, b1, color='red', lw=2,label='RandomForest')
plt.plot(a2, b2, color='orange', lw=2, label='XGB')
plt.plot(a3, b3, color='pink', lw=2,label='Gradient Boosting')
plt.plot(a4, b4, color='blue', lw=2, label='Ada Gradient Boosting')
plt.plot(a5, b5, color='green', lw=2,label='Bagging')
plt.plot(a6, b6, color='purple', lw=2, label='Extra Trees')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend()
plt.show()
plt.clf()

plt.plot(d1, c1, color='red', lw=2,label='RandomForest')
plt.plot(d2, c2, color='orange', lw=2, label='XGB')
plt.plot(d3, c3, color='pink', lw=2,label='Gradient Boosting')
plt.plot(d4, c4, color='blue', lw=2, label='Ada Gradient Boosting')
plt.plot(d5, c5, color='green', lw=2,label='Bagging')
plt.plot(d6, c6, color='purple', lw=2, label='Extra Trees')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
plt.clf()

cols = train_data.columns.tolist()
fig, axes = plt.subplots(nrows = 6, ncols = 5, figsize = (30,30))
for i, col_name in enumerate(cols):
  row = i // 5
  col = i % 5
  sns.distplot(train_data[col_name], ax = axes[row][col])
  
cols = train_data.columns.tolist()

valuelist=[0.0005, 0.005, 0.01, 0.99, 0.995, 0.9995, 1, 0.9, 0.1]

for j in range(len(valuelist)):
  print('----------------',valuelist[j],'----------------')
  for i, col_name in enumerate(cols):
    print(i,'. ',col_name,' : ',train_data[col_name].quantile(q=valuelist[j], interpolation='nearest'))
  

train_data.describe()

'''
### total figure ###

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


  #Plotting the confusion matrix
plt.figure(figsize=(12,7))
gs=gridspec.GridSpec(nrows=2, ncols=3,height_ratios=[1,1], width_ratios=[1,1,1])
ax0=plt.subplot(gs[0])
ax0.plot()
sns.heatmap(cm_df1, annot=True)
plt.title('Randomforest')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

ax1=plt.subplot(gs[1])
ax1.plot()
sns.heatmap(cm_df2, annot=True)
plt.title('XGB Classifier')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

ax2=plt.subplot(gs[2])
ax2.plot()
sns.heatmap(cm_df3, annot=True)
plt.title('Gradient Boosting Classifier')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

ax3=plt.subplot(gs[3])
ax3.plot()
sns.heatmap(cm_df4, annot=True)
plt.title('Ada Gradient Boosting Classifier')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

ax4=plt.subplot(gs[4])
ax4.plot()
sns.heatmap(cm_df5, annot=True)
plt.title('Bagging Classifier')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

ax5=plt.subplot(gs[5])
ax5.plot()
sns.heatmap(cm_df6, annot=True)
plt.title('Extra Trees Classifier')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

plt.tight_layout()
plt.show()
plt.clf()

plt.plot(a1, b1, color='red', lw=2,label='RandomForest')
plt.plot(a2, b2, color='orange', lw=2, label='XGB')
plt.plot(a3, b3, color='pink', lw=2,label='Gradient Boosting')
plt.plot(a4, b4, color='blue', lw=2, label='Ada Gradient Boosting')
plt.plot(a5, b5, color='green', lw=2,label='Bagging')
plt.plot(a6, b6, color='purple', lw=2, label='Extra Trees')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend()
plt.show()
plt.clf()

plt.plot(d1, c1, color='red', lw=2,label='RandomForest')
plt.plot(d2, c2, color='orange', lw=2, label='XGB')
plt.plot(d3, c3, color='pink', lw=2,label='Gradient Boosting')
plt.plot(d4, c4, color='blue', lw=2, label='Ada Gradient Boosting')
plt.plot(d5, c5, color='green', lw=2,label='Bagging')
plt.plot(d6, c6, color='purple', lw=2, label='Extra Trees')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
plt.clf()
'''


### PCA 적용 전처리 ###
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
printcipalComponents=pca.fit_transform(train_data)
principalDf=pd.DataFrame(data=printcipalComponents, columns=['f1','f2','f3'])

pca=PCA(n_components=5)
printcipalComponents=pca.fit_transform(train_data)
principalDf=pd.DataFrame(data=printcipalComponents, columns=['f1','f2','f3','f4','f5'])

pca=PCA(n_components=10)
printcipalComponents=pca.fit_transform(train_data)
principalDf=pd.DataFrame(data=printcipalComponents, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])

pca=PCA(n_components=15)
printcipalComponents=pca.fit_transform(train_data)
principalDf=pd.DataFrame(data=printcipalComponents, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15'])

principalDf.head()
print(sum(pca.explained_variance_ratio_))
x_train, x_test, y_train, y_test=train_test_split(principalDf, target_data, test_size=0.2,  stratify=target_data)

### ICA 적용 전처리 ###
from sklearn.decomposition import FastICA
ica=FastICA(n_components=3)
icacomponent=ica.fit_transform(train_data)
ipccomponentDf=pd.DataFrame(data=icacomponent, columns=['f1','f2','f3'])

ica=FastICA(n_components=5)
icacomponent=ica.fit_transform(train_data)
ipccomponentDf=pd.DataFrame(data=icacomponent, columns=['f1','f2','f3','f4','f5'])

ica=FastICA(n_components=10)
icacomponent=ica.fit_transform(train_data)
ipccomponentDf=pd.DataFrame(data=icacomponent, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])

ica=FastICA(n_components=15)
icacomponent=ica.fit_transform(train_data)
ipccomponentDf=pd.DataFrame(data=icacomponent, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15'])

x_train, x_test, y_train, y_test=train_test_split(ipccomponentDf, target_data, test_size=0.2,  stratify=target_data)

### Univariate selection 적용 전처리 ###
from sklearn.feature_selection import SelectKBest, f_classif

selectK=SelectKBest(score_func=f_classif, k=3)
univariate=selectK.fit_transform(train_data, target_data)
univariateDf=pd.DataFrame(data=univariate)

selectK=SelectKBest(score_func=f_classif, k=5)
univariate=selectK.fit_transform(train_data, target_data)
univariateDf=pd.DataFrame(data=univariate)

selectK=SelectKBest(score_func=f_classif, k=10)
univariate=selectK.fit_transform(train_data, target_data)
univariateDf=pd.DataFrame(data=univariate)

selectK=SelectKBest(score_func=f_classif, k=15)
univariate=selectK.fit_transform(train_data, target_data)
univariateDf=pd.DataFrame(data=univariate)

x_train, x_test, y_train, y_test=train_test_split(univariateDf, target_data, test_size=0.2,  stratify=target_data)

pd.set_option('display.max_columns', 100)
univariateDf.describe()
train_data.describe()



