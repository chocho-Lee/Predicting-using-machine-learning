#!/usr/bin/env python
# coding: utf-8

# # 기계학습 기법을 통한 신경계 질환 환자들의 중환자실 재원기간 예측 및 분석
# * mysql로 예측 모델에 사용할 데이터셋 구축 후 python으로 예측 모델 적용
# * 사용된 데이터셋(raw data): MIMIC-IV (미국 의료기관의 중환자실 의료정보시스템에서 추출된 데이터베이스)
# * 기계학습 기법 중 Support vector machine, Random forest, LogisticRegression, RidgeClassifier 적용
# * 그 중 Random forest의 hyperparameter를 조정하였을 때 가장 좋은 성능을 보임 (n_estimators = 100, max_depth = 20)
# * Random forest 모델에 영향을 미치는 feature들을 확인하기 위해 특성 영향도 파악 

# ### Dataset 구축 (using mySQL)

# 1. raw data에서 신경계 질환 환자의 입원번호만 추출한다. (524,520명 중 52,445명이 연구대상에 포함)
# 2. raw data에서 재원일수를 계산한다. (퇴원일자에서 입원일자 차감)
# 3. 신경계 질환 환자들의 재원일수를 예측하기 위한 feature를 추출한다. (주의: 테이블의 전체 row수가 52,445줄로 유지되도록 함)
# 4. 3.에서 추출한 25개의 feature 중 인코딩이 필요한 항목은 sql로 데이터 포맷을 변환한다.
# 5. 예측에 활용될 예측 데이터셋 구축이 완료된 후 DB connection을 하여 데이터를 불러온다.

# ### DB Connection

# In[1]:


import pandas as pd
import numpy as np
import pymysql
conn = pymysql.connect(host="", user="", password="", db="", charset="utf8")
curs = conn.cursor()
curs.execute("select * from neuro")
rows = curs.fetchall()
ACDEFG = pd.DataFrame(list(rows)).fillna(0)
conn.close()

ACDEFG.columns = ['hadm_id', 'subject_id','LOS_group','admission_type', 'admission_location','discharge_location', 
                        'Stroke', 'Migraine', 'Epilepsy', 'Alzheimer','Parkinson', 'Sclerosis', 'Tumor',
                        'Infection', 'flag_cnt', 'priority_cnt', 'cnt_micro', 'category_test', 'org_yn', 'ab_yn', 'interpretation_R', 'max_drg_severity', 'max_drg_mortality', 'max_emar_seq','cnt_procedures','cnt_med','pharmacy_status','sliding_scale']
ACDEFG


# ### One-hot encoding

# In[11]:


ACDEFG_encoded = pd.get_dummies(ACDEFG, columns=['admission_type','admission_location','discharge_location',
                                       'Stroke','Migraine','Epilepsy','Alzheimer',
                                       'Parkinson','Sclerosis','Tumor','Infection','category_test','org_yn','ab_yn','interpretation_R'])


# ### X(feature), y(result value) 정의 및 train/test data 분류 

# In[3]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale

X = ACDEFG_encoded.iloc[:,3:]
y = np.ravel(ACDEFG_encoded.iloc[:,[2]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)


# ### StandardScale

# In[4]:


from sklearn.preprocessing import StandardScaler, scale

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ### SVC 실행 

# In[5]:


svc = SVC()
svc.fit(X_train, y_train)


# ### Support Vector Machine

# In[6]:


print("학습 데이터 점수: {}".format(svc.score(X_train, y_train)))
print("평가 데이터 점수: {}".format(svc.score(X_test, y_test)))

linear_svc = SVC(kernel='linear')
linear_svc.fit(X_train, y_train)
print("Linear SVC 학습 데이터 점수: {}".format(linear_svc.score(X_train, y_train)))
print("Linear SVC 평가 데이터 점수: {}".format(linear_svc.score(X_test, y_test)))

polynomial_svc = SVC(kernel='poly')
polynomial_svc.fit(X_train, y_train)
print("Polynomial SVC 학습 데이터 점수: {}".format(polynomial_svc.score(X_train, y_train)))
print("Polynomial SVC 평가 데이터 점수: {}".format(polynomial_svc.score(X_test, y_test)))

rbf_svc = SVC(kernel='rbf')
rbf_svc.fit(X_train, y_train)
print("rbf SVC 학습 데이터 점수: {}".format(rbf_svc.score(X_train, y_train)))
print("rbf SVC 평가 데이터 점수: {}".format(rbf_svc.score(X_test, y_test)))


# ### LogisticRegression

# In[12]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=0).fit(X_train, y_train)
#clf.predict(X)
#clf.predict_proba(X)
print("Logistic Regression 학습 데이터 점수: {}".format(LR.score(X_train, y_train)))
print("Logistic Regression 평가 데이터 점수: {}".format(LR.score(X_test, y_test)))


# ### RidgeClassifier

# In[13]:


from sklearn.linear_model import RidgeClassifier

RC = RidgeClassifier().fit(X_train, y_train)
RC.score(X_test, y_test)

print("Ridge Classifier 학습 데이터 점수: {}".format(RC.score(X_train, y_train)))
print("Ridge Classifier 평가 데이터 점수: {}".format(RC.score(X_test, y_test)))


# ### Random Forest

# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf_1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf_1.fit(X_train, y_train)
RF_1 = clf_1.predict(X_test)

clf_2 = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
clf_2.fit(X_train, y_train)
RF_2 = clf_2.predict(X_test)

clf_3 = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=0)
clf_3.fit(X_train, y_train)
RF_3 = clf_3.predict(X_test)

clf_4 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
clf_4.fit(X_train, y_train)
RF_4 = clf_4.predict(X_test)

clf_5 = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=0)
clf_5.fit(X_train, y_train)
RF_5 = clf_5.predict(X_test)

clf_6 = RandomForestClassifier(n_estimators=50, max_depth=30, random_state=0)
clf_6.fit(X_train, y_train)
RF_6 = clf_6.predict(X_test)


print("Random Forest: {}".format(accuracy_score(y_test, RF_1)))
print("Random Forest: {}".format(accuracy_score(y_test, RF_2)))
print("Random Forest: {}".format(accuracy_score(y_test, RF_3)))
print("Random Forest: {}".format(accuracy_score(y_test, RF_4)))
print("Random Forest: {}".format(accuracy_score(y_test, RF_5)))
print("Random Forest: {}".format(accuracy_score(y_test, RF_6)))


# ### Confusion matrix & Classification report 출력

# In[8]:


from sklearn.metrics import confusion_matrix, classification_report


matrix = confusion_matrix(y_test, RF_2)
print(matrix)


print(classification_report(y_test, RF_2))


# ### heatmap 출력

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(matrix/np.sum(matrix), fmt = '.2%', annot=True, cmap='viridis')
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()


# ### Random Forest의 feature importances 확인 

# In[15]:


feature_names = [f"feature {i}" for i in range(X.shape[1])]

importances = clf_2.feature_importances_
clf_2_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
clf_2_importances.plot.bar()
#ax.set_title("Feature importances using MDI")
#ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

