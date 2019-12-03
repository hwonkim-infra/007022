import pandas as pd

#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)


df = pd.read_csv('wdbc.data', header=None)

df.head()

df.shape        # (569, 32)

from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values        # 30개 특성을 np 배열 X 에 할당
y = df.loc[:, 1].values         
le = LabelEncoder()
y = le.fit_transform(y)        # class label 을 원본 문자열에서 정수로 변환 Benign: class 1, Malignant: class 0
le.classes_                       # array(['B', 'M'], dtype=object)

le.transform(['M', 'B'])        # array([1, 0], dtype=int64)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)   # train / test 데이터셋 분류 8:2


# Pipeline 변환 및 추정기 연결


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline      

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(solver='liblinear', random_state=1))
    # make_pipeline: 여러 변환기(fit)와 추정기(predict)를 연결. 데이터는 중간 fit과 transform method 거쳐 estimator 도달. 변환된 훈련 셋을 이용해 학습
    # 스케일 조정(fit / transform), 차원 축소, 학습알고리즘, 예측 모델 수행기를 감싼 wrapper 역할. 
    # 파이프라인의 마지막 요소는 estimator 가 되어야 함. 
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('테스트 정확도: %.3f' % pipe_lr.score(X_test, y_test))


# K겹 교차 검증을 사용한 모델 성능 평가: 중복되지 않게 train 데이터셋을 k 개의 폴드로 랜덤 분할. 
# 경험적으로 10겹 (k=10) 교차 검증이 추천됨. k가 커지면 검증알고리즘 실행시간이 늘어나고 분산이 높아짐.


import numpy as np
from sklearn.model_selection import StratifiedKFold
    # (특히 클래스 비율이 동등하지 않을 때) 더 나은 편향과 분산 추정.

kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
    # n_splits: 폴드 갯수 지정. 
scores = []
for k, (train, test) in enumerate(kfold):                           # kfold 반복자를 사용하여 train 인덱스를 로지스틱 회귀 파이프라인 훈련에 사용
    pipe_lr.fit(X_train[train], y_train[train])                         # pipe_lr 파이프라인으로 각 반복에서 샘플 스케일을 조절
    score = pipe_lr.score(X_train[test], y_train[test])            # test index: 모델의 정확도 점수 계산.
    scores.append(score)                                             # 이 점수를 리스트에 모아 추정 정확도의 평균과 표준편차 계산
    print('폴드: %2d, 클래스 분포: %s, 정확도: %.3f' % (k+1, np.bincount(y_train[train]), score))
    
print('\nCV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# cross_val_score: 회귀에는 


from sklearn.model_selection import cross_val_score
    # 각 폴드의 평가를 복수 코어에 분산 연산 
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print('CV 정확도 점수: %s' % scores)
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))



from sklearn.model_selection import cross_validate
    # 교차검증에 여러 측정 지표 사용할 수 있는 함수. 각 폴드에서 훈련 테스트 시간 반환. 

scores = cross_validate(estimator=pipe_lr, X=X_train, y=y_train, scoring=['accuracy'], cv=10, n_jobs=-1, return_train_score=False)
print('CV 정확도 점수: %s' % scores['test_accuracy'])
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))


# 학습 곡선과 검증 곡선을 사용한 알고리즘 디버깅


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
    # 계층별 k겹 교차 검증 사용하여 검증 정확도 계산
    
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', penalty='l2', random_state=1))
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    # 일정 간격으로 훈련 세트 비율 10개 설정. cv = 10겹 교차 검증. 

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
    # 반환된 훈련 / 교차검증 점수로부터 세트 크기별로 평균 정확도 계산하여 그래프
    
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    # 그래프에 평균 정확도 표준편차 이용 추정 분산 표기
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()


# 검증 곡선으로 과대적합과 과소적합 조사

from sklearn.model_selection import validation_curve
    # 샘플 크기 함수 > 모델 파라미터 값 함수로 그림
    # validation_curve: 최적화할 파라미터 이름과 범위, 그리고 성능 기준을
    # param_name, param_range, scoring 인수로 받아 파라미터 범위 모든 경우에 대해 성능 기준 계산

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, param_name='logisticregression__C', param_range=param_range, cv=10)
    # LogisticRegression 분류기의 규제 매개변수 C 지정: C를 줄이면 규제 강도가 높아지고 과소적합. C를 늘리면 과대적합 양상
    
    
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.00])
plt.tight_layout()
plt.show()

# 그리드 서치를 사용한 하이퍼파라미터 튜닝: 훈련 데이터에서 학습되지 않고 별도로 최적화되는 학습알고리즘 파라미터
# 자동으로 복수개의 내부 모형을 생성하고 이를 모두 실행시켜서 최적 파라미터를 탐색
# 리스트로 지정된 여러 하이퍼파라미터 값 전체를 모두 조사

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    # SVM을 위한 파이프라인을 훈련하고 튜닝
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # 튜닝하려는 매개변수를 딕셔너리 리스트로 지정.
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']}, {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]
    # 선형 SVM 은 규제 매개변수 C만 튜닝
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
    # n_jobs : 내부적으로 멀티 프로세스를 사용하여 그리드서치 수행. CPU 코어의 수가 충분하다면 n_jobs를 늘릴 수록 속도가 증가. (default 1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# 독립적 테스트 세트를 이용하여 최고 모델 성능 추정:  
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('테스트 정확도: %.3f' % clf.score(X_test, y_test))


# 중첩 교차 검증을 사용한 알고리즘 선택: 바깥 k겹 검증 루프가 데이터를 train / test 로 분할. 안쪽 루프가 검증으로 모델 선택
# 모델이 선택되면 테스트 폴드 사용하여 모델 성능 평가


gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2)

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[20]:


from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring='accuracy', cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[21]:


from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


# In[22]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()


# In[23]:


from sklearn.metrics import precision_score, recall_score, f1_score

print('정밀도: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('재현율: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))


# In[24]:


from sklearn.metrics import make_scorer

scorer = make_scorer(f1_score, pos_label=0)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# In[25]:


from sklearn.metrics import roc_curve, auc
from scipy import interp

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(solver='liblinear', penalty='l2', random_state=1, C=100.0))

X_train2 = X_train[:, [4, 14]]
    
cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[27]:


pre_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='micro')
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100


# In[28]:


from sklearn.utils import resample
print('샘플링하기 전의 클래스 1의 샘플 개수:', X_imb[y_imb == 1].shape[0])

X_upsampled, y_upsampled = resample(X_imb[y_imb == 1], y_imb[y_imb == 1], replace=True, n_samples=X_imb[y_imb == 0].shape[0], random_state=123)
print('샘플링한 후의 클래스 1의 샘플 개수:', X_upsampled.shape[0])


# In[29]:


X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))
y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100


# In[ ]:
