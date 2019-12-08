#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()



import matplotlib.pyplot as plt
import seaborn as sns


# 산점도 데이터 행렬: 다섯 개의 열만 행렬에 포함

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()


# corrcoef 함수를 이용한 피어슨 r 도출. 특성 사이의 선형의존성 측정.
import numpy as np

cm = np.corrcoef(df[cols].values.T)
# sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)

plt.tight_layout()
plt.show()

# 선형 회귀 모델 훈련에는 타겟 변수와 상관관계가 높은 특성을 채택 (MEDV - LSTAT, RM - MEDV 등)


# 경사 하강법으로 회귀 모델 파라미터 도출. 아달린 경사하강에서 단위 계단 함수 제거
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
                # 비용함수: 제곱 오차합.
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


# In[9]:


X = df[['RM']].values
y = df['MEDV'].values


# In[10]:


# RM 변수를 특성으로 사용하여 MEDV 예측 모델 훈련
# 경사 하강법 알고리즘이 잘 수련하도록 특성을 표줂롸 전처리

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
    # y_std 계산. 배열에 새로운 차원 추가 newaxis. flatten()을 이용 1차원 배열로 복귀

lr = LinearRegressionGD()
lr.fit(X_std, y_std)


# In[11]:


# 에포크 함수로 비용 그래프: 비용함수의 최솟값으로 수렴하는지 확인
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()



# 선형회귀모델과 훈련 데이터 비교: helper 함수 형성하여 산점도와 회귀 직선 비교

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

plt.show()


# 표준화 처리된 경우 y 절편은 항상 0

print('기울기: %.3f' % lr.w_[1])
print('절편: %.3f' % lr.w_[0])




# y = 3 (방 3개 초과)를 잘라내지 말고 예측 출력 값을 원본스케일로 복원:
num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("$1,000 단위 가격: %.3f" % sc_y.inverse_transform(price_std))


# 회귀 모델 가중치 추정: 표준화되지 않은 RM 과 MEDV 변수에 훈련

from sklearn.linear_model import LinearRegression


slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('기울기: %.3f' % slr.coef_[0])
print('절편: %.3f' % slr.intercept_)



# RM에 대한 MDEV 그래프를 그려 비교
lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

plt.show()



