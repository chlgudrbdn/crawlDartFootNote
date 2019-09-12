from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mglearn


# 데이터 변환 과정과 머신러닝을 연결해주는 파이프라인
# 데이터 적재와 분할
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 훈련 데이터의 최솟값, 최댓값을 계산합니다
scaler = MinMaxScaler().fit(X_train)
# 훈련 데이터의 스케일을 조정합니다
X_train_scaled = scaler.transform(X_train)

svm = SVC()
# 스케일 조정된 훈련데이터에 SVM을 학습시킵니다
svm.fit(X_train_scaled, y_train)
# 테스트 데이터의 스케일을 조정하고 점수를 계산합니다
X_test_scaled = scaler.transform(X_test)
print("테스트 점수: {:.2f}".format(svm.score(X_test_scaled, y_test)))


# 데이터 전처리와 매개변수 선택

# 이 코드는 예를 위한 것입니다. 실제로 사용하지 마세요.
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("최상의 교차 검증 정확도: {:.2f}".format(grid.best_score_))
print("테스트 점수: {:.2f}".format(grid.score(X_test_scaled, y_test)))
print("최적의 매개변수: ", grid.best_params_)
# 최상의 교차 검증 정확도: 0.98
# 테스트 점수: 0.97
# 최적의 매개변수:  {'gamma': 1, 'C': 1}
# -> 최솟값과 최댓값을 계산할때는 학습을 위해 훈련세트에 있는 모든 데이터 활용함
# -> 그런 다음 스케일이 조정된 훈련 데이터에서 교차검증 함 -> test 데이터가 분리가 안되어있음

# 교차검증의 분할 방식이 모든 전처리 과정보다 앞서 이뤄져아 한다.
#
# 데이터셋의 모든 정보를 처리하는 과정은 훈련 부분에서만 적용하고 교차검증 반복해야 한다.
#
# CROSS_VAL_SCORE + GRIDE_SEARCHCV -> PIPELINE 사용해야 한다.
# 파이프라인 구축하기


pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)
## 훈련 데이터를 변환하고 마지막으로 변환된 데이터에 SVM 학습시킨다.

Pipeline(memory=None,
     steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svm', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

print("테스트 점수: {:.2f}".format(pipe.score(X_test, y_test)))
# SCALER를 사용하여 테스트 데이터 변환하고 변환된 데이터에 SCORE 측정한다.

# 테스트 점수: 0.95

# 그리드 서치에 파이프라인 적용하기
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("최상의 교차 검증 정확도: {:.2f}".format(grid.best_score_))
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))
print("최적의 매개변수: {}".format(grid.best_params_))
# 최상의 교차 검증 정확도: 0.98
# 테스트 세트 점수: 0.97
# 최적의 매개변수: {'svm__gamma': 1, 'svm__C': 1}


# 앞에 그림과 차이가 있다 잘 살펴보면 다르다는 것을 알 수 있다.

# 정보 누설에 대한 예시



##  X 와 Y에 아무 상관없이 만들었다.

rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100,))
##


# SELECTPERCENTILE 이용해 유용한 특성 선택하고 교차검증하고 RIDGE 회귀 하기

select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)
print("X_selected.shape: {}".format(X_selected.shape))
# X_selected.shape: (100, 500)

# 91프로 확률  말이 안되는 일이 일어 났다.
## 교차검증 밖에서 특성 선택했기때문에 -> 훈련과 테스트 폴드 양쪽에 관련된 특성이 나와서 그렇다

print("교차 검증 정확도 (릿지): {:.2f}".format(np.mean(cross_val_score(Ridge(), X_selected, y, cv=5))))
# 교차 검증 정확도 (릿지): 0.91

## PIPELINE을 이용해서 교차검증하기

## 특성 선택이 파이프라인 안쪽에서 이뤄졌기 때문에 -> 훈련데이터에서 발견된 특성이

## -> 테스트 폴드는 사용하지 안항ㅆ다는 뜻

pipe = Pipeline([("select", SelectPercentile(score_func=f_regression,
                                             percentile=5)),
                 ("ridge", Ridge())])
print("교차 검증 정확도 (파이프라인): {:.2f}".format(
      np.mean(cross_val_score(pipe, X, y, cv=5))))
# 교차 검증 정확도 (파이프라인): -0.25

# 파이프라인 인터페이스
# 어떤 추정기와도 연결해서 사용 가능
# 특성 추출 ,특성 선택 ,스케일 변경, 분류 -네 단계 포함해서 만들 수 있다.
# pipline.steps[0][1]  # 첫번째 추정기 pipline.steps[1][1] 2번째 추정기
# pipline.steps[0][0]  # 첫번 째 단계이름 pipline.steps[1][0] 두번째 단계 이름
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        # 마지막 단계를 빼고 fit과 transform을 반복합니다
        X_transformed = estimator.fit_transform(X_transformed, y)
    # 마지막 단계 fit을 호출합니다
    self.steps[-1][1].fit(X_transformed, y)
    return self

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # 마지막 단계를 빼고 transform을 반복합니다
        X_transformed = step[1].transform(X_transformed)
    # 마지막 단계 predict을 호출합니다
    return self.steps[-1][1].predict(X_transformed)

# make_pipleline을 사용한 파이프라인 생성
# pipe_long 과 pipe_short는 같은 것을 뜻하지만 / pipe_short 는 단계의 이름을 자동으로 만든다.

# 표준적인 방법
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
# 간소화된 방법
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
print("파이프라인 단계:\n{}".format(pipe_short.steps))
# 파이프라인 단계:
# [('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svc', SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False))]

# 이름이 쉽게 지정가능


pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print("파이프라인 단계:\n{}".format(pipe.steps))
# 파이프라인 단계:
# [('standardscaler-1', StandardScaler(copy=True, with_mean=True, with_std=True)), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#
#   svd_solver='auto', tol=0.0, whiten=False)), ('standardscaler-2', StandardScaler(copy=True, with_mean=True, with_std=True))]

# 단계 속성에 접근하기

# 파이프라인 단계중 하나의 속성을 하고 싶을 때 주성분이나 선형계수 등등
# cancer 데이터셋에 앞서 만든 파이프라인을 적용합니다
pipe.fit(cancer.data)
# "pca" 단계의 두 개 주성분을 추출합니다
components = pipe.named_steps["pca"].components_
print("components.shape: {}".format(components.shape))
# components.shape: (2, 30)

# 그리드 서치 안의 파이프라인의 속성에 접근하기


pipe = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
# GridSearchCV(cv=5, error_score='raise',
#        estimator=Pipeline(memory=None,
#      steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('logisticregression', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))]),
#        fit_params=None, iid=True, n_jobs=1,
#        param_grid={'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
#        pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#        scoring=None, verbose=0)
print("최상의 모델:\n{}".format(grid.best_estimator_))
# 최상의 모델:
# Pipeline(memory=None,
#      steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
# ('logisticregression', LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])



# best_estimator_ 는 standardscaler 와 logisticgression 두단계를 거친 estimator



print("로지스틱 회귀 단계:\n{}".format(
      grid.best_estimator_.named_steps["logisticregression"]))
# 로지스틱 회귀 단계:
# LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)

print("로지스틱 회귀 계수:\n{}".format(
      grid.best_estimator_.named_steps["logisticregression"].coef_))
# 로지스틱 회귀 계수:
# [[-0.389 -0.375 -0.376 -0.396 -0.115  0.017 -0.355 -0.39  -0.058  0.209
#   -0.495 -0.004 -0.371 -0.383 -0.045  0.198  0.004 -0.049  0.21   0.224
#   -0.547 -0.525 -0.499 -0.515 -0.393 -0.123 -0.388 -0.417 -0.325 -0.139]]




# 전처리와 모델의 매개변수를 위한 그리드 서치

# 파이프라인 사용 -> 머신러닝 워크플로에 필요한 모든 처리 단게 -> 캡슐화 가능
# 또 다른 장점 -> 회귀와 분류같은 지도학습 출력을 이용해 전처리 매개변수 조정 가능


from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge())



# 다항식 차수가 얼마나 되야할지, 교차항이 필요한지 이상적으로는 -> degree 매개변수 선택

# ridge 의 alpha 매개변수와 함께 degree 탐색 가능

# param_grid 정의 해줘야 가능하다.

param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
# GridSearchCV(cv=5, error_score='raise',
#        estimator=Pipeline(memory=None,
#      steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
# ('polynomialfeatures', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)),
# ('ridge', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#    normalize=False, random_state=None, solver='auto', tol=0.001))]),
#        fit_params=None, iid=True, n_jobs=-1,
#        param_grid={'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'polynomialfeatures__degree': [1, 2, 3]},
#        pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#        scoring=None, verbose=0)
mglearn.tools.heatmap(grid.cv_results_['mean_test_score'].reshape(3, -1),
                      xlabel="ridge__alpha", ylabel="polynomialfeatures__degree",
                      xticklabels=param_grid['ridge__alpha'],
                      yticklabels=param_grid['polynomialfeatures__degree'], vmin=0)


print("최적의 매개변수: {}".format(grid.best_params_))
# 최적의 매개변수: {'ridge__alpha': 10, 'polynomialfeatures__degree': 2}
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))
# 테스트 세트 점수: 0.77
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("다항 특성이 없을 때 점수: {:.2f}".format(grid.score(X_test, y_test)))
# 다항 특성이 없을 때 점수: 0.63

# 모델 선택을 위한 그리드 서치

# classfier 는 randomforest 나 svc 되어야 한다.
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

from sklearn.ensemble import RandomForestClassifier
# 비대칭 그리드 매서드 사용하기 매서드가 다르기 때문에
param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler()],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("최적의 매개변수:\n{}\n".format(grid.best_params_))
print("최상의 교차 검증 점수: {:.2f}".format(grid.best_score_))
print("테스트 세트 점수: {:.2f}".format(grid.score(X_test, y_test)))
# 최적의 매개변수:
# {'classifier': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False), 'classifier__gamma': 0.01,
#  'preprocessing': StandardScaler(copy=True, with_mean=True, with_std=True), 'classifier__C': 10}

# 최상의 교차 검증 점수: 0.99
# 테스트 세트 점수: 0.98
