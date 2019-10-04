from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
import csv
import numpy as np

# scikit learn으로 다양한 종류의 모델을 만들어 보자.
clf_1 = Pipeline([('vect', CountVectorizer()),
                  ('clf', MultinomialNB())])
clf_2 = Pipeline([('vect', HashingVectorizer(non_negative=True)),
                  ('clf', MultinomialNB())])
clf_3 = Pipeline([('vect', TfidfVectorizer()),
                  ('clf', MultinomialNB())])

# 입력 데이터를 훈련 데이터와 테스트 데이터로 분해하자.

org_data = []
org_label = []
class1_count = 0
class2_count = 0
with open('tmp.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[1] == 'CLASS1':
            org_label.append(1)
            class1_count += 1
        elif row[1] == 'CLASS2':
            org_label.append(0)
            class2_count += 1
        else:
            continue
        org_data.append(row[0])

SPLIT_PERC = 0.75
split_size = int(len(org_data) * SPLIT_PERC)
X_train = org_data[:split_size]
X_test = org_data[split_size:]
y_train = org_label[:split_size]
y_test = org_label[split_size:]
ndarr_label = np.array(org_label)

class1_rate = class1_count / float(len(org_label))
class2_rate = class2_count / float(len(org_label))

print('class1 : %f' % class1_rate)
print('class2 : %f' % class2_rate)
print('test size : %d' % len(X_test))

# 훈련 모델을 만들자.
from sklearn import metrics


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


train_and_evaluate(clf_1, X_train, X_test, y_train, y_test)


# CLASS1과 CLASS2를 구분하자.
def predict(X_train):
    y_pred = clf_1.predict(X_train)
    return y_pred


for data in datas:
    predict(data)
# https://yujuwon.tistory.com/entry/SCIKITLEARN-naive-bayes%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%B4%EC%84%9C-%EB%AC%B8%EC%84%9C-%EB%B6%84%EB%A5%98-%ED%95%98%EA%B8%B0