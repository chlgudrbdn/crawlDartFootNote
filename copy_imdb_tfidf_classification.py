# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import scipy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import tensorflow as tf
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

review = pd.read_csv('../input/IMDB Dataset.csv')
review.head(5)

le = LabelEncoder()
review['target'] = le.fit_transform(review.sentiment)

review.head()
review.drop('sentiment',axis=1,inplace=True)
review.head()

X_arr = review['review'].tolist()

# Removing Stop Words

set(stopwords.words('english'))

STOPWORDS = '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'

cleanr = re.compile(STOPWORDS)

for i in range(0, len(X_arr)):
    X_arr[i] = re.sub(cleanr, "", X_arr[i])

print(X_arr[1])

stopwords_english = set(stopwords.words('english'))

for st in stopwords_english:
    for i in range(0, len(X_arr)):
        if st in X_arr[i]:
            X_arr[i].replace(st, '')

# input_len = 2000
# X = X_arr[0:2000]
print(len(X_arr))
X = X_arr[0:30427]  # 참고로 주석은 (30427, 49154) tfidf 매트릭스가 된다. 이걸 이대로 tf-idf로 만들면 369003 차원이 되므로
# X = X_arr[:2]  # 참고로 주석은 (30427, 49154) tfidf 매트릭스가 된다. 이걸 이대로 tf-idf로 만들면 369003 차원이 되므로
print(len(X))
#Lets use TFIDF to get the numerical values for the text

tfidf = TfidfVectorizer(max_df=0.5, min_df=2,
                                   ngram_range=(1,2),
                                   stop_words='english')
words = []
for lines in X:
    tokenized = nltk.word_tokenize(lines)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
    words.append(" ".join(nouns))


features = tfidf.fit_transform(words)
print(features.shape)
save_dir ='C:/Users/lab515/PycharmProjects/crawlDartFootNote/merged_FnGuide/placebo.npz'
scipy.sparse.save_npz(save_dir, features)
