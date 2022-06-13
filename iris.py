# 基本ライブラリー
import numpy as np
import pandas as pd
 
# データセット
## データの読み込み
from sklearn.datasets import load_iris
iris = load_iris()
 
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.loc[df['target'] == 0, 'target'] = "setosa"
df.loc[df['target'] == 1, 'target'] = "versicolor"
df.loc[df['target'] == 2, 'target'] = "virginica"

# 予測モデル構築
X = iris.data[:, [0, 2]] 
y = iris.target

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)


# 予測モデル構築
X = iris.data[:, [0, 2]] 
y = iris.target
 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)

# app
import streamlit as st

# サイドバー（インプット部分）
st.sidebar.header('Input Features')
sepalValue = st.sidebar.slider('sepal length(cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length(cm)', min_value=0.0, max_value=10.0, step=0.1)

# メインパネル
st.title('Iris Classifier')
st.write('### 入力された値')

# インプット部分を１行のデータフレームにする
value_df = pd.DataFrame([], columns=['data', 'sepal length(cm)', 'petal length(cm)'])
record = pd.Series(['data', sepalValue, petalValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index('data', inplace=True)
# 随時確かめるとよい
st.write(value_df)

# 予測値のデータフレーム
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'],index=['probability'])
st.write('### 予測値')
st.write(pred_df)
name = pred_df.idxmax(axis=1).tolist()
st.write("## Result")
st.write('このアイリスはきっと',str(name[0]),"です!")