import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Prediction")

iris = load_iris()

model = RandomForestClassifier()
model.fit(iris.data, iris.target)

sl = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sw = st.slider("Sepal Width", 2.0, 5.0, 3.0)
pl = st.slider("Petal Length", 1.0, 7.0, 2.0)
pw = st.slider("Petal Width", 0.1, 3.0, 0.2)

prediction = model.predict([[sl, sw, pl, pw]])

st.write("Prediction:", iris.target_names[prediction[0]])