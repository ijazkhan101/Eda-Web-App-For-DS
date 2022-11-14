import streamlit as st               
import seaborn as sns
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# make contrainers 

header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("titanic App:")
    st.text("In this project we will work on titanic App")
   

with header:
    st.title("Data_sets:")
    st.text("In this project we will work on KAshti App")
     # import data
    df=sns.load_dataset('titanic')
    df=df.dropna()
    st.write(df.head(10))
    st.subheader("Bar chat for Sex")
    st.bar_chart(df['sex'].value_counts())
    st.subheader("Line chart for sex")
    st.line_chart(df['sex'].value_counts())
    st.subheader("Line chart for Age")
    st.bar_chart(df['age'].sample(10))


with header:
    st.title("features:")
    st.text("In this project we will work on KAshti App")
    st.markdown('1. ** Feature 1: ** this will tell us ')
    st.markdown('1. ** Feature 1: ** this will tell us ')

with header:
    st.title("model_training:")
    st.text("In this project we will work on KAshti App")

    # making colums

    input, display = st.columns(2)

    # 1st colums will have selections points

    max_depth = input.slider("How many people do you know?", min_value=10 , max_value=100, value=20)

    # n_estimators 
    n_estimators = input.selectbox("how manu tree should be there in a PR? ",options=[50,100,200,300,'No limit'])

    # add list of features
    input.write(df.columns)
    # input features from user
    user_input = input.text_input('Which Features you want just wirte')


#machine Learning Model
model = RandomForestRegressor(max_depth= max_depth,n_estimators=n_estimators)

# condition for nan
if n_estimators =="No limit":
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model =RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# define X and y

X= df[[user_input]]
y = df[['fare']]

#fit model
model.fit(X,y)
pred = model.predict(y)

# Disply  metrics

display.subheader("Mean absolute error of the model is : ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean Squre error of the model is : ")
display.write(mean_squared_error(y,pred))
display.subheader("R squre score of the model is : ")
display.write(r2_score(y,pred))