import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

# Setting the web page's width
st.set_page_config(layout="wide")


# loading data
df = pd.read_csv("dataset.csv")

# Removing unwanted/repeating columns
df = df.drop(columns=['Unnamed: 0','pclass','sibsp','parch','embarked','who','adult_male','deck','survived','alone'])
# Removing Missing Values using forward filling ffill
df["age"] = df["age"].fillna(df["age"].median())
df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)


# Web Page Title
st.markdown("""
<h1 style='text-align:center;
color:#90D5FF;
font-size:50px;'>
🚢 Titanic Dashboard
</h1>
""", unsafe_allow_html=True)
# Create headings
# st.header("We are doing data analysis on titanic dataset to check survival cases")
st.markdown("""
<h4 style='text-align:center;
color:gray;margin-left:15px;'>
Machine Learning + Exploratory Data Analysis
</h4>
""", unsafe_allow_html=True)
with st.container():

    st.subheader("📊 Dataset Information")

    col1,col2,_=st.columns([1,1,4])

    with col1:
        st.metric("Rows",df.shape[0])

    with col2:
        st.metric("Columns",df.shape[1])
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(2))
filtered_df = df.copy()


    
st.sidebar.title("Filters")

gender = st.sidebar.selectbox("Select Gender",['All']+list(df['sex'].unique()))
pclass = st.sidebar.selectbox("Select Class",['All']+list(df['class'].unique()))
town = st.sidebar.selectbox("Select Town",['All']+list(df['embark_town'].unique()))
with st.sidebar.expander("Dataset Preview"):
    st.dataframe(df)
    
if gender!='All':
    filtered_df = filtered_df[filtered_df['sex']==gender]
if pclass!='All':
    filtered_df = filtered_df[filtered_df['class']==pclass]
if town!='All':
    filtered_df = filtered_df[filtered_df['embark_town']==town]



st.markdown("""
<style>
[data-testid="stSidebar"] {
    width: 350px;
}
</style>
""", unsafe_allow_html=True)


# KPIs
col1,col2,col3,col4 = st.columns(4)
with col1:
   st.metric(
    label="Total Passengers",
    value=filtered_df.shape[0]
)
with col2:
    st.metric(
    label="Average Age of Passengers",
    value= round(filtered_df['age'].mean(),0))
with col3:
    st.metric(
    label="Average Fare",
    value= round( filtered_df['fare'].mean(),2 ))
with col4:
    st.metric(
    label="Total Survived Passengers",
    value= filtered_df['alive'].value_counts().get('yes',0))
   

    
# Data Visualization
col1 , col2 = st.columns(2)
with col1:
    fig,ax = plt.subplots(figsize=(6,3))
    sns.countplot( x='class' , data=filtered_df , hue='alive' , ax=ax )
    ax.set_title("Passenger Class vs Survival")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)
with col2:
    fig,ax = plt.subplots(figsize=(6,3))
    sns.countplot( x='embark_town' , data=filtered_df , hue='alive' , palette='viridis' , ax=ax )
    ax.set_title("Embark Town vs Survival")
    ax.set_xlabel("Embark Town")
    ax.set_ylabel("Count")
    st.pyplot(fig)


# Filtered Dataset Sample
st.subheader("🔍 Filtered Dataset")
st.dataframe(filtered_df , height=250)


# Numerical value's Ditribution
col_name = st.selectbox("Select Data", filtered_df.select_dtypes(include=np.number).columns )
fig , ax = plt.subplots( figsize=(8,3) )
sns.histplot( filtered_df[col_name] , kde=True  , ax=ax)
ax.set_title((f"{col_name} Distribution").upper())
ax.set_xlabel(col_name)
ax.set_ylabel("Frequency")
st.pyplot(fig)


# Machine Learning (Predictive Modeling)
st.markdown("---")
st.header("🧠 Global Predictive Model")
st.info("ℹ️ Note: This machine learning model evaluates overall patterns across the entire dataset and is independent of the sidebar filters.")
# Encoding Dataset 
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in df.select_dtypes(include="object"):
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['alive'])
y = df['alive']


# Split the data first
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scalling
scaler = StandardScaler()

num_cols = X_train.select_dtypes(include="float").columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])

# Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score( y_test,y_pred )
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)

col1 , col2 = st.columns(2)
with col1:
    st.subheader("Model Accuracy")

    st.markdown(
        f"""
        <h1 style="text-align:center;
               color:#4CAF50;
               font-size:60px;">
            {acc*100:.2f}%
    </h1>
    """,
    unsafe_allow_html=True
    ) 
    st.subheader("Classification Report") 
    st.code(cr)
with col2:
    st.subheader("Predicted Vs Actual")
    fig,ax = plt.subplots(figsize=(6,3))
    sns.heatmap(
    cm, fmt="d",
    annot=True,
    annot_kws={"size":18},
    ax=ax
)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    st.markdown("---")
st.caption("Developed by Aryan | Streamlit • Scikit-Learn • Python")
