import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
st.set_page_config("Linear Regression",layout="centered")
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")
st.markdown("""  
    <div class="card">
            <h1>Linear Regression</h1>
            <p>Predict <b> Tip Amount </b> from <b>Total Bill</b> using Linear Regression...</p>
    </div>
""",unsafe_allow_html=True)
#Datset Preview
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()
st.markdown("""
    <div class="card">
        <b>Datset Preview</b>
    </div>
"""
'',unsafe_allow_html=True)
st.dataframe(df.head())
#Prepare the data
x,y=df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
#Metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)
#Visualize the data
st.markdown("""
    <div class="card">
        <b>Simple Linear Regression Plot (Total Bill vs Tips)</b>
    </div>
"""
'',unsafe_allow_html=True)
fig,ax=plt.subplots()
ax.scatter(df['total_bill'],df['tip'],alpha=0.6)
ax.plot(df["total_bill"],lr.predict(sc.transform(x)),color="red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tips")
st.pyplot(fig)
#Performance
st.markdown("""
    <div class="card">
        <b>Model Performance</b>
    </div>
"""
'',unsafe_allow_html=True)
c1,c2=st.columns(2)
c1.metric('MAE (Mean Absolute Error) : ',f"{mae:.2f}")
c2.metric('RMSE (Root Mean Squared Error) : ',f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric('R2 Score : ',f"{r2:.3f}")
c4.metric('Adjusted R2 Score : ',f"{adj_r2:.3f}")
#m and c
st.markdown(f""" 
    <div class="card">
            <h3>Model Intercept & Coefficient</h3>
            <p><b>Coefficient : </b>{lr.coef_[0]:.3f}<br>
                <b>Intercept : </b>{lr.intercept_:.3f}
            </p>
    </div>
""",unsafe_allow_html=True)
#Prediction
st.markdown(""" 
    <div class="card">
        Predicted Tip Amount
    </div>
""",unsafe_allow_html=True)
bill=st.slider("Total Bill : ",float(df.total_bill.min()),float(df.total_bill.max()),30.0)
tip=lr.predict(sc.transform([[bill]]))[0]
st.markdown(f""" 
    <div class="prediction-box">
        Prediction Tip : ${tip:.2f}
    </div>
""",unsafe_allow_html=True)