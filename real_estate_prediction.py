#X =df[["bed", "bath", "house_size"]]#
import streamlit as st 
import joblib as jb
import numpy as np
import time 
scaler = jb.load("scaler.pkl")
model = jb.load("model.pkl")

st.title("HOUSE PRICE PREDICTIONS (BETA)")

st.divider()
# give info
bed = st.number_input("Enter your number of bedroom preference", value=2, step = 1)
bath = st.number_input("Enter your number of bathroom preference:",value=1, step = 1)
size =  st.number_input("Enter your house size preference:", value=3000, step= 100)

X =[bed,bath,size]
st.divider()
predict_button = st.button("Click here for result")
st.divider()

if predict_button:
    with st.status("Downloading data..."):
        st.write("Searching for data...")
        time.sleep(2)
        st.write("DONE!")
        time.sleep(2)
        
    X1 = np.array(X)
    X_arr  = scaler.transform([X1])
    prediction  = model.predict(X_arr)[0]
    st.write(f"Your ideal house's price is {prediction:.2f}")
else:
    print("")
