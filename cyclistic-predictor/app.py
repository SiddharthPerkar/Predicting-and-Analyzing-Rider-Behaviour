import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Cyclistic Rider Predictor", layout="centered")
st.title("🚴 Cyclistic Rider Behavior Prediction")
st.markdown("Enter your ride details to receive a smart report predicting whether the rider is a Member or Casual user.")

# Load model and encoders
model = pickle.load(open("rider_model.pkl", "rb"))
le_day = pickle.load(open("le_day.pkl", "rb"))
le_bike = pickle.load(open("le_bike.pkl", "rb"))
le_user = pickle.load(open("le_user.pkl", "rb"))

# Input UI
duration = st.slider("🔁 Ride Duration (minutes)", 1, 120, 30)
day = st.selectbox("📅 Day of the Week", le_day.classes_)
hour = st.slider("🕒 Ride Start Time (Hour of Day)", 0, 23, 10)
bike_type = st.selectbox("🚴 Bike Type", le_bike.classes_)

if st.button("Generate Prediction Report"):
    input_df = pd.DataFrame({
        'ride_duration': [duration],
        'day_of_week': le_day.transform([day]),
        'hour': [hour],
        'rideable_type': le_bike.transform([bike_type])
    })

    prediction = model.predict(input_df)
    predicted_type = le_user.inverse_transform(prediction)[0]

    st.subheader("📊 Prediction Result")
    st.markdown(f"""
    - 🕒 **Duration:** {duration} mins  
    - 📅 **Day:** {day}  
    - 🕘 **Hour:** {hour}:00  
    - 🚴 **Bike Type:** {bike_type}  

    ## 🎯 Predicted Rider Type: `{predicted_type}`
    """)

    if predicted_type == "casual":
        st.info("🎉 Likely a leisure rider — often seen on weekends or tourist trips.")
    else:
        st.success("👔 Likely a subscriber/commuter — mostly rides on weekdays.")

