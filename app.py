import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model, scaler, columns
# -------------------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# -------------------------------
# UI Title
# -------------------------------
st.title("❤️ Heart Disease Prediction App")
st.caption("Built by Swathi | ML + Streamlit Project")

# -------------------------------
# User Inputs
# -------------------------------
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------------------
# Initialize prediction
# -------------------------------
prediction = None
prob = None

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict"):

    # Create full input dictionary with all columns = 0
    input_dict = dict.fromkeys(expected_columns, 0)

    # Fill numeric values
    input_dict.update({
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak
    })

    # Fill categorical (one-hot encoding)
    input_dict['Sex_' + sex] = 1
    input_dict['ChestPainType_' + chest_pain] = 1
    input_dict['RestingECG_' + resting_ecg] = 1
    input_dict['ExerciseAngina_' + exercise_angina] = 1
    input_dict['ST_Slope_' + st_slope] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\nProbability: {prob:.2f}")

# -------------------------------
# Chatbot Section
# -------------------------------
st.subheader("💬 Health Assistant Chatbot")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "You":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")

# User input
user_input = st.text_input("Ask your question:")

if user_input:
    st.session_state.messages.append({"role": "You", "content": user_input})

    user_input_lower = user_input.lower()

    # Chatbot logic
    if "heart" in user_input_lower:
        response = "Heart disease occurs due to reduced blood flow to the heart."

    elif "cholesterol" in user_input_lower:
        response = "High cholesterol increases risk of heart disease."

    elif "bp" in user_input_lower or "blood pressure" in user_input_lower:
        response = "High blood pressure puts extra strain on your heart."

    elif "risk" in user_input_lower or "result" in user_input_lower:
        if prediction is None:
            response = "Please click Predict first to get your result."
        elif prediction == 1:
            response = "Your result shows HIGH risk. Please consult a doctor."
        else:
            response = "Your result shows LOW risk. Maintain a healthy lifestyle."

    else:
        response = "Please ask about heart health, cholesterol, BP, or your result."

    st.session_state.messages.append({"role": "Bot", "content": response})