import streamlit as st
import joblib
import pandas as pd
from gtts import gTTS
import tempfile
from transformers import pipeline

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="centered"
)

# -----------------------------------
# Load Saved Resources
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("disease_prediction_model.pkl")

@st.cache_resource
def load_symptoms():
    return joblib.load("symptom_columns.pkl")

@st.cache_resource
def load_precautions():
    return joblib.load("precautions.pkl")

@st.cache_resource
def load_ai_model():
    return pipeline("text-generation", model="distilgpt2")


model = load_model()
symptom_columns = load_symptoms()
precautions_df = load_precautions()
ai_model = load_ai_model()

# -----------------------------------
# Helper Functions
# -----------------------------------

def predict_disease(user_symptoms):

    input_vector = [0] * len(symptom_columns)

    for symptom in user_symptoms:
        if symptom in symptom_columns:
            index = symptom_columns.index(symptom)
            input_vector[index] = 1

    prediction = model.predict([input_vector])[0]

    return prediction


def get_precautions(disease):

    try:
        row = precautions_df[precautions_df["Disease"] == disease]

        precautions = row.iloc[0].values[1:]

        return [p for p in precautions if str(p) != "nan"]

    except:
        return ["Consult a doctor for medical advice."]


def ai_explanation(symptoms, disease):

    prompt = f"""
Symptoms: {symptoms}

Possible disease: {disease}

Explain this condition and give basic health advice.
"""

    result = ai_model(prompt, max_length=120)

    return result[0]["generated_text"]


def generate_voice(text):

    tts = gTTS(text=text)

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    tts.save(temp_file.name)

    return temp_file.name


# -----------------------------------
# UI Layout
# -----------------------------------

st.title("🏥 AI Healthcare Assistant")

st.markdown(
"""
Enter your symptoms and the AI will predict a possible condition and suggest precautions.

⚠️ This tool is for **educational purposes only** and should not replace professional medical advice.
"""
)

st.divider()

user_input = st.text_input(
    "Enter symptoms separated by spaces",
    placeholder="Example: fever cough headache"
)

predict_button = st.button("Analyze Symptoms")

# -----------------------------------
# Prediction Logic
# -----------------------------------

if predict_button:

    if user_input.strip() == "":
        st.warning("Please enter at least one symptom.")
    else:

        symptoms = user_input.lower().split()

        with st.spinner("Analyzing symptoms..."):

            disease = predict_disease(symptoms)

            precautions = get_precautions(disease)

            explanation = ai_explanation(user_input, disease)

        st.success("Analysis Complete")

        st.subheader("Predicted Condition")

        st.write(f"**{disease}**")

        st.subheader("Recommended Precautions")

        for p in precautions:
            st.write("•", p)

        st.subheader("AI Medical Explanation")

        st.write(explanation)

        voice_text = f"""
The predicted condition is {disease}.
Recommended precautions are {', '.join(precautions)}.
"""

        audio_file = generate_voice(voice_text)

        st.subheader("AI Voice Assistant")

        st.audio(audio_file)

st.divider()

st.caption("AI Healthcare Assistant • Powered by AI")
