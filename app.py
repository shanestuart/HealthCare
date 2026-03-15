import streamlit as st
import pickle
import pandas as pd
from gtts import gTTS
import tempfile
from transformers import pipeline

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="centered"
)

# ---------------------------------------------------
# Load Resources (Cached for Performance)
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return pickle.load(open("disease_prediction_model.pkl", "rb"))

@st.cache_resource
def load_symptoms():
    return pickle.load(open("symptom_columns.pkl", "rb"))

@st.cache_resource
def load_precautions():
    return pickle.load(open("precautions.pkl", "rb"))

@st.cache_resource
def load_ai_model():
    return pipeline("text-generation", model="gpt2")


model = load_model()
symptom_columns = load_symptoms()
precautions_df = load_precautions()
ai_model = load_ai_model()

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

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
        return ["Consult a doctor for proper medical advice."]


def generate_ai_explanation(symptoms, disease):

    prompt = f"""
A patient has the following symptoms: {symptoms}

The predicted disease is: {disease}

Explain this disease in simple terms and give basic health advice.
"""

    try:
        result = ai_model(prompt, max_length=120)
        return result[0]["generated_text"]

    except:
        return "AI explanation currently unavailable."


def generate_voice(text):

    tts = gTTS(text=text)

    temp_audio = tempfile.NamedTemporaryFile(delete=False)

    tts.save(temp_audio.name)

    return temp_audio.name


# ---------------------------------------------------
# UI Layout
# ---------------------------------------------------

st.title("🏥 AI Healthcare Assistant")

st.markdown(
"""
This AI system predicts **possible diseases from symptoms** and suggests **basic precautions**.

⚠️ This tool is for **educational purposes only** and is **not a medical diagnosis system**.
"""
)

st.divider()

# User Input
symptom_input = st.text_input(
    "Enter your symptoms (separated by spaces)",
    placeholder="Example: fever cough headache"
)

predict_button = st.button("Analyze Symptoms")

# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------

if predict_button:

    if symptom_input.strip() == "":
        st.warning("Please enter at least one symptom.")
    else:

        symptoms = symptom_input.lower().split()

        with st.spinner("Analyzing symptoms..."):

            disease = predict_disease(symptoms)

            precautions = get_precautions(disease)

            explanation = generate_ai_explanation(symptom_input, disease)

        st.success("Analysis Complete")

        st.subheader("Predicted Condition")

        st.write(f"**{disease}**")

        st.subheader("Recommended Precautions")

        for p in precautions:
            st.write("•", p)

        st.subheader("AI Medical Explanation")

        st.write(explanation)

        # Voice Response
        voice_text = f"""
The predicted condition is {disease}.
Recommended precautions are {', '.join(precautions)}.
"""

        audio_file = generate_voice(voice_text)

        st.subheader("AI Voice Assistant")

        st.audio(audio_file)

st.divider()

st.caption("AI Healthcare Assistant • Built with Streamlit")
