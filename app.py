import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import os


# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# Relative Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
GENDER_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_gender.pkl")
GEO_ENCODER_PATH = os.path.join(BASE_DIR, "one_hot_encoder_geo.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")



# Cache Model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)



# Cache Preprocessing Objects
@st.cache_resource
def load_preprocessing():

    with open(GENDER_ENCODER_PATH, "rb") as file:
        gender_encoder = pickle.load(file)

    with open(GEO_ENCODER_PATH, "rb") as file:
        geo_encoder = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return gender_encoder, geo_encoder, scaler


# Load everything once
model = load_model()
gender_encoder, geo_encoder, scaler = load_preprocessing()


# UI
st.title("📉 Customer Churn Prediction System")
st.write("Fill in the customer information below.")

st.divider()

# User Inputs

geography = st.selectbox(
    "Geography",
    geo_encoder.categories_[0]
)

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=35
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=0.0,
    step=100.0
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

st.divider()


# Predict Button

if st.button("🔍 Predict Churn", use_container_width=True):

    # Label Encoding
    gender_encoded = gender_encoder.transform([gender])[0]

    # Main DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Geography Encoding
    geo_df = pd.DataFrame({
        "Geography": [geography]
    })

    geo_encoded = geo_encoder.transform(geo_df)

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=geo_encoder.get_feature_names_out(["Geography"])
    )

    # Combine Features
    input_data = pd.concat(
        [input_data.reset_index(drop=True),
         geo_encoded_df.reset_index(drop=True)],
        axis=1
    )


    # Scale Features
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled, verbose=0)
    prediction_prob = prediction[0][0]

    st.subheader("Prediction Result")

    if prediction_prob > 0.5:
        st.error(
            f"⚠️ Customer is likely to churn.\n\n"
            f"Probability: **{prediction_prob:.2%}**"
        )
    else:
        st.success(
            f"✅ Customer is NOT likely to churn.\n\n"
            f"Probability: **{prediction_prob:.2%}**"
        )

    st.progress(float(prediction_prob))

    st.metric(
        "Churn Probability",
        f"{prediction_prob:.2%}"
    )