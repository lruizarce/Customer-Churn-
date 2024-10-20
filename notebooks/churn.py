import os
from dotenv import load_dotenv
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import utils

# Retrieve the API key from the environment variable
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize the OpenAI client
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)


# Load model function
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


# Load models
xgboost_model = load_model("model/xgb_model.pkl")
dc_model = load_model("model/dc_model.pkl")
nb_model = load_model("model/nb_model.pkl")
rf_model = load_model("model/rf_model.pkl")
voting_model = load_model("model/voting_clf.pkl")
voting_hard_model = load_model("model/xgb_model.pkl")
xgboost_feature = load_model("model/xgboost_feature_engineering.pkl")
xgboost_smote = load_model("model/xgboost-SMOTE.pkl")
kn_model = load_model("model/kn_model.pkl")


# Prepare input data for prediction
def prepare_input(
    credit_score,
    location,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
):
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_German": 1 if location == "German" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


# Make predictions and display results
def make_predictions(input_df, input_dict):
    probabilities = {
        "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
        "Random Forest": rf_model.predict_proba(input_df)[0][1],
        "K-Nearest Neighbors": kn_model.predict_proba(input_df)[0][1],
    }
    avg_probabilities = np.mean(list(probabilities.values()))

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob}")
    st.write(f"Average probability: {avg_probabilities}")

    col1, col2 = st.columns(2)

    with col1:
        fig = utils.create_gauge_chart(avg_probabilities)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f'The customer has a {avg_probabilities: 2%} probability of churning')

    with col2:
        fig_prob = utils.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_prob, use_container_width=True)
    return avg_probabilities

# Explain predictions using the OpenAI client
def explain_predictions(probability, input_dict, surname):
    prompt = f"""
    You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

    Here are the machine learning model's top 100 important features for predicting churn:

    Feature | Importance
    --------------------
    NumOfProducts    | 0.323888
    IsActiveMember   | 0.164146
    Age              | 0.109550
    Geography_German | 0.091373
    Balance          | 0.052786
    Geography_France | 0.046463
    Gender_Female    | 0.045283
    CreditScore      | 0.036855
    EstimatedSalary  | 0.032655
    HasCrCard        | 0.031940
    Tenure           | 0.030054
    Gender_Female    | 0.000000
    
    {pd.set_option('display.max_columns', None)}

    Here are summary statistics for churned customers
    {df[df['Exited'] == 0].describe()}

    Here are summary statistics for non-churned customers
    {df[df['Exited'] == 1].describe()}

    - If the customer has over 30% risk of churning, generate a 3-sentence explanation of why they are at risk.
    - If the customer has less than a 40% risk of churning, generate a 3-sentence explanation why they are not at risk.
    - Your explanations should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.
    
    Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction
    """

    print("EXPLANATION PROMPT", prompt)
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview", messages=[{"role": "user", "content": prompt}]
    )
    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are an expert at handling client relationships. You are responsible for ensuring customers stay with your bank and are incentived with various offers.
    
    You noticed a customer named {surname} has a {round(probability* 100, 1)}% probability of churning.
    
    Here is some explanation as to why the customer might be at risk of churning:
    {explanation}
    
    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.
    
    Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
    """
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview", messages=[{"role": "user", "content": prompt}]
    )

    print("\n\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content


# Streamlit app setup
st.title("Customer Churn Prediction")
df = pd.read_csv("data/churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split("-")[0])
    selected_surname = selected_customer_option.split("-")[1]
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=800,
            value=int(selected_customer["CreditScore"]),
        )
        location = st.selectbox(
            "Location",
            ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(selected_customer["Geography"]),
        )
        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            index=0 if selected_customer["Gender"] == "Male" else 1,
        )
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=int(selected_customer["Age"])
        )
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"]),
        )

    with col2:
        balance = st.number_input(
            "Balance", min_value=0.0, value=float(selected_customer["Balance"])
        )
        num_of_products = st.number_input(
            "Number of Services",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"]),
        )
        has_credit_card = st.checkbox(
            "Has Credit Card", value=bool(selected_customer["HasCrCard"])
        )
        is_active_member = st.checkbox(
            "Is Active member", value=bool(selected_customer["IsActiveMember"])
        )
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"]),
        )

    input_df, input_dict = prepare_input(
        credit_score,
        location,
        gender,
        age,
        tenure,
        balance,
        num_of_products,
        has_credit_card,
        is_active_member,
        estimated_salary,
    )
    avg_probabilities = make_predictions(input_df, input_dict)

    explanation = explain_predictions(
        avg_probabilities, input_dict, selected_customer["Surname"]
    )
    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    email = generate_email(
        avg_probabilities, input_dict, explanation, selected_customer["Surname"]
    )

    st.markdown("---")
    st.subheader("Personalized Email")
    st.markdown(email)
