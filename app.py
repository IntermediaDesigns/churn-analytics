from groq import Groq
import os
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import utils as ut


def get_groq_client():
    """
    Initialize and return the Groq client using credentials from .env file.
    Returns None if initialization fails.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        # Use the Groq client directly instead of OpenAI
        client = Groq(api_key=api_key)

        return client

    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None


def send_groq_chat_completion(prompt, system_message=None):
    """
    Send a chat completion request to Groq.

    Args:
        prompt (str): The user prompt
        system_message (str, optional): System message to set context
    """
    try:
        client = get_groq_client()
        if not client:
            return "Unable to generate explanation due to client connection error."

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        return "Unable to generate explanation at this time."


# Example function to test the connection
def test_groq_connection():
    """
    Test the Groq client connection and API key.
    Returns True if successful, False otherwise.
    """
    client = get_groq_client()
    if not client:
        return False

    try:
        # Simple test completion
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1,
        )
        return True
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False


def explain_prediction(probability, input_dict, surname, df):
    # Format the input data more readably
    formatted_data = "\n".join([f"- {k}: {v}" for k, v in input_dict.items()])

    # Calculate key statistics
    churned_stats = df[df["Exited"] == 1].describe()
    avg_age_churned = churned_stats["Age"]["mean"]
    avg_balance_churned = churned_stats["Balance"]["mean"]
    avg_products_churned = churned_stats["NumOfProducts"]["mean"]

    prompt = f"""You are a senior data scientist at a bank specializing in customer retention analysis. You need to explain why a customer might or might not be at risk of leaving the bank (churning).

Customer Profile for {surname}:
{formatted_data}

Context:
- The number of products a customer has is the strongest indicator of churn risk
- Active membership status is the second most important factor
- Age is the third most important factor
- Location (specifically being in Germany) is the fourth most important factor
- Balance and other demographic factors have moderate importance

Statistical Context:
- Average age of churning customers: {avg_age_churned:.1f} years
- Average balance of churning customers: ${avg_balance_churned:,.2f}
- Average number of products of churning customers: {avg_products_churned:.1f}

Task:
{probability * 100:.1f}% of similar customers have churned. Write a clear, actionable analysis in exactly three sentences that:
1. Identifies the primary risk factors or protective factors for this customer
2. Compares their profile to typical churning patterns
3. Suggests one specific action that could impact their retention

Style Guide:
- Write in a professional but accessible tone
- Focus on actionable insights
- Avoid technical jargon
- Don't mention percentages, probabilities, or models
- Don't use phrases like "based on the data" or "analysis shows"
"""

    system_message = "You are a senior banking data scientist who explains customer retention patterns clearly and concisely."

    return send_groq_chat_completion(prompt, system_message)


def generate_email(probability, input_dict, explanation, surname):
    """
    Generate a personalized retention email for a customer using Groq's LLM.

    Args:
        probability (float): Churn probability
        input_dict (dict): Customer information
        explanation (str): Previous analysis of customer's situation
        surname (str): Customer's surname
    """
    # Format customer information more readably
    customer_info = "\n".join([f"- {k}: {v}" for k, v in input_dict.items()])

    prompt = f"""You are a thoughtful bank relationship manager writing a personalized email to a valued customer.

Customer Details:
Name: {surname}
{customer_info}

Previous Analysis:
{explanation}

Task:
Write a professional email to this customer that:
1. Opens with a warm, personalized greeting
2. Acknowledges their relationship with the bank
3. Offers specific, relevant incentives based on their profile
4. Closes with a clear call to action

Required Elements:
- Subject line
- Professional email body
- 3-4 bullet points of personalized offers/incentives
- Clear next steps
- Professional signature

Style Guidelines:
- Warm and professional tone
- No mention of churn risk or technical analysis
- Focus on value and appreciation
- Specific to their profile and usage patterns
- Keep incentives realistic and relevant to their profile

Format the email with clear sections and proper spacing."""

    system_message = """You are an experienced bank relationship manager who excels at personalizing communication and building customer loyalty."""

    try:
        email_content = send_groq_chat_completion(prompt, system_message)
        return email_content

    except Exception as e:
        st.error(f"Error generating email: {str(e)}")
        return "Unable to generate email at this time."


def update_predictions_and_explanation(input_df, input_dict, surname):
    probabilities = {
        "XGB": xgb_model.predict_proba(input_df)[0][1],
        "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
        "KNN": knn_model.predict_proba(input_df)[0][1],
    }

    avg_prob = np.mean(list(probabilities.values()))
    
    col1, col2 = st.columns(2)
    with col1:
        fig = ut.create_guage_chart(avg_prob)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_prob:.2f} probability of churning.")
        
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob:.2f}")
    st.write(f"Average Probability: {avg_prob:.2f}")

    explanation = explain_prediction(avg_prob, input_dict, surname, df)

    st.markdown("### Analysis")
    st.write(explanation)

    # Generate and display email
    email_content = generate_email(avg_prob, input_dict, explanation, surname)

    st.markdown("### Suggested Email")
    st.write(email_content)

    return avg_prob


def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


xgb_model = load_model("xgb_model.pkl")
naive_bayes_model = load_model("nb_model.pkl")
random_forest_model = load_model("rf_model.pkl")
decision_tree_model = load_model("dt_model.pkl")
svm_model = load_model("svm_model.pkl")
knn_model = load_model("knn_model.pkl")
voting_classifier_model = load_model("voting_clf.pkl")
xgb_SMOTE_model = load_model("xgb_SMOTE.pkl")
xgb_featureEngineered_model = load_model("xgb_featureEngineered.pkl")


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
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


st.title("Churn Analytics")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    customer_id = int(selected_customer_option.split(" - ")[0])
    customer = df[df["CustomerId"] == customer_id].iloc[0]

    selected_surname = selected_customer_option.split(" - ")[1]
    selected_customer = df.loc[df["CustomerId"] == customer_id]

    col1, col2 = st.columns(2)
    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer["CreditScore"]),
        )
        location = st.selectbox(
            "Location",
            ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(
                selected_customer["Geography"].iloc[0]
            ),
        )
        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            index=0 if selected_customer["Gender"].iloc[0] == "Male" else 1,
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer["Age"].iloc[0]),
        )
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=10,
            value=int(selected_customer["Tenure"].iloc[0]),
        )

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=float(selected_customer["Balance"]),
        )
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"]),
        )
        has_credit_card = st.checkbox(
            "Has Credit Card",
            value=bool(selected_customer["HasCrCard"].iloc[0]),
        )
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer["IsActiveMember"].iloc[0]),
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
        num_products,
        has_credit_card,
        is_active_member,
        estimated_salary,
    )
    avg_prob = update_predictions_and_explanation(
        input_df, input_dict, selected_surname
    )
