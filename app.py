from groq import Groq
import os
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pickle


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


# def explain_prediction(probability, input_dict, surname):

#     prompt = f"""You are an expert data scientist at a bank, where you specialise in interpreting and explaining the predictions of machine learning models.

#     your machine learning model has predicted that a customer named {surname} has a {round(probability*100, 1)}% probability of churning, based on the information provided below.
#         Here is the customer's information:
#         {input_dict}

#         Here are the machine learning model's top most important features for predicting churn:

#         Feature | Importance
#         ______________________
#         NumOfProducts | 0.323888
#         IsActiveMember | 0.164146
#         Age | 0.109550
#         Geography_Germany | 0.091373
#         Balance | 0.052786
#         Geography_France | 0.046463
#         Gender_Female | 0.045283
#         Geography_Spain | 0.036855
#         CreditScore | 0.035005
#         EstimatedSalary | 0.032655
#         HasCrCard | 0.031940
#         Tenure | 0.030054
#         Gender_Male | 0.000000

#     {pd.set_option('display.max_columns', None)}

#     Here are summary statistics for the churned customers:
#     {df[df['Exited'] == 1].describe()}

# - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
# - If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
# - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

#     Don't mention the probablity of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction..." or "The model predicts that...". Instead, write as if you are an expert data scientist at a bank, and you are explaining the prediction to a non-technical audience.
#     """

#     print("Explanation Prompt: ", prompt)


#     raw_response = client.chat.completions.create(
#         model="llama-3.2-3b-preview",
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return raw_response.choices[0].message.content
# Update the explain_prediction function to use the new client
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


def update_predictions_and_explanation(input_df, input_dict, surname):
    probabilities = {
        "XGB": xgb_model.predict_proba(input_df)[0][1],
        "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
        "KNN": knn_model.predict_proba(input_df)[0][1],
    }

    avg_prob = np.mean(list(probabilities.values()))

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob:.2f}")
    st.write(f"Average Probability: {avg_prob:.2f}")

    explanation = explain_prediction(avg_prob, input_dict, surname, df)

    st.markdown("### Analysis")
    st.write(explanation)

    return avg_prob


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
