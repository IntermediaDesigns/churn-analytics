import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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


def make_predictions(input_df, input_dict):

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
