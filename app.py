import streamlit as st
import pandas as pd
import pickle

def load_model(filename):
    with open(filename, "rb") as file:
        model = pickle.load(file)

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