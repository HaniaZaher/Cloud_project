# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import hashlib

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Enterprise Finance Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# DATABASE
# -----------------------------------

conn = sqlite3.connect("finance.db", check_same_thread=False)
cursor = conn.cursor()

# Users Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# Transactions Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    type TEXT,
    category TEXT,
    amount REAL,
    payment_method TEXT,
    department TEXT,
    status TEXT,
    description TEXT
)
""")

conn.commit()

# -----------------------------------
# AUTH FUNCTIONS
# -----------------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    return cursor.fetchone()

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------------
# LOGIN PAGE
# -----------------------------------

if not st.session_state.logged_in:

    st.title("🏢 Enterprise Finance Management System")

    menu = st.tabs(["Login", "Register"])

    # LOGIN
    with menu[0]:

        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            user = login_user(username, password)

            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Credentials")

    # REGISTER
    with menu[1]:

        st.subheader("Create Account")

        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):

            if create_user(new_user, new_pass):
                st.success("Account Created")

            else:
                st.error("Username Already Exists")

# -----------------------------------
# MAIN APP
# -----------------------------------

else:

    st.sidebar.title("💼 Finance System")
    st.sidebar.write(f"Welcome, {st.session_state.username}")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Add Transaction",
            "Transactions",
            "Analytics",
            "Reports"
        ]
    )

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    df = pd.read_sql_query(
        "SELECT * FROM transactions",
        conn
    )

    # -----------------------------------
    # DASHBOARD
    # -----------------------------------

    if menu == "Dashboard":

        st.title("📊 Executive Dashboard")

        total_income = df[df["type"] == "Income"]["amount"].sum()
        total_expense = df[df["type"] == "Expense"]["amount"].sum()
        balance = total_income - total_expense

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Income", f"${total_income:,.2f}")
        col2.metric("Total Expenses", f"${total_expense:,.2f}")
        col3.metric("Net Profit", f"${balance:,.2f}")

        st.divider()

        if not df.empty:

            # Monthly Finance Trend
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.strftime("%Y-%m")

            monthly = df.groupby(
                ["month", "type"]
            )["amount"].sum().reset_index()

            fig = px.line(
                monthly,
                x="month",
                y="amount",
                color="type",
                markers=True,
                title="Monthly Financial Trend"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Department Expenses
            expense_df = df[df["type"] == "Expense"]

            if not expense_df.empty:

                dep_chart = expense_df.groupby(
                    "department"
                )["amount"].sum().reset_index()

                fig2 = px.pie(
                    dep_chart,
                    names="department",
                    values="amount",
                    title="Department Expense Distribution"
                )

                st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------
    # ADD TRANSACTION
    # -----------------------------------

    elif menu == "Add Transaction":

        st.title("➕ Add Financial Transaction")

        with st.form("transaction_form"):

            date = st.date_input("Transaction Date")

            trans_type = st.selectbox(
                "Transaction Type",
                ["Income", "Expense"]
            )

            category = st.selectbox(
                "Category",
                [
                    "Sales",
                    "Marketing",
                    "Operations",
                    "Payroll",
                    "Software",
                    "Taxes",
                    "Utilities",
                    "Other"
                ]
            )

            amount = st.number_input(
                "Amount",
                min_value=0.0,
                step=1.0
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Cash",
                    "Bank Transfer",
                    "Credit Card",
                    "PayPal"
                ]
            )

            department = st.selectbox(
                "Department",
                [
                    "Finance",
                    "HR",
                    "IT",
                    "Sales",
                    "Operations"
                ]
            )

            status = st.selectbox(
                "Status",
                [
                    "Completed",
                    "Pending",
                    "Cancelled"
                ]
            )

            description = st.text_area("Description")

            submitted = st.form_submit_button("Save Transaction")

            if submitted:

                cursor.execute("""
                INSERT INTO transactions
                (
                    date,
                    type,
                    category,
                    amount,
                    payment_method,
                    department,
                    status,
                    description
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(date),
                    trans_type,
                    category,
                    amount,
                    payment_method,
                    department,
                    status,
                    description
                ))

                conn.commit()

                st.success("Transaction Added Successfully")

    # -----------------------------------
    # TRANSACTIONS
    # -----------------------------------

    elif menu == "Transactions":

        st.title("📁 Financial Records")

        if not df.empty:

            # Filters
            col1, col2 = st.columns(2)

            with col1:
                selected_type = st.selectbox(
                    "Filter by Type",
                    ["All", "Income", "Expense"]
                )

            with col2:
                selected_department = st.selectbox(
                    "Filter by Department",
                    ["All"] + list(df["department"].unique())
                )

            filtered_df = df.copy()

            if selected_type != "All":
                filtered_df = filtered_df[
                    filtered_df["type"] == selected_type
                ]

            if selected_department != "All":
                filtered_df = filtered_df[
                    filtered_df["department"] == selected_department
                ]

            st.dataframe(
                filtered_df,
                use_container_width=True
            )

            # CSV Download
            csv = filtered_df.to_csv(index=False).encode()

            st.download_button(
                "⬇ Download CSV",
                csv,
                "finance_report.csv",
                "text/csv"
            )

        else:
            st.warning("No transactions available")

    # -----------------------------------
    # ANALYTICS
    # -----------------------------------

    elif menu == "Analytics":

        st.title("📈 Financial Analytics")

        if not df.empty:

            expense_df = df[df["type"] == "Expense"]

            # Expense by Category
            category_chart = expense_df.groupby(
                "category"
            )["amount"].sum().reset_index()

            fig = px.bar(
                category_chart,
                x="category",
                y="amount",
                title="Expenses by Category"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cash Flow Gauge
            income = df[df["type"] == "Income"]["amount"].sum()
            expense = df[df["type"] == "Expense"]["amount"].sum()

            balance = income - expense

            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=balance,
                title={'text': "Company Balance"},
                gauge={'axis': {'range': [None, max(balance*2, 1000)]}}
            ))

            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.warning("No analytics available")

    # -----------------------------------
    # REPORTS
    # -----------------------------------

    elif menu == "Reports":

        st.title("📄 Financial Reports")

        if not df.empty:

            total_income = df[df["type"] == "Income"]["amount"].sum()
            total_expense = df[df["type"] == "Expense"]["amount"].sum()
            net_profit = total_income - total_expense

            st.subheader("Summary Report")

            st.write(f"### Total Income: ${total_income:,.2f}")
            st.write(f"### Total Expenses: ${total_expense:,.2f}")
            st.write(f"### Net Profit: ${net_profit:,.2f}")

            st.subheader("Top Expenses")

            expense_report = df[df["type"] == "Expense"] \
                .groupby("category")["amount"] \
                .sum() \
                .sort_values(ascending=False)

            st.table(expense_report)

        else:
            st.warning("No reports available")