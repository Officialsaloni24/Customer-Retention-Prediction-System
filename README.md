# Customer-Retention-Prediction-System
End-to-end machine learning project to predict repeat purchases of first-time e-commerce customers using Logistic Regression and an interactive Streamlit dashboard.
📊 Customer Retention Prediction System

Predicting repeat purchase probability of first-time customers using Machine Learning and an interactive Streamlit dashboard.

🧠 Problem Statement

E-commerce businesses face a major challenge:
👉 Most customers make only one purchase and never return.

Customer acquisition is expensive, while retention is more profitable.
This project focuses on predicting whether a first-time customer will make a repeat purchase within 30 days, enabling businesses to take early retention actions.
🎯 Project Objective

Identify early signals of customer retention

Predict repeat purchase probability

Provide a user-friendly dashboard for business users

Convert raw transaction data into actionable insights

🏗️ System Overview
CSV Transaction Data
        ↓
Data Cleaning & Date Parsing
        ↓
Feature Engineering (Customer Behavior)
        ↓
Machine Learning Model (Logistic Regression)
        ↓
Retention Probability Prediction
        ↓
Interactive Streamlit Dashboard

📁 Dataset Description

Dataset Used: Online Retail Dataset (UK-based E-commerce)

Source: Kaggle / UCI Machine Learning Repository

Key Columns:

Column Name	Description
InvoiceNo	Transaction ID
InvoiceDate	Date & time of purchase
CustomerID	Unique customer identifier
Quantity	Number of items purchased
UnitPrice	Price per item

🧩 Feature Engineering

Customer-level features were created to capture purchasing behavior:

Feature	Description
PurchaseCount	Number of transactions made
TotalSpend	Total monetary value of purchases
RecencyDays	Time difference between first and last purchase
RepeatPurchase	Target variable (1 = repeat customer, 0 = one-time customer)
🤖 Machine Learning Model

Model Used: Logistic Regression

Why Logistic Regression?

Interpretable

Efficient for binary classification

Suitable for probability prediction

Preprocessing: StandardScaler

Evaluation: Train-test split

The model predicts the probability of a customer making a repeat purchase within 30 days.

📊 Dashboard Features

Built using Streamlit, the dashboard allows:

Uploading a CSV dataset

Automatic model training

Viewing processed customer data

Manual input of customer features

Real-time prediction of retention probability

🖥️ Tech Stack

Programming Language

Python

Libraries

Pandas

NumPy

Scikit-learn

Streamlit

Plotly

Tools

VS Code







