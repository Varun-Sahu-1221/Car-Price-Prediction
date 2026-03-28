# 🏎️ Car Price Prediction App

Welcome to the **Car Price Prediction App**! This is a machine learning-powered web application built with Streamlit that estimates the market value of a vehicle based on its technical specifications and features.

## 🚀 Live Demo
Click here to view the Live App : https://varun-sahu-1221-car-price-prediction-app-sclxtb.streamlit.app/

## Linkedin Profile
Get in-touch with me : www.linkedin.com/in/varun-sahu-85835a381

## 📖 Overview

Pricing a car accurately requires analyzing multiple variables, from engine size and horsepower to body style and fuel efficiency. This project takes a dataset of car specifications, trains a Ridge Regression model to understand the pricing patterns, and deploys it through a user-friendly interactive web interface. 

### Key Features:
* **Interactive UI:** Clean and responsive interface built with Streamlit.
* **Instant Predictions:** Users can adjust sliders and dropdowns to see how different car features impact the estimated price in real-time.
* **Comprehensive Inputs:** Supports both numerical features (e.g., MPG, horsepower, curb weight) and categorical features (e.g., car body style, engine type).

## 🛠️ Tech Stack

* **Frontend & Deployment:** Streamlit / Streamlit Community Cloud
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Ridge Regression, Data Scaling)
* **Model Serialization:** Joblib

## 📂 Project Structure
```bash
Car-Price-Prediction/
├── app.py              # Main application code
├── requirements.txt    # List of dependencies (streamlit, pandas, numpy, joblib, scikit-learn)
├── ridge_model.pkl    
├── scaler.pkl   
└── README.md           # Project documentation
