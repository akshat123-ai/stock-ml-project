# 📈 Stock Market Prediction System
End-to-End Machine Learning Pipeline with FastAPI & Streamlit

## Overview
This project implements a production-style machine learning system that predicts
short-term stock price movement using technical indicators. It includes a full
ML pipeline, FastAPI backend for model inference, and a Streamlit dashboard for
interactive analytics and predictions.

## Architecture
User → Streamlit Dashboard → FastAPI API → Machine Learning Model

## Features
- Dataset exploration dashboard
- Model benchmarking across multiple ML algorithms
- Feature engineering from financial time-series data
- Feature selection and scaling
- Trading strategy simulation
- Explainable AI (Permutation Feature Importance)
- Live stock predictions using Yahoo Finance
- REST API inference service

## Technologies
Python, Scikit-learn, XGBoost, LightGBM, FastAPI, Streamlit, Plotly, Pandas, NumPy

## Project Structure
stock-ml-dashboard/
│
├── backend/
│   └── api.py
│
├── dashboard.py
├── train_model.py
│
├── model.pkl
├── scaler.pkl
├── selector.pkl
├── selected_features.pkl
│
└── dataset/

## Installation
pip install -r requirements.txt

## Run Backend
python -m uvicorn backend.api:app --reload

API Docs:
http://127.0.0.1:8000/docs

## Run Dashboard
streamlit run dashboard.py

## Author
Akshat – Machine Learning & Data Science Enthusiast