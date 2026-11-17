"""Shared pytest fixtures and synthetic data generators for StreamLoop tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def local_dataset_path() -> Path:
    """Return path to local fallback dataset."""
    return Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


@pytest.fixture
def sample_raw_telco_df() -> pd.DataFrame:
    """Generate a realistic synthetic DataFrame adhering to the Telco schema."""
    n_samples = 100
    rng = np.random.default_rng(42)

    genders = ["Female", "Male"]
    yes_no = ["Yes", "No"]
    mult_lines = ["Yes", "No", "No phone service"]
    internet = ["DSL", "Fiber optic", "No"]
    internet_add_ons = ["Yes", "No", "No internet service"]
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    data = {
        "customerID": [f"{i:04d}-TEST{rng.integers(100, 999)}" for i in range(n_samples)],
        "gender": rng.choice(genders, size=n_samples),
        "SeniorCitizen": rng.choice([0, 1], size=n_samples, p=[0.84, 0.16]),
        "Partner": rng.choice(yes_no, size=n_samples),
        "Dependents": rng.choice(yes_no, size=n_samples),
        "tenure": rng.integers(0, 73, size=n_samples),
        "PhoneService": rng.choice(yes_no, size=n_samples),
        "MultipleLines": rng.choice(mult_lines, size=n_samples),
        "InternetService": rng.choice(internet, size=n_samples),
        "OnlineSecurity": rng.choice(internet_add_ons, size=n_samples),
        "OnlineBackup": rng.choice(internet_add_ons, size=n_samples),
        "DeviceProtection": rng.choice(internet_add_ons, size=n_samples),
        "TechSupport": rng.choice(internet_add_ons, size=n_samples),
        "StreamingTV": rng.choice(internet_add_ons, size=n_samples),
        "StreamingMovies": rng.choice(internet_add_ons, size=n_samples),
        "Contract": rng.choice(contracts, size=n_samples),
        "PaperlessBilling": rng.choice(yes_no, size=n_samples),
        "PaymentMethod": rng.choice(payments, size=n_samples),
        "MonthlyCharges": np.round(rng.uniform(18.0, 120.0, size=n_samples), 2),
        "TotalCharges": ["" for _ in range(n_samples)],
        "Churn": rng.choice(["Yes", "No"], size=n_samples, p=[0.27, 0.73]),
    }

    df = pd.DataFrame(data)

    # TotalCharges contains whitespace for tenure==0, otherwise approximate tenure * MonthlyCharges
    total_charges = []
    for t, m in zip(df["tenure"], df["MonthlyCharges"]):
        if t == 0:
            total_charges.append(" ")
        else:
            total_charges.append(f"{round(float(t * m), 2)}")
    df["TotalCharges"] = total_charges

    return df
