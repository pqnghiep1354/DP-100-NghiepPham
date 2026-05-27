"""
train.py - Training Script cho Hyperparameter Sweep
Dự án: ER Triage Classification
"""

import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args):
    print(f"\n{'*'*60}")
    print(f"  reg_rate = {args.reg_rate}")
    print(f"{'*'*60}")

    df = get_data(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(args.reg_rate, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print(f"{'*'*60}\n")


def get_data(path):
    df = pd.read_csv(path)

    le_arrival  = LabelEncoder()
    le_symptom  = LabelEncoder()
    le_target   = LabelEncoder()

    df['Arrival_Mode']    = le_arrival.fit_transform(df['Arrival_Mode'])
    df['Primary_Symptom'] = le_symptom.fit_transform(df['Primary_Symptom'])
    df['Routing_Class']   = le_target.fit_transform(df['Routing_Class'])

    scaler = MinMaxScaler()
    feature_cols = ['Age', 'Arrival_Mode', 'Heart_Rate', 'Pain_Scale', 'Primary_Symptom']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df


def split_data(df):
    feature_cols = ['Age', 'Arrival_Mode', 'Heart_Rate', 'Pain_Scale', 'Primary_Symptom']
    X = df[feature_cols].values
    y = df['Routing_Class'].values
    return train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)


def train_model(reg_rate, X_train, y_train):
    # Log hyperparameter
    mlflow.log_param("Regularization rate", reg_rate)
    mlflow.log_param("solver", "liblinear")

    model = LogisticRegression(
        C=1 / reg_rate,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ).fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    y_hat    = model.predict(X_test)
    y_scores = model.predict_proba(X_test)

    acc    = accuracy_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    f1     = f1_score(y_test, y_hat)
    auc    = roc_auc_score(y_test, y_scores[:, 1])

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")

    # Log metrics — training_accuracy_score là primary metric cho sweep
    mlflow.log_metric("training_accuracy_score", acc)
    mlflow.log_metric("Recall",   recall)
    mlflow.log_metric("F1_Score", f1)
    mlflow.log_metric("AUC",      auc)

    # ROC Curve artifact
    fpr, tpr, _ = roc_curve(y_test, y_scores[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr, tpr, color='#E53935', linewidth=2, label=f'AUC={auc:.3f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (reg_rate={str(args.reg_rate)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ROC-Curve.png", dpi=100)
    mlflow.log_artifact("ROC-Curve.png")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--reg_rate", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
