"""
train-model-mlflow.py
Training Script với MLflow Tracking
Dự án: ER Triage Classification - Phân loại bệnh nhân khẩn cấp
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, confusion_matrix,
    classification_report, recall_score, precision_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(args):
    print("\n" + "="*60)
    print("  ER TRIAGE CLASSIFICATION - MLflow Training")
    print("="*60)

    # Đọc và xử lý data
    df = get_data(args.training_data)

    # Chia train/test
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    model = train_model(args.reg_rate, X_train, y_train)

    # Đánh giá và log metrics
    evaluate_model(model, X_test, y_test)

    print("="*60)
    print("  Training hoàn thành!")
    print("="*60 + "\n")


# ── ĐỌC VÀ TIỀN XỬ LÝ DATA ──────────────────────────────────────────────────
def get_data(path):
    print("\n📂 Đang đọc dataset...")
    df = pd.read_csv(path)
    print(f"   Đã load: {len(df)} bệnh nhân, {df.shape[1]} cột")

    # Label Encoding cho biến phân loại
    le_arrival  = LabelEncoder()
    le_symptom  = LabelEncoder()
    le_target   = LabelEncoder()

    df['Arrival_Mode']    = le_arrival.fit_transform(df['Arrival_Mode'])
    df['Primary_Symptom'] = le_symptom.fit_transform(df['Primary_Symptom'])
    df['Routing_Class']   = le_target.fit_transform(df['Routing_Class'])

    print(f"   Routing_Class encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # Min-Max Normalization
    scaler = MinMaxScaler()
    feature_cols = ['Age', 'Arrival_Mode', 'Heart_Rate', 'Pain_Scale', 'Primary_Symptom']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print("   ✅ Label encoding + normalization hoàn thành")
    return df


# ── CHIA TRAIN/TEST ───────────────────────────────────────────────────────────
def split_data(df):
    print("\n✂️  Đang chia dữ liệu train/test (70/30)...")
    feature_cols = ['Age', 'Arrival_Mode', 'Heart_Rate', 'Pain_Scale', 'Primary_Symptom']
    X = df[feature_cols].values
    y = df['Routing_Class'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu")
    return X_train, X_test, y_train, y_test


# ── TRAINING MODEL ────────────────────────────────────────────────────────────
def train_model(reg_rate, X_train, y_train):
    print(f"\n🤖 Đang train Logistic Regression (reg_rate={reg_rate})...")

    # Log hyperparameter vào MLflow
    mlflow.log_param("regularization_rate", reg_rate)
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("test_size", 0.30)
    mlflow.log_param("random_state", 42)

    model = LogisticRegression(
        C=1 / reg_rate,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ).fit(X_train, y_train)

    print("   ✅ Model đã được train!")
    return model


# ── ĐÁNH GIÁ MODEL ────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    print("\n📊 Đang đánh giá model...")

    y_hat    = model.predict(X_test)
    y_scores = model.predict_proba(X_test)

    # ── Tính các metrics ──
    acc       = accuracy_score(y_test, y_hat)
    recall    = recall_score(y_test, y_hat)
    precision = precision_score(y_test, y_hat)
    f1        = f1_score(y_test, y_hat)

    # AUC - dùng scores của class 1
    try:
        auc = roc_auc_score(y_test, y_scores[:, 1])
    except Exception:
        auc = roc_auc_score(y_test, y_scores[:, 0])

    # In kết quả
    print(f"\n   {'Metric':<20} {'Giá trị':<10}")
    print(f"   {'-'*30}")
    print(f"   {'Accuracy':<20} {acc:.4f}")
    print(f"   {'Recall (Sensitivity)':<20} {recall:.4f}")
    print(f"   {'Precision':<20} {precision:.4f}")
    print(f"   {'F1-Score':<20} {f1:.4f}")
    print(f"   {'AUC-ROC':<20} {auc:.4f}")

    # ── Log metrics vào MLflow ──
    mlflow.log_metric("Accuracy",  acc)
    mlflow.log_metric("Recall",    recall)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("F1_Score",  f1)
    mlflow.log_metric("AUC",       auc)

    # ── Log Confusion Matrix ──
    cm = confusion_matrix(y_test, y_hat)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Fast-Track (0)', 'Regular (1)'])
    ax.set_yticklabels(['Fast-Track (0)', 'Regular (1)'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color='black' if cm[i, j] < cm.max() / 2 else 'white')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig_cm.tight_layout()
    fig_cm.savefig("Confusion-Matrix.png", dpi=100)
    mlflow.log_artifact("Confusion-Matrix.png")
    plt.close(fig_cm)

    # ── Log ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_scores[:, 1])
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    ax_roc.plot(fpr, tpr, color='#E53935', linewidth=2,
                label=f'Model (AUC={auc:.3f})')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curve - ER Triage Classification', fontsize=13)
    ax_roc.legend(fontsize=11)
    ax_roc.grid(True, alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig("ROC-Curve.png", dpi=100)
    mlflow.log_artifact("ROC-Curve.png")
    plt.close(fig_roc)

    # Log classification report
    report = classification_report(y_test, y_hat,
                                   target_names=['Fast-Track', 'Regular'])
    print(f"\n📋 Classification Report:\n{report}")
    mlflow.log_text(report, "classification_report.txt")

    print("   ✅ Đã log tất cả metrics, biểu đồ vào MLflow")


# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression cho ER Triage Classification"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Đường dẫn đến file CSV training data"
    )
    parser.add_argument(
        "--reg_rate",
        type=float,
        default=0.01,
        help="Regularization rate (default: 0.01)"
    )
    return parser.parse_args()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    main(args)
