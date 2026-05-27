# DP-100 Project: ER Triage Classification (Phân loại Bệnh nhân Khẩn cấp)

## 🏥 Bài toán

Phân loại bệnh nhân cấp cứu vào 2 luồng xử lý:
- **Class 0 → Fast-Track**: Bệnh nhẹ, xử lý nhanh
- **Class 1 → Regular/Resuscitation**: Bệnh nặng, cần chăm sóc đặc biệt

> ⚠️ **Quan trọng**: False Negative (bỏ sót bệnh nặng) nguy hiểm hơn nhiều so với False Positive.  
> Tối ưu **Recall** song song với Accuracy!

---

## 📊 Dataset: ER_Triage_Dataset.csv

| Feature | Kiểu | Mô tả |
|---------|------|-------|
| Age | int | Tuổi bệnh nhân (1-94) |
| Arrival_Mode | string | Walk-in / Ambulance |
| Heart_Rate | int | Nhịp tim (bpm) |
| Pain_Scale | int | Mức đau (1-10) |
| Primary_Symptom | string | Laceration / Abdominal Pain / Fever / Orthopedic / Chest Pain |
| **Routing_Class** | int | **Target: 0=Fast-Track, 1=Regular** |

**Stats**: 500 mẫu | Không có missing values | Imbalanced (73% class 1)

---

## 🗂️ Cấu trúc Project

```
project/
├── Task1-AutoML/
│   ├── Task1_AutoML_Classification.ipynb    ← Milestone 1
│   └── ERTriage-Data/
│       ├── ER_Triage_Dataset.csv
│       └── MLTable                          ← Schema definition
│
├── Task2-CommandJob-MLflow/
│   ├── Task2_CommandJob_MLflow.ipynb        ← Milestone 2 & 3
│   └── src/
│       ├── ER_Triage_Dataset.csv
│       └── train-model-mlflow.py            ← Script với MLflow logging
│
└── Task3-Hyperparameter/
    ├── Task3_Hyperparameter_Sweep.ipynb     ← Milestone 4
    └── src/
        ├── ER_Triage_Dataset.csv
        └── train.py                          ← Script cho sweep
```

---

## 🚀 Hướng dẫn Thực hiện

### Phase 1: Setup Azure (Một lần duy nhất)

```
Azure Portal → Azure Machine Learning → Create Workspace
├── Workspace Name: [unique-name]
├── Resource Group: rg-dp100-project
└── Region: East US

Compute:
├── Instance: aml-compute (Standard_DS11_v2)
└── Cluster:  aml-cluster (0-2 nodes, Standard_DS11_v2)
```

### Phase 2: Clone & Install

```bash
# Trên Compute Instance terminal
pip uninstall azure-ai-ml -y
pip install azure-ai-ml

git clone https://github.com/Thanshidha/DP-100-Project.git azure-ml-project
```

### Phase 3: Chạy từng Milestone theo thứ tự

| Milestone | File | Mục tiêu |
|-----------|------|---------|
| **M1: AutoML** | `Task1-AutoML/Task1_AutoML_Classification.ipynb` | Tự động tìm best model |
| **M2: Command Job** | `Task2-CommandJob-MLflow/Task2_CommandJob_MLflow.ipynb` | Chạy script trên cluster |
| **M3: MLflow** | (cùng notebook M2) | Theo dõi params/metrics/artifacts |
| **M4: Sweep** | `Task3-Hyperparameter/Task3_Hyperparameter_Sweep.ipynb` | Tối ưu reg_rate |

---

## 📐 Preprocessing Pipeline

```python
# 1. Label Encoding (categorical → numeric)
LabelEncoder: Arrival_Mode, Primary_Symptom, Routing_Class

# 2. Normalization
MinMaxScaler: Age, Arrival_Mode, Heart_Rate, Pain_Scale, Primary_Symptom

# 3. Train/Test Split
train_test_split(test_size=0.30, random_state=42, stratify=y)
```

---

## 🔧 Hyperparameter Sweep

```
Search Space: reg_rate ∈ {0.001, 0.01, 0.1, 1, 10}
Algorithm: Grid Search
Primary Metric: training_accuracy_score (Maximize)
Max Trials: 5 | Concurrent: 2
```

---

## 📊 MLflow Metrics được log

| Metric | Ý nghĩa |
|--------|---------|
| Accuracy | Tỷ lệ dự đoán đúng tổng thể |
| **Recall** | % bệnh nặng được phát hiện đúng ← QUAN TRỌNG NHẤT |
| Precision | % dự đoán Regular thực sự là Regular |
| F1-Score | Cân bằng Precision & Recall |
| AUC-ROC | Khả năng phân biệt 2 class |

### Artifacts:
- `ROC-Curve.png` → Biểu đồ ROC
- `Confusion-Matrix.png` → Ma trận nhầm lẫn
- `classification_report.txt` → Báo cáo chi tiết

---

## ⚙️ Environment

- **Azure ML SDK**: `azure-ai-ml >= 1.5`
- **Environment**: `AzureML-sklearn-1.5@latest`
- **Python**: 3.10
- **Libraries**: scikit-learn, mlflow, pandas, numpy, matplotlib
