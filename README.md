# Telecom-Customer-Churn-Prediction-using-Machine-Learning
Xây dựng Machine Learning model dự đoán khách hàng có khả năng churn dựa trên dữ liệu:  thông tin hợp đồng  mức sử dụng dịch vụ  chi phí hàng tháng  thời gian sử dụng dịch vụ
Telecom Customer Churn Prediction using Machine Learning
1. Project Overview
Problem Statement

Các công ty viễn thông thường gặp vấn đề Customer Churn – khách hàng rời bỏ dịch vụ để chuyển sang đối thủ.

Việc dự đoán sớm khách hàng có khả năng churn giúp doanh nghiệp:

Giữ chân khách hàng

Giảm chi phí marketing

Tăng doanh thu dài hạn

Project Goal

Xây dựng Machine Learning model dự đoán khách hàng có khả năng churn dựa trên dữ liệu:

thông tin hợp đồng

mức sử dụng dịch vụ

chi phí hàng tháng

thời gian sử dụng dịch vụ

Business Value

Nếu dự đoán chính xác:

gửi promotion

chăm sóc VIP customers

giảm churn rate

2. Tech Stack
Programming

Python

Data Processing

Pandas

NumPy

Machine Learning

Scikit-learn

XGBoost

LightGBM

Visualization

Matplotlib

Seaborn

Database

SQL

Optional (Portfolio mạnh hơn)

Streamlit Dashboard

SHAP Explainability

3. Project Architecture
telco-churn-prediction/
│
├── data/
│   ├── raw/
│   │     telco_churn.csv
│   │
│   └── processed/
│         cleaned_data.csv
│
├── notebooks/
│     01_data_exploration.ipynb
│     02_feature_engineering.ipynb
│     03_model_training.ipynb
│     04_model_evaluation.ipynb
│
├── src/
│
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│
├── models/
│     churn_model.pkl
│
├── dashboard/
│     streamlit_app.py
│
├── reports/
│     figures/
│
├── requirements.txt
├── README.md
└── main.py
4. Dataset

Dataset đề xuất:

Telco Customer Churn Dataset

Nguồn:

Kaggle

Các cột chính:

Feature	Meaning
gender	giới tính
SeniorCitizen	khách hàng cao tuổi
tenure	số tháng sử dụng
Contract	loại hợp đồng
MonthlyCharges	chi phí hàng tháng
TotalCharges	tổng chi phí
InternetService	loại internet
TechSupport	hỗ trợ kỹ thuật
Churn	khách rời mạng

Target:

Churn (Yes / No)
5. Data Cleaning

Các bước:

Handle Missing Values
TotalCharges -> convert to numeric

Fill missing:

median
Remove Duplicates
df.drop_duplicates()
Encode Categorical Variables

Label Encoding

One-hot Encoding

Ví dụ:

Contract
InternetService
PaymentMethod
6. Feature Engineering

Tạo thêm feature để model học tốt hơn.

Example Features
1️⃣ Customer Lifetime
tenure
2️⃣ Average Monthly Spend
TotalCharges / tenure
3️⃣ Contract Risk Level
Month-to-month -> high churn
4️⃣ Service Count
PhoneService
InternetService
StreamingTV
StreamingMovies
7. Exploratory Data Analysis (EDA)

Mục tiêu:

Hiểu pattern churn.

Phân tích cần làm
Churn Distribution
Churn rate
Churn by Contract Type
Month-to-month
1 year
2 year
Churn by Tenure

Khách mới dễ churn.

Churn by Monthly Charges

Chi phí cao -> churn cao.

Visualization

Countplot

Boxplot

Heatmap correlation

8. Model Training

Train nhiều model để so sánh.

Models

1️⃣ Logistic Regression

Baseline model.

2️⃣ Random Forest

3️⃣ XGBoost

4️⃣ LightGBM

Train Test Split
80% train
20% test
9. Model Evaluation

Metrics quan trọng:

Accuracy
Precision
Recall
F1-score
ROC-AUC

Trong churn prediction:

Recall quan trọng hơn

Vì:

miss churn customer = mất khách
10. Model Optimization

Dùng:

GridSearchCV

Tuning:

Random Forest
n_estimators
max_depth
min_samples_split
XGBoost
learning_rate
max_depth
n_estimators
11. Model Explainability

Sử dụng:

SHAP

Để hiểu:

feature nào ảnh hưởng churn nhất

Ví dụ:

Top features:

tenure

contract

monthly charges

tech support

12. Deployment (Optional)
Streamlit Dashboard

Hiển thị:

Customer input

Churn probability

Feature importance

Ví dụ:

Customer Info -> Predict Churn
13. Example Output

Model sẽ dự đoán:

Customer A -> 82% churn probability

Hệ thống có thể:

trigger retention campaign
14. Project Outcome

Kết quả kỳ vọng:

Xây dựng ML pipeline hoàn chỉnh

So sánh nhiều model

Hiểu hành vi churn của khách hàng
