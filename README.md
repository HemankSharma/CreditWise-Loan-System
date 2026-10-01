# 💳 CreditWise Loan System

### Machine Learning-Based Loan Approval Prediction

CreditWise is a **Machine Learning classification project** that predicts whether a loan application should be **approved or rejected** based on applicant information such as income, credit score, loan amount, savings, DTI ratio, age, employment status, education, and other applicant characteristics.

The project follows a complete ML workflow — from **data preprocessing and exploratory data analysis to feature engineering, model training, evaluation, and model comparison**.

---

## 📌 Project Overview

Loan approval is an important decision for financial institutions because approving high-risk applicants can result in financial losses, while rejecting reliable applicants can result in lost business opportunities.

The goal of this project is to build a machine learning model capable of learning patterns from historical loan application data and predicting the `Loan_Approved` outcome.

### 🎯 Objective

Build a classification model that can:

* Predict whether a loan application will be approved or rejected.
* Analyze the factors associated with loan approval.
* Compare multiple machine learning classification algorithms.
* Evaluate models using appropriate classification metrics.
* Reduce both **False Positives** and **False Negatives** as much as possible.

---

# 🧠 Machine Learning Workflow

The project follows this pipeline:

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Missing Value Handling
     ↓
Exploratory Data Analysis
     ↓
Categorical Encoding
     ↓
Correlation Analysis
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Baseline Models
     ↓
Feature Engineering
     ↓
Additional Models
     ↓
Decision Tree Pruning
     ↓
Model Evaluation
     ↓
Best Model Selection
```

---

# 📂 Dataset

The project uses:

```text
loan_approval_data_minor_project_1.csv
```

The dataset contains applicant-related information and a target variable:

```text
Loan_Approved
```

which represents whether the loan was approved.

### Important Features Used

The notebook works with features including:

* `Applicant_ID`
* `Gender`
* `Age`
* `Education_Level`
* `Employment_Status`
* `Marital_Status`
* `Applicant_Income`
* `Coapplicant_Income`
* `Loan_Amount`
* `Loan_Purpose`
* `Property_Area`
* `Credit_Score`
* `DTI_Ratio`
* `Savings`
* `Employer_Category`
* `Loan_Approved`

> `Applicant_ID` is removed before model training because it is an identifier rather than a meaningful predictive feature.

---

# 🧹 1. Data Cleaning

The first step is to understand the structure and quality of the dataset.

The notebook performs:

* Dataset inspection
* Data type analysis
* Missing-value detection
* Statistical summary
* Separation of numerical and categorical columns

### Missing Value Handling

Different strategies are used for numerical and categorical variables.

#### Numerical Features

Missing numerical values are filled using the **mean**:

```python
SimpleImputer(strategy="mean")
```

#### Categorical Features

Missing categorical values are filled using the **most frequent value**:

```python
SimpleImputer(strategy="most_frequent")
```

This ensures that the dataset contains no missing values before model training.

---

# 📊 2. Exploratory Data Analysis

EDA is performed to understand the dataset and identify relationships between applicant characteristics and loan approval.

### Analysis performed

The notebook analyzes:

* Loan approval class distribution
* Gender distribution
* Education-level distribution
* Applicant income distribution
* Coapplicant income distribution
* Credit score
* DTI ratio
* Savings
* Loan amount
* Age

### Visualizations

The project uses:

* Pie charts
* Bar plots
* Histograms
* Box plots
* Correlation heatmaps

One important analysis focuses on the relationship between **Credit Score** and **Loan Approval**.

---

# 🔤 3. Categorical Encoding

Machine learning algorithms require numerical input, so categorical variables are converted into numerical representations.

Two encoding techniques are used.

## Label Encoding

`LabelEncoder` is used for:

* `Education_Level`
* `Loan_Approved`

Example:

```python
le = LabelEncoder()

df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])
```

## One-Hot Encoding

One-hot encoding is used for categorical variables where the categories do not represent an inherent numerical order.

Features encoded include:

```text
Employment_Status
Marital_Status
Loan_Purpose
Property_Area
Gender
Employer_Category
```

The project uses:

```python
OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)
```

Using `drop="first"` helps avoid redundant dummy variables.

---

# 🔥 4. Correlation Analysis

A correlation matrix is generated to understand relationships between numerical variables and the target variable.

```python
corr_matrix = num_cols.corr()
```

The project then examines the correlation of every numerical feature with:

```text
Loan_Approved
```

A heatmap is also created to visually inspect relationships between variables.

---

# ✂️ 5. Train-Test Split

The dataset is divided into training and testing sets.

```python
train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

### Split

```text
80% → Training
20% → Testing
```

`random_state=42` is used to make the split reproducible.

---

# 📏 6. Feature Scaling

The project uses **StandardScaler** to standardize numerical features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

---

# 🤖 7. Machine Learning Models

Several classification algorithms are implemented and compared.

## 1. Logistic Regression

Logistic Regression is used as one of the baseline classification models.

```python
LogisticRegression()
```

It predicts the probability of an applicant belonging to the approval class.

---

## 2. K-Nearest Neighbors

The project also implements **KNN** with:

```python
KNeighborsClassifier(n_neighbors=5)
```

KNN makes predictions based on the nearest observations in the feature space.

---

## 3. Gaussian Naive Bayes

The project uses:

```python
GaussianNB()
```

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

---

## 4. Decision Tree Classifier

Decision Tree classification is also implemented.

The project explores **pre-pruning** using:

```python
DecisionTreeClassifier(max_depth=5)
```

This limits the maximum depth of the tree and helps control model complexity.

---

# 🌳 Decision Tree Pruning

The notebook additionally explores **post-pruning** using Cost Complexity Pruning.

The pruning path is obtained using:

```python
cost_complexity_pruning_path()
```

Different values of `ccp_alpha` are evaluated to investigate tree complexity.

The project also explores combining:

* Maximum depth
* Cost-complexity pruning

This provides a practical demonstration of controlling decision-tree complexity.

---

# 🛠️ 8. Feature Engineering

Additional features are created to capture nonlinear relationships.

### DTI Ratio Squared

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
```

### Credit Score Squared

```python
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

The original:

```text
Credit_Score
DTI_Ratio
```

features are subsequently removed from the modeling dataset in this stage.

Feature engineering allows the models to work with transformed representations of important variables.

---

# 📈 9. Model Evaluation

The project evaluates classification models using:

### Accuracy

Measures the overall percentage of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive cases were actually positive.

```text
Precision = TP / (TP + FP)
```

Precision is particularly important in this project because a **False Positive** represents an applicant predicted as approved when the actual class is not approved.

### Recall

Measures how many actual positive cases were correctly identified.

```text
Recall = TP / (TP + FN)
```

Recall is also important because a **False Negative** represents an applicant who could be approved but is predicted as rejected.

### F1 Score

F1 Score provides a balance between precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### Confusion Matrix

The project also uses a confusion matrix to examine:

```text
True Positive
True Negative
False Positive
False Negative
```

---

# 🏆 Reported Model Performance

The notebook reports the following results for its selected Decision Tree model:

| Metric    |      Score |
| --------- | ---------: |
| Precision | **82.35%** |
| Recall    | **91.80%** |
| F1 Score  | **86.82%** |
| Accuracy  | **91.50%** |

### Reported Confusion Matrix

```text
[[127  12]
 [  5  56]]
```

This corresponds to:

```text
                Predicted
              Negative Positive

Actual Negative   127      12
Actual Positive     5      56
```

---

# 📌 Why These Metrics Matter

For a loan approval system, accuracy alone does not provide the complete picture.

The notebook gives particular importance to:

### 1️⃣ Precision

The project focuses on reducing **False Positives**, where an application is predicted as approved incorrectly.

Reported precision:

```text
82.35%
```

### 2️⃣ Recall

The project also considers **False Negatives**, where an application that belongs to the positive class is incorrectly rejected.

Reported recall:

```text
91.80%
```

### 3️⃣ F1 Score

F1 Score provides a balance between precision and recall.

Reported F1 Score:

```text
86.82%
```

---

# 🧰 Technologies & Libraries

The project is implemented using **Python**.

### Programming Language

* Python

### Data Processing

* NumPy
* Pandas

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### ML Algorithms

* Logistic Regression
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Decision Tree Classifier

### Preprocessing Techniques

* Simple Imputation
* Label Encoding
* One-Hot Encoding
* Standard Scaling

### Feature Engineering

* Polynomial-style squared features
* DTI Ratio transformation
* Credit Score transformation

---

# 📁 Project Structure

A recommended GitHub structure for this project is:

```text
CreditWise-Loan-System/
│
├── 📓 CreditWise Loan System_code.ipynb
│
├── 📊 loan_approval_data_minor_project_1.csv
│
├── 📄 README.md
│
└── 📁 images/
    └── project_visualizations/
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

## 2. Navigate to the Project

```bash
cd CreditWise-Loan-System
```

## 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Notebook

Open:

```text
CreditWise Loan System_code.ipynb
```

Make sure the dataset:

```text
loan_approval_data_minor_project_1.csv
```

is available in the expected working directory.

---

# 🔍 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Data preprocessing
* Missing-value imputation
* Exploratory Data Analysis
* Categorical feature encoding
* Label Encoding
* One-Hot Encoding
* Correlation analysis
* Train-test splitting
* Feature scaling
* Feature engineering
* Binary classification
* Logistic Regression
* KNN
* Naive Bayes
* Decision Trees
* Pre-pruning
* Post-pruning
* Cost Complexity Pruning
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Accuracy
* Model comparison

---

# 💡 Project Highlights

### 🔹 Complete ML Pipeline

The project covers the complete journey from raw data to model evaluation.

### 🔹 Multiple Models

Rather than relying on a single algorithm, multiple classification techniques are explored and compared.

### 🔹 Feature Engineering

The project creates transformed features such as:

```text
DTI_Ratio_sq
Credit_Score_sq
```

to capture additional relationships.

### 🔹 Decision Tree Optimization

Both pre-pruning and cost-complexity post-pruning concepts are explored.

### 🔹 Business-Oriented Evaluation

The project does not rely solely on accuracy and considers the consequences of **False Positives and False Negatives** in loan decisions.

---

# ⚠️ Important Note

This project is an **educational machine learning implementation** based on the dataset and workflow contained in the notebook. The reported metrics are specific to the dataset, preprocessing pipeline, train-test split, and implementation used in the notebook.

The model should not be treated as a production-ready financial decision system without additional validation, fairness analysis, probability calibration, external testing, monitoring, and appropriate financial/regulatory review.

---

# 🔮 Future Improvements

Potential extensions to make CreditWise more production-oriented include:

* Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
* Cross-validation
* ROC-AUC and PR-AUC analysis
* Feature importance analysis
* SHAP-based explainability
* Probability calibration
* Handling class imbalance if required
* Model serialization using Joblib
* Building a prediction interface using Streamlit
* REST API using FastAPI/Flask
* Automated preprocessing pipeline using `Pipeline` and `ColumnTransformer`
* Model monitoring and drift detection
* Fairness and bias analysis
* Testing on an independent dataset

---

# 👨‍💻 Author

**Hemank Sharma**

B.Tech — Computer Science Engineering
Specialization: Artificial Intelligence & Machine Learning

---

# ⭐ If You Find This Project Useful

If you found this project helpful for learning Machine Learning, consider giving the repository a ⭐ on GitHub.
