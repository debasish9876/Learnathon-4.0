from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load model
model = joblib.load(r"A:\fmodel\models\fraud_rf_model.pkl")

# Load data
data = pd.read_csv(r"A:\fmodel\Auto_Insurance_Fraud_Claims_File01.csv")

@app.route('/')
def home():
    return render_template("index.html")  # ✅ Make sure this file exists in 'templates/'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # 🚨 You must match these keys to your form field names
            features = [
                float(request.form['age']),
                float(request.form['policy_number']),
                float(request.form['vehicle_price']),
                float(request.form['accident_area'])
            ]
            prediction = model.predict([features])[0]
            result = "Fraudulent" if prediction else "Not Fraudulent"
            return render_template("predict.html", prediction=result)
        except Exception as e:
            return render_template("predict.html", prediction=f"❌ Error: {e}")
    return render_template("predict.html")

@app.route('/visualize')
def visualize():
    plots_dir = os.path.join("static", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "heatmap.png"))
    plt.close()

    # 2. Fraud Case Count
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="fraud_reported", palette="coolwarm")
    plt.title("Fraud Cases Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fraud_case_distribution.png"))
    plt.close()

    # 3. Fraud by Policy State
    plt.figure(figsize=(9, 6))
    sns.countplot(data=data, x="policy_state", hue="fraud_reported", palette="Set2")
    plt.title("Fraud Cases by Policy State")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fraud_vs_state.png"))
    plt.close()

    return render_template("visualize.html")  # ✅ Make sure this file exists
