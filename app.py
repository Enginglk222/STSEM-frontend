from flask import Flask, request, jsonify, url_for, send_file
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib
import matplotlib.pyplot as plt
import openai
from flask_cors import CORS
import os
from openai import OpenAI, completions
from openai import RateLimitError, OpenAIError  # Changed import here

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)
openai.api_key = 'sk-proj-TGO34OnkIB1KiVcMd5IST3BlbkFJij15LZVdnQjR8KDFiPW3'
shap.initjs()

# Load and preprocess data
data = pd.read_csv("loan_data_1.csv")
data.drop(columns=['Loan_ID', 'Unnamed: 0'], axis=1, inplace=True)

# Fill missing values
categoricals_nulls = ["Gender", "Dependents", "Education", "Credit_History", "Self_Employed"]
for col in categoricals_nulls:
    vals = data[col].mode().values[0]
    data[col].fillna(vals, inplace=True)

numericals_nulls = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
for col in numericals_nulls:
    vals = data[col].median()
    data[col].fillna(vals, inplace=True)

# Encoding categorical features
categoricals = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
                "Credit_History", "Property_Area", "Loan_Status"]
numericals = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

le = LabelEncoder()
for col in categoricals:
    data[col] = le.fit_transform(data[col])

# Prepare data for modeling
y = data["Loan_Status"]
X = data.drop("Loan_Status", axis=1)

# SMOTE for oversampling
smote = SMOTE(sampling_strategy="all")
X_sm, y_sm = smote.fit_resample(X, y)

data = pd.concat([X_sm, y_sm], axis=1)

# Split the dataset
y = data.Loan_Status
X = data.drop("Loan_Status", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def process():
    return "Welcome to the Loan Prediction App!"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    observation_index = input_data.get("index")  # Kullanıcının girdiği indeks

    # Kullanıcının istediği gözlemi alma
    observation = X_train.iloc[observation_index]

    # Tahmin yapma
    prediction = model.predict([observation])[0]  # Tek gözlemle tahmin

    # SHAP değerlerini hesaplama
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(observation)

    # Waterfall grafiği
    plt.figure()
    shap.waterfall_plot(shap.Explanation(values=shap_values[prediction], 
                                         base_values=explainer.expected_value[prediction], 
                                         data=observation, 
                                         feature_names=X_train.columns), max_display=len(X_train.columns))
    
    plot_path = f'static/waterfall_plot_{observation_index}.png'
    plt.tight_layout()
    plt.savefig(plot_path)  # Grafiği kaydet
    plt.close()

    plot_url = url_for('serve_static', filename=os.path.basename(plot_path), _external=True)
                       
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
            {
            "role": "user",
            "content": "Loan status prediction: {prediction}\n\nSHAP values: {shap_values}\n\nBu grafik kredi alabilirlik (yes/no) tahmini için oluşturulan ML modelinde SHAP kullanılarak bir kişi için local explainability sonucu. Bu grafiği işin uzmanı olmayan bu kişinin anlayabileceği şekilde öznitelikleri dikkate alarak yorumlar mısın? Yorumlarının yanına bu kişiye özel önerini de ekle\n\n![Waterfall Plot]({plot_url})"
            }
        ],
        max_tokens=300
    )

        interpretation =response.choices[0].message.content
    except RateLimitError as e:
        interpretation = "API kotası aşıldı. Lütfen daha sonra tekrar deneyin."
    except OpenAIError as e:
        interpretation = f"OpenAI API hatası: {str(e)}"

    return jsonify({
        "prediction": int(prediction),
        "waterfall_plot": plot_url,
        "interpretation": interpretation
    })

# Bu endpoint, statik dosyaları sunar
@app.route('/static/<filename>')
def serve_static(filename):
    return send_file(f'static/{filename}', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
