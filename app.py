from flask import Flask, request, jsonify, send_file, url_for
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
import base64
import requests

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

api_key = 'sk-proj-BsFMuuMiBkGYEuHImVaBT3BlbkFJlu9bbZ5c7r5sENrBTnjq'

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
                                         feature_names=X_train.columns), max_display=len(X_train.columns), show = False)
    
    plot_path = f'static/waterfall_plot_{observation_index}.png'
    plt.tight_layout()
    plt.savefig(plot_path)  # Grafiği kaydet
    plt.close()
    plot_url = url_for('serve_static', filename=os.path.basename(plot_path), _external=True)

    # Görüntüyü base64 formatına dönüştürme
    with open(plot_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "Sen finansal okuryazar olan, grafik yorumlayabilen ve finansal okuryazar olmayan kişilere açıklayan bir asistansın."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Bu grafik kredi alabilirlik (yes/no) tahmini için oluşturulan ML modelinde, SHAP yönteminin bir kişinin (lokal) shapley değerlerini kullanarak yaptığı bir waterfall grafiği. Kısaca Grafiğe bakarak özniteliklerin etkisini açıkla. Sayısal veri kullanma. Özniteliklere göre bu kişi özelinde  öneri ver."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 3000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        interpretation = response.json()["choices"][0]["message"]["content"]
    else:
        interpretation = f"OpenAI API hatası: {response.json()}"

    return jsonify({
        "prediction": int(prediction),
        "waterfall_plot": img_base64,
        "plot_url": plot_url,
        "interpretation": interpretation
    })

# Bu endpoint, statik dosyaları sunar
@app.route('/static/<filename>')
def serve_static(filename):
    return send_file(f'static/{filename}', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
