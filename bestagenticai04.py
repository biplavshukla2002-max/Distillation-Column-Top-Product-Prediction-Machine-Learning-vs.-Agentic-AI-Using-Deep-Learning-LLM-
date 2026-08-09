# =====================================================
#  AGENTIC AI FOR DISTILLATION COLUMN
# =====================================================
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from google import genai
# =====================================================
# AGENTIC AI SETUP
# =====================================================
client = genai.Client(
    api_key="key"
)
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="hello, I am an AI agent for monitoring a chemical distillation column. " \
    "Please provide me with the molar flow rate, and" \
    " I will predict the top product purity and give recommendations."
)
print(response.text)

# =====================================================
# DATASET
# =====================================================

data = pd.read_excel(
    r"C:\Users\user\Desktop\python code test.xlsx",
    sheet_name="MAIN"
)

# =====================================================
# PREPARE SCALER
# =====================================================

X = data[['molar flow rate']].values

scaler = StandardScaler()
scaler.fit(X)

# =====================================================
# LOAD TRAINED DEEP LEARNING MODEL
# =====================================================

model = load_model("Distillation_DeepLearning_Model.keras")
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
print("\n==========================================")
print(" DISTILLATION COLUMN AI AGENT ")
print("==========================================")

# =====================================================
# AI AGENT LOOP
# =====================================================

while True:

    value = input("\nEnter Molar Flow Rate (or type 'exit'): ")

    if value.lower() == "exit":
        print("\nAI Agent Closed.")
        break

    flow_rate = float(value)

    # Scale input
    sample = scaler.transform([[flow_rate]])

    # Predict
    prediction = model.predict(sample, verbose=0)[0][0]
    print("\n----------- AI Prediction -----------")
    print(f"Molar Flow Rate : {flow_rate:.2f}")
    print(f"Predicted Top Product : {prediction:.4f}")
    # =====================================================
    # AI DECISION MAKING
    # =====================================================
    print("\n----------- Agent Recommendation -----------")
    if prediction > 1:
        status = "invalid prediction"

        print("Status :", status)
        print("Recommendation :")
        print("- predicated mole fraction is greater than 1.")
        print("- Mole fraction cannot exceed 1.")
        print("- check the model predication or input conditions.")

    elif prediction >= 0.98:
        status = "excellent"

        print("Status :", status)
        print("Recommendation :")
        print("- top product purity is very high.")
        print("- maitain current operating conditions.")
        print("- no corrective action required.")

    elif prediction >= 0.95:
        status = "good"

        print("Status :", status)
        print("Recommendation :")
        print("- top product purity is good.")
        print("- continue monitoring process.")
        print("- small optimizations may improve purity.")

    elif prediction >= 0.90:
        status = "moderate"
    
        print("Status :", status)
        print("Recommendation :")
        print("- top product purity is moderate.")
        print("- increase process monitoring.")
        print("- optimize molar flow rate if required.")
        print("- inspect operating parameters.")
        

    else:
        status = "Poor"
        print("Status :", status)
        print("Recommendation :")
        print("- top product purity is below acceptable levels.")
        print("- review operating conditions.")
        print("- Recalibrate process if required.")

    print("------------------------------------------")

    # =====================================================
    # LLM ANALYSIS using Gemini 3.6
    # =====================================================

    prompt = f"""
    You are an AI assistant for monitoring a chemical distillation column.

    Molar Flow Rate: {flow_rate}
    Predicted Top Product: {prediction:.4f}
    Status: {status}
    Analyze the distillation column from an engineering perspective.
    Explain:
    1. What this prediction means.
    2. Whether the process is stable.
    3. Recommended operator action.
    4. Whether the process appears stable based ONLY on the available data.
    5. What operating parameters should be monitored.
    6. What operator action may be appropriate if product quality is low.
    7. Do NOT recommend changing operating conditions unless sufficient
       process information is available.

    Important:
    - A mole fraction cannot be greater than 1.0.
    - If prediction > 1.0, identify it as a physically invalid/model prediction
    and recommend checking the model, scaling, or input data.
    - Do not treat a model prediction as a measured value.
    - Do not invent temperature, pressure, reflux ratio, composition, or other
    process measurements that were not provided.

    Keep the answer under 100 words.
    """

    try:
        response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

        print("\n=========== LLM ANALYSIS ===========")
        print(response.text)
        print("====================================")

    except Exception as e:
        print("\nLLM Error:", e)