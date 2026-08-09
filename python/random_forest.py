import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# =====================================================
# DATA
# =====================================================
datapoints = pd.read_excel(
    r"C:\Users\user\Desktop\python code test.xlsx",
    sheet_name="MAIN"
)
# =====================================================
#FEATURES AND TARGET VARIABLE
# =====================================================
X = datapoints[["molar flow rate"]]
Y = datapoints[3]

print("X shape:", X.shape)
print("Y shape:", Y.shape)
# =====================================================
# TRAIN-TEST SPLIT system
# =====================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
# =====================================================
# RANDOM FOREST MODEL
# =====================================================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, Y_train)
# =====================================================
# USER INPUT
# =====================================================
molar_flow = float(input("Enter Molar Flow Rate: "))

user_input = pd.DataFrame({
    "molar flow rate": [molar_flow]
})
# =====================================================
# PREDICTION
# =====================================================
prediction = model.predict(user_input)
print(f"Predicted Top Product (3): {prediction[0]:.4f}")
