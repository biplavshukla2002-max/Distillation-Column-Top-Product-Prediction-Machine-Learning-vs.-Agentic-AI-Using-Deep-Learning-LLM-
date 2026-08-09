
# ============================================================
# ml model selection
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# ============================================================
#  DATA
# ============================================================

file_add = r"C:\Users\user\Desktop\python code test.xlsx"

datapoint = pd.read_excel(
    file_add,
    sheet_name="MAIN"
)

# ============================================================
# 2. FEATURES AND TARGET VARIABLE
# ============================================================

# Input variable
X = datapoint[["molar flow rate"]]

# Target variable
Y = datapoint[3]

distillated = "Top Product (3)"
# ============================================================
# 3. ALL MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Polynomial Regression":
        make_pipeline(
            PolynomialFeatures(degree=3),
            LinearRegression()
        ),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Support Vector":
        SVR(kernel="rbf")
}
# ============================================================
# 4. TRAIN-TEST SPLIT SYSTEM
# ============================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. MODEL CHECKING
# ============================================================

output = []
print("MODEL PERFORMANCE")
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    r2 = r2_score(Y_test, prediction)
    mse = mean_squared_error(
        Y_test,
        prediction
    )
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        Y_test,
        prediction
    )  
    output.append({
        "Model": model_name,
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })
    print(f"\n{model_name}")
    print("-" * 40)
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.4f}")
# ============================================================
# 6.  MODEL SCORE
# ============================================================
comparison = pd.DataFrame(output)
print("MODEL COMPARISON TABLE")
print(
    comparison.round(4).to_string(index=False)
)
comparison.to_excel(
    "Model_Comparison.xlsx",
    index=False
)
print("\nComparison table saved as Model_Comparison.xlsx")
# ============================================================
# 7. MOST ACCURATE MODEL
# ============================================================
accu = comparison.loc[
    comparison["R2"].idxmax()
]
print("BEST MODEL")
print("Output      :", distillated)
print("Best Model  :", accu["Model"])
print("R²          :", round(accu["R2"], 4))
print("RMSE        :", round(accu["RMSE"], 4))
print("MAE         :", round(accu["MAE"], 4))
# ============================================================
# 8. recheck
# ============================================================
rechecking_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

cv_scores = cross_val_score(
    rechecking_model,
    X,
    Y,
    cv=5,
    scoring="r2"
)
print("RANDOM FOREST CROSS-VALIDATION")
print(
    "R² scores:",
    np.round(cv_scores, 4)
)
print(
    "Average R²:",
    round(cv_scores.mean(), 4)
)
# ============================================================
# 9. plotting
# ============================================================
flow_range = np.linspace(
    X["molar flow rate"].min(),
    X["molar flow rate"].max(),
    300
).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(
    X_test["molar flow rate"],
    Y_test,
    s=30,
    label="Actual Test Data"
)
for model_name, model in models.items():

    predicted_curve = model.predict(
        flow_range
    )

    plt.plot(
        flow_range,
        predicted_curve,
        linewidth=2,
        label=model_name
    )
plt.title(
    "Selecting best ML Model"
)
plt.xlabel("Molar Flow Rate")
plt.ylabel("Top Product Mole Fraction")
plt.grid(True)
plt.legend()
plt.show()

