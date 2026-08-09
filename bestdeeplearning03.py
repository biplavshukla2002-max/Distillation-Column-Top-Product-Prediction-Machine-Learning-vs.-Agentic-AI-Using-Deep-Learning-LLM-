# =====================================================
# DEEP LEARNING MODEL 
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# LOAD DATA
# =====================================================

datapoints = pd.read_excel(
    r"C:\Users\user\Desktop\python code test.xlsx",
    sheet_name="MAIN"
)

# =====================================================
# INPUT AND OUTPUT
# =====================================================

X = datapoints[['molar flow rate']].values

# Change this to 'E2' if you want to predict Reboiler Energy
Y = datapoints[3].values

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
#  DEEP LEARNING MODEL
# =====================================================

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(1,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
# =====================================================
# TRAINING WITH EARLY STOPPING
# =====================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    Y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)
# =====================================================
# PREDICTION
# =====================================================
prediction = model.predict(X_test)
# =====================================================
# PERFORMANCE
# =====================================================
r2 = r2_score(Y_test, prediction)
mse = mean_squared_error(Y_test, prediction)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, prediction)
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
# =====================================================
# SAVE PREDICTIONS
# =====================================================
result = pd.DataFrame({
    "Actual": Y_test,
    "Predicted": prediction.flatten()
})
result.to_excel(
    "DeepLearning_Predictions.xlsx",
    index=False
)

print("\nPredictions saved successfully.")
# =====================================================
# ACTUAL VS PREDICTED
# =====================================================
plt.figure(figsize=(8,5))

plt.scatter(
    Y_test,
    prediction,
    color="royalblue"
)

plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    'r--'
)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted (Deep Learning)")
plt.grid(True)
plt.show()
# =====================================================
# SAVE MODEL
# =====================================================
model.save("Distillation_DeepLearning_Model.keras")
print("\nDeep Learning model saved successfully.")