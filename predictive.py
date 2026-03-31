import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("ai4i2020.csv")

# Keep required columns
data = data[[
    "Type",
    "Air temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]]

# Convert machine type
data["Type"] = data["Type"].map({"L":0, "M":1, "H":2})

# Remove missing values
data = data.dropna()

# Features and target
X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


print("\nEnter Machine Sensor Data\n")

# Machine inputs
machine_type = input("Type (L/M/H): ").upper()
air_temp = float(input("Air Temperature K: "))
rpm = float(input("Rotational Speed rpm: "))
torque = float(input("Torque Nm: "))
tool_wear = float(input("Tool Wear min: "))

# Convert type
type_value = {"L":0, "M":1, "H":2}[machine_type]

# Prediction
new_data = pd.DataFrame([[type_value, air_temp, rpm, torque, tool_wear]],
                        columns=X.columns)

prediction = model.predict(new_data)

# Result
if prediction[0] == 0:
    print("\nMachine Health: HEALTHY")
else:
    print("\nMachine Health: FAILURE RISK")