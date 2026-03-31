from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Initialize model implicitly. Since training is fast, train it on boot.
model = None
feature_columns = [
    "Type",
    "Air temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

def train_model():
    global model
    try:
        # Load dataset
        data = pd.read_csv("ai4i2020.csv")
        
        # Keep required columns
        columns_to_keep = feature_columns + ["Machine failure"]
        data = data[columns_to_keep]
        
        # Convert machine type
        data["Type"] = data["Type"].map({"L":0, "M":1, "H":2})
        
        # Remove missing values
        data = data.dropna()
        
        # Features and target
        X = data.drop("Machine failure", axis=1)
        y = data["Machine failure"]
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        print("Model trained successfully.")
    except Exception as e:
        print(f"Error training model: {e}")

# Call it upon startup
train_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained. Ensure ai4i2020.csv exists in the root directory."}), 500
        
    try:
        data = request.json
        
        m_type = data.get("machine_type", "L").upper()
        # Ensure type is valid
        if m_type not in ["L", "M", "H"]:
            m_type = "L"
            
        type_val = {"L":0, "M":1, "H":2}[m_type]
        air_temp = float(data.get("air_temperature", 0))
        rpm = float(data.get("rotational_speed", 0))
        torque = float(data.get("torque", 0))
        tool_wear = float(data.get("tool_wear", 0))
        
        new_data = pd.DataFrame(
            [[type_val, air_temp, rpm, torque, tool_wear]],
            columns=feature_columns
        )
        prediction = model.predict(new_data)
        result = int(prediction[0])
        
        prob = model.predict_proba(new_data)[0][1]
        
        return jsonify({"prediction": result, "probability": float(prob)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
