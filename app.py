from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("car_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    present_price = float(request.form["present_price"])
    kms_driven = float(request.form["kms_driven"])
    fuel_type = int(request.form["fuel_type"])
    seller_type = int(request.form["seller_type"])
    transmission = int(request.form["transmission"])
    owner = int(request.form["owner"])

    car_data = np.array([[
        year,
        present_price,
        kms_driven,
        fuel_type,
        seller_type,
        transmission,
        owner
    ]])

    prediction = model.predict(car_data)

    predicted_price = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction=predicted_price
    )


if __name__ == "__main__":
    app.run(debug=True)  