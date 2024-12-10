from flask import Flask, render_template, request, jsonify
from deployment.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np
app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        tv = float(request.form["tv"])
        radio = float(request.form["radio"])
        newspaper = float(request.form["newspaper"])
        
        data = np.array([[tv, radio, newspaper]])
        prediction_pipeline = PredictionPipeline()
        predicted_sales = prediction_pipeline.predict(data)
    
        return render_template("predict.html", predicted_sales = predicted_sales)
    else:
        return render_template("input.html")
    
if __name__ == "__main__":
    app.run(debug=True)