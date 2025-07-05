from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import json
import os

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Base directory and model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Load model
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'traffic_stack_model.joblib'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load feature columns
try:
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'r') as f:
        feature_columns = json.load(f)
    print("Feature columns loaded successfully")
except Exception as e:
    print(f"Error loading feature columns: {e}")
    feature_columns = None

# Load scaler
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Load metrics
try:
    with open(os.path.join(MODEL_DIR, 'performance_metrics.json'), 'r') as f:
        metrics = json.load(f)
    print("Performance metrics loaded successfully")
except Exception as e:
    print(f"Error loading performance metrics: {e}")
    metrics = None

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics, model_loaded=model is not None)

@app.route('/dataset')
def dataset():
    try:
        dataset_path = os.path.join(BASE_DIR, 'Metro_Interstate_Traffic_Volume.csv')
        data = pd.read_csv(dataset_path)
        sample_data = data.head(100)
        return render_template('dataset.html',
                               data=sample_data.to_html(classes='table table-striped', index=False))
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

@app.route('/analysis')
def analysis():
    images = {
        'Traffic by Hour': 'static/images/traffic_by_hour.png',
        'Actual vs Predicted': 'static/images/actual_vs_predicted.png',
        'Residual Plot': 'static/images/residual_plot.png',
        'Feature Importances': 'static/images/feature_importances.png'
    }
    return render_template('analysis.html', images=images)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    prediction_text = None
    traffic_alert = None
    historical_comparison = None
    smart_tip = None
    error_msg = None

    if request.method == 'POST':
        if not all([model, scaler, feature_columns]):
            error_msg = "⚠️ Model assets not fully loaded. Check server logs."
        else:
            try:
                hour = int(request.form['hour'])
                weekday = int(request.form['weekday'])

                input_data = {
                    'temp': float(request.form['temperature']),
                    'rain_1h': float(request.form['rain']),
                    'snow_1h': float(request.form['snow']),
                    'clouds_all': float(request.form['cloud_cover']),
                    'hour': hour,
                    'weekday': weekday,
                    'month': int(request.form['month']),
                    'is_weekend': 1 if request.form.get('is_weekend') == 'yes' else 0,
                    'is_peak_hour': 1 if request.form.get('is_peak_hour') == 'yes' else 0,
                    'is_daytime': 1 if request.form.get('is_daytime') == 'yes' else 0,
                    'holiday_None': 1
                }

                weather = request.form['weather']
                input_data[weather] = 1

                input_df = pd.DataFrame([input_data])
                for col in feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_columns]

                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                prediction_text = f"🚗 Predicted Traffic Volume: {prediction:,.0f} vehicles"

                if prediction < 2000:
                    traffic_alert = "🟢 Light traffic expected. Ideal time to travel."
                elif prediction < 4000:
                    traffic_alert = "🟡 Moderate traffic. Plan accordingly."
                else:
                    traffic_alert = "🔴 Heavy traffic expected. Consider delaying your trip or using alternate routes."

                historical_avg = 3200
                diff = prediction - historical_avg
                pct_change = (diff / historical_avg) * 100
                if diff > 0:
                    historical_comparison = f"📈 {abs(pct_change):.1f}% higher than average for this hour."
                else:
                    historical_comparison = f"📉 {abs(pct_change):.1f}% lower than average for this hour."

                if hour in [7, 8, 9, 17, 18]:
                    smart_tip = "💡 Try leaving 30 minutes earlier or later to avoid peak congestion."
                elif prediction > 4500:
                    smart_tip = "💡 Consider using public transport or alternate routes if available."
                elif prediction < 2500:
                    smart_tip = "✅ Roads look clear — a great time to travel."

            except Exception as e:
                error_msg = f"Prediction error: {str(e)}"
                print(error_msg)

    return render_template('predict.html',
                           prediction=prediction,
                           prediction_text=prediction_text,
                           traffic_alert=traffic_alert,
                           historical_comparison=historical_comparison,
                           smart_tip=smart_tip,
                           error_msg=error_msg,
                           metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
