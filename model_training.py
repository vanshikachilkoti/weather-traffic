import os
os.makedirs('model', exist_ok=True)
os.makedirs('static/images', exist_ok=True)  

#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

# DATA PREPROCESSING
df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date_time'])

# Extract time features
df['hour'] = df['date_time'].dt.hour
df['weekday'] = df['date_time'].dt.weekday
df['month'] = df['date_time'].dt.month

# Perform feature engineering
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
df['is_daytime'] = df['hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

# One-hot encoding
df = pd.get_dummies(df, columns=['weather_main', 'holiday'], drop_first=True)

# Select features and target
weather_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
X = df.drop(['date_time', 'weather_description', 'traffic_volume'], axis=1)
X = X[weather_cols + [col for col in X.columns if col not in weather_cols]]  # Ensure weather columns come first
y = df['traffic_volume']

# DATA VISUALIZATION 
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='traffic_volume', data=df, ci=None)
plt.title('Average Traffic Volume by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Traffic Volume')
plt.savefig('static/images/traffic_by_hour.png')
plt.close()

# SCALING & SPLITTING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# MODEL TRAINING
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Random Forest
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    },
    cv=3, n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Gradient Boosting
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid={
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    cv=3, n_jobs=-1, verbose=1
)
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_

# STACKING REGRESSOR
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

stack = StackingRegressor(
    estimators=[('rf', best_rf), ('gb', best_gb)],
    final_estimator=RidgeCV(),
    passthrough=True,
    n_jobs=-1
)
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)

# EVALUATION METRICS & VISUALIZATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import numpy as np
import os

# EVALUATION METRICS
mae = mean_absolute_error(y_test, y_pred_stack)
mse = mean_squared_error(y_test, y_pred_stack)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_stack)

# Create dictionary to store the metrics
performance_metrics = {
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2": round(r2, 4)
}

# Save metrics as JSON
with open('model/performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=4)

print("Performance metrics saved successfully.")

# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_stack, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.title('Actual vs Predicted Traffic Volume')
plt.savefig('static/images/actual_vs_predicted.png')
plt.close()

# Visualization: Residual Plot
residuals = y_test - y_pred_stack
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_stack, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('static/images/residual_plot.png')
plt.close()

# Visualization: Feature Importance (from Random Forest)
plt.figure(figsize=(12, 8))
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-15:]  # Top 15 features
plt.title('Feature Importances (Random Forest)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.savefig('static/images/feature_importances.png')
plt.close()








# =============================================
# Enhanced Traffic Pattern Visualizations
# =============================================
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# 1. Hourly Traffic Pattern (already exists, but let's enhance it)
plt.figure(figsize=(14, 7))
hourly_avg = df.groupby('hour')['traffic_volume'].mean()
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, 
             color='#4a6fa5', linewidth=2.5)
plt.title('Hourly Traffic Volume Patterns', fontsize=16, pad=20)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Average Traffic Volume', fontsize=14)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.savefig('static/images/hourly_traffic_patterns.png', 
           bbox_inches='tight', dpi=300)
plt.close()

# 2. Daily Traffic Pattern (Weekday vs Weekend)
plt.figure(figsize=(14, 7))
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 
                'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_avg = df.groupby('weekday')['traffic_volume'].mean()
sns.barplot(x=daily_avg.index, y=daily_avg.values, 
           palette=['#4a6fa5' if x < 5 else '#ff7e5f' for x in daily_avg.index])
plt.title('Daily Traffic Volume Patterns (Weekday vs Weekend)', fontsize=16, pad=20)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Average Traffic Volume', fontsize=14)
plt.xticks(ticks=range(7), labels=weekday_names)
plt.grid(True, alpha=0.3)
plt.savefig('static/images/daily_traffic_patterns.png', 
           bbox_inches='tight', dpi=300)
plt.close()

# 3. Monthly Traffic Pattern
plt.figure(figsize=(14, 7))
monthly_avg = df.groupby('month')['traffic_volume'].mean()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, 
            marker='o', color='#4a6fa5', linewidth=2.5)
plt.title('Monthly Traffic Volume Patterns', fontsize=16, pad=20)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Traffic Volume', fontsize=14)
plt.xticks(ticks=range(1, 13), labels=month_names)
plt.grid(True, alpha=0.3)
plt.savefig('static/images/monthly_traffic_patterns.png', 
           bbox_inches='tight', dpi=300)
plt.close()

# 4. Combined Traffic Pattern Visualization
plt.figure(figsize=(16, 12))

# Subplot 1: Hourly
plt.subplot(3, 1, 1)
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, 
             color='#4a6fa5', linewidth=2)
plt.title('Traffic Volume Patterns', fontsize=18, pad=20)
plt.ylabel('Hourly Traffic', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.2)

# Subplot 2: Daily
plt.subplot(3, 1, 2)
sns.barplot(x=daily_avg.index, y=daily_avg.values, 
           palette=['#4a6fa5' if x < 5 else '#ff7e5f' for x in daily_avg.index])
plt.ylabel('Daily Traffic', fontsize=12)
plt.xticks(ticks=range(7), labels=weekday_names)
plt.grid(True, alpha=0.2)

# Subplot 3: Monthly
plt.subplot(3, 1, 3)
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, 
            marker='o', color='#4a6fa5', linewidth=2)
plt.ylabel('Monthly Traffic', fontsize=12)
plt.xticks(ticks=range(1, 13), labels=month_names)
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('static/images/combined_traffic_patterns.png', 
           bbox_inches='tight', dpi=300)
plt.close()
# RESULTS & SAVING
print("Results for Final Enhanced Stacking Regressor")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

import joblib
joblib.dump(stack, 'model/traffic_stack_model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')

# Save feature names
with open('model/feature_columns.json', 'w') as f:
    import json
    json.dump(list(X.columns), f)

print("\nVisualizations saved to static/images/ directory")