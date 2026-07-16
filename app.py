from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# تحميل النموذج والمقياس
model = joblib.load("model.pkl")     
scaler = joblib.load("scaler.pkl")

# قائمة الأعمدة الـ 40 الدقيقة بالترتيب المطلوب من الـ Scaler لديك
FEATURES_COLUMNS = [
    'Hour', 'Temperature_C', 'Humidity_pct', 'Wind_Speed_ms', 'Visibility_m', 
    'Solar_Radiation_MJm2', 'Rainfall_mm', 'Month', 'Day', 'Is_Peak_Hour', 
    'Is_Weekend', 'Season_Spring', 'Season_Summer', 'Season_Unknown', 'Season_Winter', 
    'Holiday_Unknown', 'Holiday_Yes', 'Library_Branch_Al Rawdah Branch', 
    'Library_Branch_Corniche Kiosk', 'Library_Branch_Downtown Central', 
    'Library_Branch_University Branch', 'Top_Category_Business', 'Top_Category_Children', 
    'Top_Category_Fiction', 'Top_Category_History', 'Top_Category_Non-Fiction', 
    'Top_Category_Science', 'Top_Category_Technology', 'Membership_Type_Regular', 
    'Membership_Type_Student', 'Membership_Type_Unknown', 'Membership_Type_Walk-In', 
    'Day_of_Week_Monday', 'Day_of_Week_Saturday', 'Day_of_Week_Sunday', 
    'Day_of_Week_Thursday', 'Day_of_Week_Tuesday', 'Day_of_Week_Wednesday',
    'Temperature_Bin_Warm', 'Temperature_Bin_Hot'
]

# 5.1 — إضافة نقطة النهاية للفحص الصحي (Health Check Endpoint)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# 5.2 — نقطة التنبؤ ومعالجة الأخطاء بشكل مرن
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload received"}), 400
        
        # إنشاء قاموس يحتوي على قيم افتراضية لجميع الأعمدة
        input_data = {col: 0.0 for col in FEATURES_COLUMNS}
        
        # ملء القيم الأساسية التي يرسلها المستخدم مع التحقق من النوع البياني وتحويله
        try:
            input_data['Hour'] = float(data.get('Hour', 12))
            input_data['Temperature_C'] = float(data.get('Temperature_C', 25.0))
            input_data['Humidity_pct'] = float(data.get('Humidity_pct', 50.0))
            input_data['Wind_Speed_ms'] = float(data.get('Wind_Speed_ms', 1.5))
            input_data['Visibility_m'] = float(data.get('Visibility_m', 2000.0))
            input_data['Solar_Radiation_MJm2'] = float(data.get('Solar_Radiation_MJm2', 0.0))
            input_data['Rainfall_mm'] = float(data.get('Rainfall_mm', 0.0))
            input_data['Month'] = float(data.get('Month', 7)) 
            input_data['Day'] = float(data.get('Day', 16))
            input_data['Is_Peak_Hour'] = float(data.get('Is_Peak_Hour', 0))
            input_data['Is_Weekend'] = float(data.get('Is_Weekend', 0))
        except ValueError as val_err:
            return jsonify({"status": "error", "message": f"Invalid numerical value: {str(val_err)}"}), 400
        
        # معالجة المتغيرات الفئوية وتفعيل الـ Dummy المناسب
        categories = {
            "Season": "Season_",
            "Library_Branch": "Library_Branch_",
            "Top_Category": "Top_Category_",
            "Membership_Type": "Membership_Type_",
            "Day_of_Week": "Day_of_Week_",
            "Temperature_Bin": "Temperature_Bin_"
        }
        
        for key, prefix in categories.items():
            if key in data:
                col_name = f"{prefix}{data[key]}"
                if col_name in input_data:
                    input_data[col_name] = 1.0

        # تحويل القاموس إلى DataFrame للحفاظ على ترتيب الأعمدة الدقيق للـ Scaler
        df_input = pd.DataFrame([input_data])[FEATURES_COLUMNS]
        
        # تطبيق التحجيم (Scaling) والتوقع
        features_scaled = scaler.transform(df_input)
        prediction = model.predict(features_scaled)
        
        predicted_val = prediction[0]
        if hasattr(predicted_val, '__len__'):
            predicted_val = predicted_val[0]
            
        return jsonify({
            "status": "success",
            "predicted_rentals": float(predicted_val)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 400

if __name__ == "__main__":
    # 5.3 — قراءة المنفذ من متغير بيئي (PORT) والافتراضي هو 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)