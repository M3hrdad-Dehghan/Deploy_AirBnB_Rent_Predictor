from flask import Flask, request, jsonify, app, url_for, render_template
import pandas as pd
import numpy as np
import joblib
# from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor


app = Flask(__name__) # create an app instance
model = joblib.load('./artifacts/Regressor_Model.pkl')
Numerical_transformer = joblib.load('./artifacts/Numerical_Transformer.pkl')
OneHot_encoder = joblib.load('./artifacts/OneHot_Encoder.pkl')
Area_encoder = joblib.load('./artifacts/Area_Encoder.pkl')
Price_transformer = joblib.load('./artifacts/Price_yeo_johnson_transformer.pkl') 


# @app.route('/',methods=['GET'])
# def Home():
#     return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        Date = request.form['Date'] #از فرم به صورت تقویم
        Area = request.form['Area']  #از فرم به صورت دراپ داون
        Location = request.form['Location']  #از فرم به صورت دراپ داون
        Type = request.form['Type'] #از فرم به صورت دراپ داون
        MinDayNights = int(request.form['MinDayNights'])  #از فرم
        CountReview = float(request.form['CountReview'])  #از فرم
        AvgReview = int(request.form['AvgReview'])  #از فرم
        TotalHostListings = int(request.form['TotalHostListings'])  #از فرم
        DayAvailability = int(request.form['DayAvailability'])  #از فرم

        # Create a DataFrame for the input data
        input_df = pd.DataFrame({
            'LastDateReview': [Date],
            'Area': [Area],
            'Location': [Location],
            'Type': [Type],
            'MinDayNights': [MinDayNights],
            'CountReview': [CountReview],
            'AvgReview': [AvgReview],
            'TotalHostListings': [TotalHostListings],
            'DayAvailability': [DayAvailability]
        })
        
        
        # Split Date for Year, Month, Day
        input_df['LastDateReview'] = pd.to_datetime(input_df['LastDateReview'])
        input_df['Year'] = input_df['LastDateReview'].dt.year
        input_df['Month'] = input_df['LastDateReview'].dt.month
        input_df['Day'] = input_df['LastDateReview'].dt.day
        input_df.drop(columns=['LastDateReview'], inplace=True)


        # Encode Month and Day
        def encode(data, col, max_val):
            data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
            data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
            return data

        input_df = encode(input_df, 'Month', 12)
        input_df = encode(input_df, 'Day', 31)
        input_df = input_df.drop(columns=['Month', 'Day'])


        # Encode Area
        input_df['Area'] = input_df['Area'].map(Area_encoder)
        input_df['Area'].fillna(input_df['Area'].mean(), inplace=True)


        # Encode ocation & Type
        categorical_cols = ['Location', 'Type']
        encoded_array = OneHot_encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=OneHot_encoder.get_feature_names_out(categorical_cols))
        input_df_encoded = input_df.drop(columns=categorical_cols).reset_index(drop=True)
        input_df = pd.concat([input_df_encoded, encoded_df], axis=1)


        # Transform Numerical
        columns_to_transform = ['MinDayNights' , 'CountReview' , 'AvgReview' , 'TotalHostListings' , 'DayAvailability']
        input_df[columns_to_transform] = Numerical_transformer.transform(input_df[columns_to_transform])


        # Scale Numerical 
        columns_to_scale = ['MinDayNights', 'CountReview', 'AvgReview', 'TotalHostListings', 'DayAvailability', 'Year']
        scaler = ColumnTransformer([('scaler', StandardScaler(), columns_to_scale)], remainder='passthrough')
        X_scaled = scaler.fit_transform(input_df)
        all_columns = columns_to_scale + [col for col in input_df.columns if col not in columns_to_scale]
        X_scaled_df = pd.DataFrame(X_scaled, columns=all_columns)
        input_df = X_scaled_df.copy()


        # ReOrder
        input_df = input_df[['MinDayNights', 'CountReview', 'AvgReview', 'TotalHostListings',
                            'DayAvailability', 'Year', 'Area', 'Location_Brooklyn', 'Location_Manhattan', 
                            'Location_Queens', 'Location_Staten Island', 'Type_Private room', 'Type_Shared room', 
                            'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']]

        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_original = Price_transformer.inverse_transform(prediction.reshape(-1, 1))
        output = round(prediction_original[0], 2)
        
        if output < 0:
            return render_template('index.html', prediction_texts="Sorry you cannot rent your room")
        else:
            return render_template('index.html', prediction_text="You can rent at: $ {}".format(output))
    else:
        return render_template('index.html')
        
if __name__ == "__main__":
    app.run(debug=True)