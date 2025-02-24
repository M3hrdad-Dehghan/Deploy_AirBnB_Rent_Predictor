# جداسازی تاریخ به سال و ماه و روز
# LastDateReview
loaded_function = joblib.load('../../artifacts/year_extraction.pkl')
df_test = loaded_function(df_test)

# ترفسنرم کردن عددی ها
# MinDayNights / CountReview / AvgReview / TotalHostListings / DayAvailability
pt_loaded = joblib.load('../../artifacts/Numerical_Transformer.pkl')
df_test[columns_to_transform] = pt_loaded.transform(df_test[columns_to_transform])


# لیبل اینکود
# Area
encoder = joblib.load('../../artifacts/Area_Encoder.pkl')
input_df['XXX'] = input_df['XXXX'].map(Brand_Encoder)
input_df['XXXX'].fillna(input_df['XXXX'].mean(), inplace=True)


# هات اینکود
# Location / Type
OneHot_Encoder = joblib.load('../../artifacts/OneHot_Encoder.pkl')
categorical_cols = ['Location', 'Type']
encoded_array = OneHot_Encoder.transform(df_test[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=OneHot_Encoder.get_feature_names_out(categorical_cols))

input_df_encoded = df_test.drop(columns=categorical_cols).reset_index(drop=True)
df_test = pd.concat([input_df_encoded, encoded_df], axis=1)


#  اینکو ماه و روز
# day , month
cyclic_params = joblib.load('../../artifacts/DayMonth_Encoder.pkl')

df_test = encode(df_test, 'Month', cyclic_params['Month_max'])
df_test = encode(df_test, 'Day', cyclic_params['Day_max'])
df_test = df_test.drop(columns=['Month', 'Day'], errors='ignore')



# اسکیل 
# MinDayNights / CountReview / AvgReview / TotalHostListings / DayAvailability / Year



# ------------------------------
# تبدیل خروجی قیمت به عدد واقعی
loaded_transformer = joblib.load('../../artifacts/Price_yeo_johnson_transformer.pkl')
df['Price_original'] = loaded_transformer.inverse_transform(df[['Price']])
