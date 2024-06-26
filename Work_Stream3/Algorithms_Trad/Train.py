import pandas as pd
import joblib

train = pd.read_csv(r"C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1\F1\train.csv")

train['sex'] = train['sex'].replace({'Male': 1, 'Female': 0})
train['race'] = train['race'].replace({'African':0, 'White/Europid' :1, 'Mixed/other' :2,'South Asian':3, 'Oriental':4})
train['Vascular'] = train['Vascular'].replace({'Yes':1, 'No':0})
train['Coronary'] = train['Coronary'].replace({'Yes':1, 'No':0})
train['Diabsube'] = train['Diabsube'].replace({'Yes':1, 'No':0})


y_train = train['sbpmean_visit_4']
x_train = train.drop(['sbpmean_visit_4', 'subj_ID'], axis=1)

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x_train, y_train)
# joblib.dump(model, 'linear_regression_model.pkl')


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(x_train, y_train)
joblib.dump(model, 'random_forest_model.pkl')


# from sklearn.ensemble import GradientBoostingRegressor
# model = GradientBoostingRegressor(n_estimators=200,learning_rate=.02)
# model.fit(x_train, y_train)
# joblib.dump(model, 'gradient_boosting_model.pkl')



# Load the saved model from file and test


import pandas as pd
import joblib

val = pd.read_csv(r"C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1\F1\val.csv")

val['sex'] = val['sex'].replace({'Male': 1, 'Female': 0})
val['race'] = val['race'].replace({'African':0, 'White/Europid' :1, 'Mixed/other' :2,'South Asian':3, 'Oriental':4})
val['Vascular'] = val['Vascular'].replace({'Yes':1, 'No':0})
val['Coronary'] = val['Coronary'].replace({'Yes':1, 'No':0})
val['Diabsube'] = val['Diabsube'].replace({'Yes':1, 'No':0})


y_val = val['sbpmean_visit_4']
x_val = val.drop(['sbpmean_visit_4'], axis=1)

import pandas as pd
import joblib
import pandas as pd
loaded_model = joblib.load('random_forest_model.pkl')
# Initialize lists to store SubjID, input variables, target values, predictions, and errors
subjID_list = []
inputs_list = []
target_values = []
prediction_values = []
absolute_errors = []

# Iterate through each row in the test dataset
for index, row in x_val.iterrows():
    # Store the SubjID for the current row
    subjID_list.append(row['subj_ID'])
    
    # Remove the 'SubjID' column before making predictions
    row = row.drop('subj_ID')
    
    # Make predictions for the current row using the loaded model
    prediction = loaded_model.predict([row])[0]
    
    # Store the input variables, target value, prediction, and absolute error
    inputs_list.append(row.values.tolist())
    target_value = y_val.loc[index]
    target_values.append(target_value)
    prediction_values.append(prediction)
    absolute_error = abs(prediction - target_value)
    absolute_errors.append(absolute_error)

# Create a DataFrame containing input variables, target values, prediction values, and absolute errors
results_df = pd.DataFrame(inputs_list, columns=x_val.columns.drop('subj_ID'))
results_df.insert(0, 'subj_ID', subjID_list)
results_df['Target'] = target_values
results_df['Prediction'] = prediction_values
results_df['Absolute Error'] = absolute_errors


# Save the DataFrame to a CSV file
results_df.to_csv('gradient_boosting_model_F1.csv', index=False)

# Calculate the average error
average_error = sum(absolute_errors) / len(absolute_errors)
print("Average Error:", average_error)


import pandas as pd
import joblib


def norm_age(age):
    return (age-40.05)/(85.87-40.05)

def norm_bmi(bmi):
    return (bmi-15.0)/(61.5-15.0)

def norm_days(days):
    return days/30

train = pd.read_csv(r"C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1\F1\train.csv")

train['sex'] = train['sex'].replace({'Male': 1, 'Female': 0})
train['race'] = train['race'].replace({'African':0, 'White/Europid' :1, 'Mixed/other' :2,'South Asian':3, 'Oriental':4})
train['Vascular'] = train['Vascular'].replace({'Yes':1, 'No':0})
train['Coronary'] = train['Coronary'].replace({'Yes':1, 'No':0})
train['Diabsube'] = train['Diabsube'].replace({'Yes':1, 'No':0})

max_sbmean = train['sbpmean_visit_1'].max()
min_sbmean = train['sbpmean_visit_1'].min()

print(f"Maximum value in 'sbmean' column: {max_sbmean}")
print(f"Minimum value in 'sbmean' column: {min_sbmean}")



y_train = train['sbpmean_visit_4']
x_train = train.drop(['sbpmean_visit_4', 'subj_ID'], axis=1)


x_train['age_visit_1'] = x_train['age_visit_1'].apply(norm_age)
x_train['age_visit_2'] = x_train['age_visit_2'].apply(norm_age)
x_train['age_visit_3'] = x_train['age_visit_3'].apply(norm_age)
x_train['age_visit_4'] = x_train['age_visit_4'].apply(norm_age)

x_train['bmi_visit_1'] = x_train['bmi_visit_1'].apply(norm_bmi)
x_train['bmi_visit_2'] = x_train['bmi_visit_2'].apply(norm_bmi)
x_train['bmi_visit_3'] = x_train['bmi_visit_3'].apply(norm_bmi)
x_train['bmi_visit_4'] = x_train['bmi_visit_4'].apply(norm_bmi)


# for v in range(4)
# x_train['days_0_visit_4'] = x_train['days_0_visit_4'].apply(norm_days)
# x_train['days_1_visit_4'] = x_train['days_1_visit_4'].apply(norm_days)
# x_train['days_2_visit_4'] = x_train['days_2_visit_4'].apply(norm_days)
# x_train['days_3_visit_4'] = x_train['days_3_visit_4'].apply(norm_days)
# x_train['days_4_visit_4'] = x_train['days_4_visit_4'].apply(norm_days)


        
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x_train, y_train)
# joblib.dump(model, 'linear_regression_model.pkl')


# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100)
# model.fit(x_train, y_train)
# joblib.dump(model, 'random_forest_model.pkl')


from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=200,learning_rate=.02)
model.fit(x_train, y_train)
joblib.dump(model, 'gradient_boosting_model.pkl')



# Load the saved model from file and test


import pandas as pd
import joblib

val = pd.read_csv(r"C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1\F1\val.csv")

val['sex'] = val['sex'].replace({'Male': 1, 'Female': 0})
val['race'] = val['race'].replace({'African':0, 'White/Europid' :1, 'Mixed/other' :2,'South Asian':3, 'Oriental':4})
val['Vascular'] = val['Vascular'].replace({'Yes':1, 'No':0})
val['Coronary'] = val['Coronary'].replace({'Yes':1, 'No':0})
val['Diabsube'] = val['Diabsube'].replace({'Yes':1, 'No':0})

max_sbmean = val['sbpmean_visit_4'].max()
min_sbmean = val['sbpmean_visit_4'].min()

print(f"Maximum value in 'sbmean' column: {max_sbmean}")
print(f"Minimum value in 'sbmean' column: {min_sbmean}")




y_val = val['sbpmean_visit_4']
x_val = val.drop(['sbpmean_visit_4'], axis=1)

x_val['age_visit_1'] = x_val['age_visit_1'].apply(norm_age)
x_val['age_visit_2'] = x_val['age_visit_2'].apply(norm_age)
x_val['age_visit_3'] = x_val['age_visit_3'].apply(norm_age)
x_val['age_visit_4'] = x_val['age_visit_4'].apply(norm_age)

x_val['bmi_visit_1'] = x_val['bmi_visit_1'].apply(norm_bmi)
x_val['bmi_visit_2'] = x_val['bmi_visit_2'].apply(norm_bmi)
x_val['bmi_visit_3'] = x_val['bmi_visit_3'].apply(norm_bmi)
x_val['bmi_visit_4'] = x_val['bmi_visit_4'].apply(norm_bmi)


import pandas as pd
import joblib
import pandas as pd
loaded_model = joblib.load('gradient_boosting_model.pkl')
# Initialize lists to store SubjID, input variables, target values, predictions, and errors
subjID_list = []
inputs_list = []
target_values = []
prediction_values = []
absolute_errors = []

# Iterate through each row in the test dataset
for index, row in x_val.iterrows():
    # Store the SubjID for the current row
    subjID_list.append(row['subj_ID'])
    
    # Remove the 'SubjID' column before making predictions
    row = row.drop('subj_ID')
    
    # Make predictions for the current row using the loaded model
    prediction = loaded_model.predict([row])[0]
    
    # Store the input variables, target value, prediction, and absolute error
    inputs_list.append(row.values.tolist())
    target_value = y_val.loc[index]
    target_values.append(target_value)
    prediction_values.append(prediction)
    absolute_error = abs(prediction - target_value)
    absolute_errors.append(absolute_error)

# Create a DataFrame containing input variables, target values, prediction values, and absolute errors
results_df = pd.DataFrame(inputs_list, columns=x_val.columns.drop('subj_ID'))
results_df.insert(0, 'subj_ID', subjID_list)
results_df['Target'] = target_values
results_df['Prediction'] = prediction_values
results_df['Absolute Error'] = absolute_errors


# Save the DataFrame to a CSV file
results_df.to_csv('gradient_boosting_model_F1.csv', index=False)

# Calculate the average error
average_error = sum(absolute_errors) / len(absolute_errors)
print("Average Error:", average_error)


