from flask import Flask
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='./templates')
model = pickle.load(open('best_classifier.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    df = pd.read_csv(request.files.get('file'))
    data = predict(df)
    return data.to_json()


def predict(df):
    group_8_df = df.loc[~df.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]

    group_8_df['OUTPUT_CLASS'] = (group_8_df.readmitted == '<30').astype('int')
    group_8_df = group_8_df.replace('?', np.nan)
    cols_num = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    group_8_df[cols_num].isnull().sum()

    # Adding Categorical Features (Object) columns into cols_cat list as it is easy to use
    # Step: 21
    cols_cat = ['race', 'gender',
                'max_glu_serum', 'A1Cresult',
                'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone', 'change', 'diabetesMed', 'payer_code', 'medical_specialty']

    # find missing values in the categorical data.
    # Step: 22
    group_8_df[cols_cat].isnull().sum()

    group_8_df['race'] = group_8_df['race'].fillna('UNK')
    group_8_df['payer_code'] = group_8_df['payer_code'].fillna('UNK')
    group_8_df['medical_specialty'] = group_8_df['medical_specialty'].fillna('UNK')
    group_8_df.groupby('medical_specialty').size().sort_values(ascending=False)

    top_10 = ['UNK', 'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General',
              'Nephrology', 'Orthopedics',
              'Orthopedics-Reconstructive', 'Radiologist']

    group_8_df['med_spec'] = group_8_df['medical_specialty'].copy()

    group_8_df.loc[~group_8_df.med_spec.isin(top_10), 'med_spec'] = 'Other'

    group_8_df.groupby('med_spec').size()

    cols_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    group_8_df[cols_cat_num] = group_8_df[cols_cat_num].astype('str')

    group_8_df_cat = pd.get_dummies(group_8_df[cols_cat + cols_cat_num + ['med_spec']], drop_first=True)

    group_8_df = pd.concat([group_8_df, group_8_df_cat], axis=1)

    cols_all_cat = list(group_8_df_cat.columns)

    group_8_df.groupby('age').size()

    age_id = {'[0-10)': 0,
              '[10-20)': 10,
              '[20-30)': 20,
              '[30-40)': 30,
              '[40-50)': 40,
              '[50-60)': 50,
              '[60-70)': 60,
              '[70-80)': 70,
              '[80-90)': 80,
              '[90-100)': 90}
    group_8_df['age_group'] = group_8_df.age.replace(age_id)

    group_8_df.weight.notnull().sum()

    group_8_df['has_weight'] = group_8_df.weight.notnull().astype('int')

    cols_extra = ['age_group', 'has_weight']

    col2use = cols_num + cols_all_cat + cols_extra
    group_8_df_data_out = group_8_df[['patient_nbr'] + col2use + ['OUTPUT_CLASS']]
    group_8_df_data = group_8_df[col2use + ['OUTPUT_CLASS']]
    group_8_df_data = group_8_df_data.sample(n=len(group_8_df_data), random_state=42)
    group_8_df_data = group_8_df_data.reset_index(drop=True)

    x_test = group_8_df_data[col2use].values
    y_test = group_8_df_data['OUTPUT_CLASS'].values
    scaler = pickle.load(open('scaler.sav', 'rb'))
    x_test_tf = scaler.transform(x_test)

    r_x_test = group_8_df_data_out[['patient_nbr']]
    y_test_preds = model.predict_proba(x_test_tf)[:, 1]
    r_x_test["predictions value in %"] = pd.Series(np.round(y_test_preds * 100, 2), index=r_x_test.index)

    return r_x_test


if __name__ == '__main__':
    app.run()
