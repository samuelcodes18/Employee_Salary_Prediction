import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Updated title and subtitle
st.title("💼Employee Salary Prediction [Using ML Libraries]")
st.subheader("\nPlease enter the below information to predict the salary:")

# Load the best saved model
import joblib
model = joblib.load('best_model.pkl')

def main():
    # Age slider
    age = st.slider('Age', min_value=17, max_value=90, value=25)

    workclass = st.selectbox('Select Work Class', [' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',
                                                    ' Local-gov', 'Others', ' Self-emp-inc', ' Without-pay', ' Never-worked'])
    workclass_dict = {' State-gov': 0, ' Self-emp-not-inc': 1, ' Private': 2, ' Federal-gov': 3,
                      ' Local-gov': 4, 'Others': 5, ' Self-emp-inc': 6, ' Without-pay': 7, ' Never-worked': 8}
    workclass_1 = workclass_dict.get(workclass, 5)

    fnlwgt = st.number_input('Final Weight', min_value=12285, max_value=1484705, value=12285)
    education_num = st.number_input('Education Number', min_value=1, max_value=16, value=1)

    marital_status = st.selectbox('Marital Status', [' Never-married', ' Married-civ-spouse', ' Divorced',
                                                     ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
                                                     ' Widowed'])
    marital_dict = {' Never-married': 0, ' Married-civ-spouse': 1, ' Divorced': 2,
                    ' Married-spouse-absent': 3, ' Separated': 4, ' Married-AF-spouse': 5, ' Widowed': 6}
    marital_status_1 = marital_dict.get(marital_status, 0)

    occupation = st.selectbox('Occupation', [' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
                                             ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',
                                             ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
                                             ' Tech-support', ' Others', ' Protective-serv', ' Armed-Forces',
                                             ' Priv-house-serv'])
    occupation_dict = {' Adm-clerical': 0, ' Exec-managerial': 1, ' Handlers-cleaners': 2, ' Prof-specialty': 3,
                       ' Other-service': 4, ' Sales': 5, ' Craft-repair': 6, ' Transport-moving': 7,
                       ' Farming-fishing': 8, ' Machine-op-inspct': 9, ' Tech-support': 10, ' Others': 11,
                       ' Protective-serv': 12, ' Armed-Forces': 13, ' Priv-house-serv': 14}
    occupation_1 = occupation_dict.get(occupation, 11)

    relationship = st.selectbox('Relationship', [' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried', ' Other-relative'])
    relationship_dict = {' Not-in-family': 0, ' Husband': 1, ' Wife': 2, ' Own-child': 3, ' Unmarried': 4, ' Other-relative': 5}
    relationship_1 = relationship_dict.get(relationship, 0)

    race = st.selectbox('Race', [' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other'])
    race_dict = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4}
    race_1 = race_dict.get(race, 0)

    sex = st.selectbox('Sex', [' Male', ' Female'])
    sex_1 = 0 if sex == ' Male' else 1

    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=4356, value=0)
    hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=99, value=1)

    native_country = st.selectbox('Native Country', [' United-States', ' Cuba', ' Jamaica', ' India', ' Others', ' Mexico',
                                                     ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada',
                                                     ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland',
                                                     ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos',
                                                     ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',
                                                     ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan',
                                                     ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',
                                                     ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong',
                                                     ' Ireland', ' Hungary', ' Holand-Netherlands'])
    country_dict = {val: idx for idx, val in enumerate([' United-States', ' Cuba', ' Jamaica', ' India', ' Others', ' Mexico',
                                                        ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada',
                                                        ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland',
                                                        ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos',
                                                        ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',
                                                        ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan',
                                                        ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',
                                                        ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong',
                                                        ' Ireland', ' Hungary', ' Holand-Netherlands'])}
    native_country_1 = country_dict.get(native_country, 4)

    if st.button("Predict Salary Range"):
        data = [age, workclass_1, fnlwgt, education_num, marital_status_1,
                occupation_1, relationship_1, race_1, sex_1, capital_gain, capital_loss,
                hours_per_week, native_country_1]

        # Apply scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform([data])

        result = model.predict(scaled_data)
        if result == 0:
            st.success("The salary is likely **less than 50K**.")
        else:
            st.success("The salary is likely **more than 50K**.")
        
        # Divider line after prediction
        st.markdown("<hr>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
