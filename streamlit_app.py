import streamlit as st
import requests
import pandas as pd
import joblib

def main():
    st.title("Home Credit Risk Classification")

    # Get the data from the user
    contract_type = st.selectbox("Select contract type", 
                                 ["Cash loans", "Revolving loans"])
    gender = st.selectbox("Select gender", ["F", "M"])
    own_car = st.selectbox("Do you own a car?", ["Y", "N"])
    own_realty = st.selectbox("Do you own a realty?", ["Y", "N"])
    children_count = st.number_input("Enter the number of children", min_value=0)
    income_total = st.number_input("Enter the total income", min_value=0)
    credit_amount = st.number_input("Enter the credit amount", min_value=0)
    goods_price = st.number_input("Enter the goods price", min_value=0)
    income_type = st.selectbox("Select income type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"])
    education_type = st.selectbox("Select education type", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    birth_days = st.number_input("Enter the number of days since birth", min_value=0)
    employed_days = st.number_input("Enter the number of days employed", min_value=0)
    family_members = st.number_input("Enter the number of family members", min_value=0)
    ext_source1 = st.number_input("Enter the value of ext_source1", min_value=0)
    ext_source2 = st.number_input("Enter the value of ext_source2", min_value=0)
    ext_source3 = st.number_input("Enter the value of ext_source3", min_value=0)

    # Create the dataframe
    data = {
        'NAME_CONTRACT_TYPE': contract_type,
        'CODE_GENDER': gender,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children_count,
        'AMT_INCOME_TOTAL': income_total,
        'AMT_CREDIT': credit_amount,
        'AMT_GOODS_PRICE': goods_price,
        'NAME_INCOME_TYPE': income_type,
        'NAME_EDUCATION_TYPE': education_type,
        'DAYS_BIRTH': birth_days,
        'DAYS_EMPLOYED': employed_days,
        'CNT_FAM_MEMBERS': family_members,
        'EXT_SOURCE_1': ext_source1,
        'EXT_SOURCE_2': ext_source2,
        'EXT_SOURCE_3': ext_source3
        }
    features = pd.DataFrame(data, index=[0])
    # Make the prediction
    model = joblib.load('src/model/lgb_credits.joblib')
    scaler = joblib.load('src/model/minMax_scaler_credits.joblib')
    
    prediction = model.predict(features)

    # Show the results
    if prediction == 1:
        st.success("The client is at high risk of default.")
    else:
        st.success("The client is not at high risk of default.")


if __name__ == '__main__':
    main()