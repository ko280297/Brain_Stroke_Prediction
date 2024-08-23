import streamlit as st
import pandas as pd
import joblib

# Load the decision tree model
model = joblib.load('model/random_forest_classifier_model.pkl')

# Load example data (if needed)
data = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

def main():
    # Title and Description
    st.title('Brain Stroke Prediction App')
    st.write('This app uses a Random Forest model to predict the likelihood of a stroke based on user inputs.')

    # Sidebar for Input Features
    st.sidebar.header('Input Features')
    gender = st.sidebar.radio('Gender', ['Male', 'Female', 'Other'])
    gender_encoded = 0 if gender == 'Female' else 1 if gender == 'Male' else 2
    
    smoking_status = st.sidebar.selectbox('Do you smoke?', ['Unknown', 'Formerly smoked', 'Never smoked', 'Smokes'])
    smoking_encoded = {'Unknown': 0, 'Formerly smoked': 1, 'Never smoked': 2, 'Smokes': 3}[smoking_status]
    
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
    
    hypertension = st.sidebar.radio('Do you have hypertension?', ['Yes', 'No'])
    hypertension_encoded = 1 if hypertension == 'Yes' else 0
    
    work_type = st.sidebar.selectbox('What is your work status?', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
    work_encoded = {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'Children': 4}[work_type]
    
    heart_disease = st.sidebar.radio('Do you have heart disease?', ['Yes', 'No'])
    heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
    
    ever_married = st.sidebar.radio('Are you married?', ['Yes', 'No'])
    ever_married_encoded = 1 if ever_married == 'Yes' else 0
    
    residence_type = st.sidebar.radio('Where do you currently reside?', ['Urban', 'Rural'])
    residence_type_encoded = 1 if residence_type == 'Urban' else 0
    
    avg_glucose_level = st.sidebar.number_input('Average Glucose Level', min_value=0.0, format="%.2f")
    bmi = st.sidebar.number_input('BMI (Body Mass Index)', min_value=0.0, format="%.2f")

    # Button to make prediction
    if st.sidebar.button('Predict'):
        # Create a DataFrame with the input features
        input_data = {
            'gender': [gender_encoded],
            'age': [age],
            'hypertension': [hypertension_encoded],
            'heart_disease': [heart_disease_encoded],
            'ever_married': [ever_married_encoded],
            'work_type': [work_encoded],
            'Residence_type': [residence_type_encoded],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_encoded]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Debugging: Print the input data
        st.write("Input Data:")
        st.write(input_df)

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Debugging: Print the prediction value and probabilities
        st.write(f"Prediction Value: {prediction[0]}")
        st.write(f"Prediction Probabilities: {prediction_proba[0]}")

        # Display prediction and probabilities
        if prediction[0] == 1:
            st.warning('Prediction: High chance of stroke', icon="⚠️")
        else:
            st.success('Prediction: Low chance of stroke', icon="✅")
        
        
        st.write(f'**Probability of Low chance of stroke:** {prediction_proba[0][0]:.2f}')
        st.write(f'**Probability of High chance of stroke:** {prediction_proba[0][1]:.2f}')


  

    

    # Display Example Data (optional)
    #if st.checkbox('Show example data'):
     #   st.write(data.head())

if __name__ == '__main__':
    main()
