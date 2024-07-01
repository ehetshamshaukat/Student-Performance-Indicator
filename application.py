import streamlit as st
from src.pipeline.predicting_pipeline import GetFeatures,Prediction


gender_ranking=["male","female"]
race_ethnicity_ranking=["group A","group B","group C","group D","group E"]
parental_level_of_education_ranking=['some high school',"high school","associate's degree",'some college',"bachelor's degree","master's degree"]
lunch_ranking=["standard","free/reduced"]
test_preparation_course_ranking=["completed","none"]

st.subheader("Student Performance Indicator")
gender=st.selectbox("Gender",gender_ranking)
race_ethnicity=st.selectbox("Race ethnicity",race_ethnicity_ranking)
parental_level_of_education=st.selectbox("parental_level_of_education",parental_level_of_education_ranking)
lunch=st.selectbox("lunch",lunch_ranking)
test_preparation_course=st.selectbox("test preparation course",test_preparation_course_ranking)
reading_score=st.number_input("enter reading score",value=0)
writing_score=st.number_input("enter writing score",value=0)




ok =st.button("PREDICT")

if ok:
    gf=GetFeatures(gender=gender,
                   race_ethnicity=race_ethnicity,
                   parental_level_of_education=parental_level_of_education,
                   lunch=lunch,
                   test_preparation_course=test_preparation_course,
                   reading_score=reading_score,
                   writing_score=writing_score)
    df=gf.to_dataframe()
    pred=Prediction()
    op=pred.initiate_prediction(df)
    st.subheader(round(float(op),2))


