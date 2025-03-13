import plotly.graph_objects as go
import wbdata
import plotly.express as px
import datetime
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import google.generativeai as genai


genai.configure(api_key="AIzaSyAjn2-Jg4ke_15hhpgz4LxzioHrPQu5I7s")

# إعداد الواجهة
st.set_page_config(page_title="تحليل البيانات باستخدام ", layout="wide")
def multiple_regression():
    df = pd.DataFrame()
    st.subheader("Multiple Regression Analysis")
    uploaded_file = st.file_uploader("تحميل ملف البيانات", type=".xlsx")
    use_example_file = st.checkbox(
        "استخدم المثال الموجود", False, help="Use in-built example file to demo the app"
    )

    # If CSV is not uploaded and checkbox is filled, use values from the example file
    # and pass them down to the next if block
    if use_example_file:
        uploaded_file = "data_sa.xlsx"
    if uploaded_file:
        df = pd.read_excel(uploaded_file).copy()

    # إدخال عنوان الدراسة ووصف المتغيرات
    study_description = st.text_area("📝 Enter Study Title & Variable Description:",
                                     """The study examines the impact of inflation on economic growth in Tunisia using a multiple regression model. 
    The available data includes:
    - **Growth**: Represents economic growth.
    - **L**: Represents the number of workers.
    - **INF**: Represents inflation.
    - **GFCF**: Represents gross fixed capital formation.
    The dataset covers the period from **1991 to 2023**.""" ,height=200)
    target_var = st.selectbox("Select Dependent Variable (Y):", df.columns, index=3)
    predictors = st.multiselect("Select Independent Variables (X):", df.columns,
                                default=[col for col in df.columns if col != target_var])
    tab1, tab2 = st.tabs(["📊 النتائج", "💻 الكود"])
    with tab1:
        if st.button("Run Multiple Regression") and target_var and predictors:
            model1 = genai.GenerativeModel("gemini-1.5-flash")
            # مقدمة الدراسة
            input_intro = study_description + "\nProvide an introduction for this research."
            response_intro = model1.generate_content(input_intro)
            st.subheader("📌 Introduction")
            st.write(response_intro.text)

            # الأدبيات السابقة
            input_lit_review = study_description + "\nSummarize previous literature related to this topic."
            response_lit_review = model1.generate_content(input_lit_review)
            st.subheader("📚 Literature Review")
            st.write(response_lit_review.text)

            # المنهجية
            input_methodology = study_description + "\nDescribe the methodology used in this research."
            response_methodology = model1.generate_content(input_methodology)
            st.subheader("🛠️ Methodology")
            st.write(response_methodology.text)
            Y = df[[target_var]]
            X = df[predictors]
            X.insert(0, 'Intercept', 1)

            model = sm.OLS(Y, X).fit()
            st.subheader("Full Regression Results:")
            st.write(model.summary())
            input = str(
                model.summary()) + "اريد منك تحليل هذا الجدول ولا تنسى اختبار التوزيع الطبيعي واختبار الارتباط الذاتي "
            response = model1.generate_content(input)
            st.write(response.text)
            vif_df = compute_vif(X)
            st.subheader("Variance Inflation Factor (VIF) for Each Variable:")
            st.table(vif_df)
            input = str(vif_df) + "اريد منك تحليل هذا الجدول الخاص ب Variance Inflation Factor (VIF) for Each Variable"
            response = model1.generate_content(input)
            st.write(response.text)
    with tab2:

        code1 = """  
def compute_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variables"] = X.columns
    vif_data["VIF"] = [1 / (1 - sm.OLS(X[col], X.drop(columns=[col])).fit().rsquared) for col in X.columns]
    return vif_data
def multiple_regression():
    df = pd.DataFrame()
    st.subheader("Multiple Regression Analysis")
    uploaded_file = st.file_uploader("تحميل ملف البيانات", type=".xlsx")
    use_example_file = st.checkbox(
        "استخدم المثال الموجود", False, help="Use in-built example file to demo the app"
    )

    # If CSV is not uploaded and checkbox is filled, use values from the example file
    # and pass them down to the next if block
    if use_example_file:
        uploaded_file = "data_sa.xlsx"
    if uploaded_file:
        st.session_state.df = pd.read_excel(uploaded_file).copy()
    df = st.session_state.df
    # إدخال عنوان الدراسة ووصف المتغيرات
    study_description = st.text_area("📝 Enter Study Title & Variable Description:",
                                     The study examines the impact of inflation on economic growth in Tunisia using a multiple regression model. 
    The available data includes:
    - **Growth**: Represents economic growth.
    - **L**: Represents the number of workers.
    - **INF**: Represents inflation.
    - **GFCF**: Represents gross fixed capital formation.
    The dataset covers the period from **1991 to 2023**. ,height= 200)
    target_var = st.selectbox("Select Dependent Variable (Y):", df.columns,index=3)
    predictors = st.multiselect("Select Independent Variables (X):", df.columns,
                                default=[col for col in df.columns if col != target_var])
    tab1, tab2 = st.tabs(["📊 النتائج", "💻 الكود"])
    with tab1:
        if st.button("Run Multiple Regression") and target_var and predictors:
            model1 = genai.GenerativeModel("gemini-1.5-flash")
            # مقدمة الدراسة
            input_intro = study_description + "\nProvide an introduction for this research."
            response_intro = model1.generate_content(input_intro)
            st.subheader("📌 Introduction")
            st.write(response_intro.text)

            # الأدبيات السابقة
            input_lit_review = study_description + "\nSummarize previous literature related to this topic."
            response_lit_review = model1.generate_content(input_lit_review)
            st.subheader("📚 Literature Review")
            st.write(response_lit_review.text)

            # المنهجية
            input_methodology = study_description + "\nDescribe the methodology used in this research."
            response_methodology = model1.generate_content(input_methodology)
            st.subheader("🛠️ Methodology")
            st.write(response_methodology.text)
            Y = df[[target_var]]
            X = df[predictors]
            X.insert(0, 'Intercept', 1)

            model = sm.OLS(Y, X).fit()
            st.subheader("Full Regression Results:")
            st.write(model.summary())
            input = str(
                model.summary()) + "اريد منك تحليل هذا الجدول ولا تنسى اختبار التوزيع الطبيعي واختبار الارتباط الذاتي "
            response = model1.generate_content(input)
            st.write(response.text)
            vif_df = compute_vif(X)
            st.subheader("Variance Inflation Factor (VIF) for Each Variable:")
            st.table(vif_df)
            input = str(vif_df) + "اريد منك تحليل هذا الجدول الخاص ب Variance Inflation Factor (VIF) for Each Variable"
            response = model1.generate_content(input)
            st.write(response.text) 

        """
        st.code(code1)

multiple_regression()
