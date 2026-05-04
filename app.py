import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 設定頁面
st.set_page_config(page_title="信用風險預測系統", layout="centered")

# 載入模型
@st.cache_resource
def load_model():
    try:
        # 使用 joblib 載入模型
        return joblib.load('model.pkl')
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return None

model = load_model()

st.title("🛡️ 銀行信用風險評估預測")
st.markdown("---")
st.sidebar.header("客戶資料輸入面板")

def user_input_features():
    util = st.sidebar.slider("額度使用率 (Utilization)", 0.0, 1.0, 0.3)
    age = st.sidebar.slider("年齡 (Age)", 18, 100, 35)
    late_30 = st.sidebar.number_input("30-59天逾期次數", 0, 20, 0)
    debt = st.sidebar.number_input("負債比 (DebtRatio)", 0.0, 10.0, 0.3)
    income = st.sidebar.number_input("月收入 (MonthlyIncome)", 0, 1000000, 50000)
    open_loans = st.sidebar.number_input("開放式貸款數量", 0, 50, 5)
    late_90 = st.sidebar.number_input("90天以上逾期次數", 0, 20, 0)
    estate = st.sidebar.number_input("不動產貸款數量", 0, 20, 1)
    late_60 = st.sidebar.number_input("60-89天逾期次數", 0, 20, 0)
    dep = st.sidebar.number_input("家屬人數", 0, 20, 0)

    data = {
        'RevolvingUtilizationOfUnsecuredLines': util,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30,
        'DebtRatio': debt,
        'MonthlyIncome': income,
        'NumberOfOpenCreditLinesAndLoans': open_loans,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': estate,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60,
        'NumberOfDependents': dep
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

if st.button("🚀 開始評估風險"):
    if model is not None:
        prediction_proba = model.predict_proba(input_df)[0][1]
        st.subheader(f"預測違約機率：{prediction_proba:.2%}")
        if prediction_proba > 0.5:
            st.error("❌ 高風險 (建議拒貸)")
        else:
            st.success("✅ 低風險 (准予核貸)")
    else:
        st.warning("系統尚未準備好，請檢查模型檔案。")
