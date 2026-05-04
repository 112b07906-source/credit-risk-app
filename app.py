import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 設定頁面語系與標題
st.set_page_config(page_title="信用風險預測系統", layout="centered")

# 1. 載入模型的函式 (加入快取避免重複讀取)
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ 模型檔案 (model.pkl) 載入失敗：{e}")
        return None

model = load_model()

# 2. 建立網頁介面
st.title("🛡️ 銀行信用風險評估系統")
st.markdown("請在左側面板輸入客戶資料，系統將即時計算未來兩年的違約機率。")
st.sidebar.header("📋 客戶特徵輸入")

def user_input_features():
    # 依照訓練時的 10 個特徵順序排列
    util = st.sidebar.slider("額度使用率 (RevolvingUtilization)", 0.0, 1.0, 0.3, help="信用卡與信貸額度的使用比例")
    age = st.sidebar.slider("年齡 (Age)", 18, 100, 35)
    late_30 = st.sidebar.number_input("30-59天逾期次數", 0, 20, 0)
    debt = st.sidebar.number_input("負債比 (DebtRatio)", 0.0, 10.0, 0.3)
    income = st.sidebar.number_input("月收入 (MonthlyIncome)", 0, 1000000, 50000)
    open_loans = st.sidebar.number_input("開放式貸款數量", 0, 50, 5)
    late_90 = st.sidebar.number_input("90天以上逾期次數", 0, 20, 0)
    estate = st.sidebar.number_input("不動產貸款數量", 0, 20, 1)
    late_60 = st.sidebar.number_input("60-89天逾期次數", 0, 20, 0)
    dep = st.sidebar.number_input("家屬人數", 0, 20, 0)
    
    # 建立與訓練時完全相同的欄位名稱順序
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

# 3. 顯示輸入摘要與預測結果
st.subheader("📊 當前輸入特徵摘要")
st.write(input_df)

if st.button("🚀 開始計算風險"):
    if model is not None:
        # 預測違約機率 (類別 1 的機率)
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        st.subheader("🎯 評估結果")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("預測違約機率", f"{prediction_proba:.2%}")
        
        with col2:
            if prediction_proba > 0.5:
                st.error("❌ 高風險 (建議拒貸)")
            elif prediction_proba > 0.2:
                st.warning("⚠️ 中風險 (建議加強審核)")
            else:
                st.success("✅ 低風險 (准予核貸)")
        
        # 額外提示
        st.info("註：此結果由隨機森林模型根據歷史數據運算得出，僅供風控決策參考。")
    else:
        st.warning("系統尚未準備好，請檢查模型檔案是否存在。")
