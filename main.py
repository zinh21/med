import streamlit as st
import pandas as pd

# ---------------------------------------------------
# 페이지 기본 설정 (브라우저 탭 제목 & 아이콘)
# ---------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# 앱 제목
# ---------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 발생과 관련된 데이터를 살펴보고 탐구하는 공간입니다.")

st.divider()

# ---------------------------------------------------
# 데이터 불러오기 (캐싱으로 속도 향상)
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------
# 큰 숫자 카드 4개
# ---------------------------------------------------
st.subheader("📌 데이터 한눈에 보기")

total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = round(stroke_count / total_people * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열(컬럼) 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중 발생자 수", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 발생 비율", value=f"{stroke_ratio} %")

st.divider()

# ---------------------------------------------------
# 열 이름 / 우리말 뜻 / 값의 종류 / 빈 값 개수 표
# ---------------------------------------------------
st.subheader("📋 열(컬럼) 정보 살펴보기")
st.caption("※ '우리말 뜻' 칸은 비어 있습니다. 교재를 참고하여 아래 코드의 빈칸을 직접 채워보세요!")

# 각 열의 값 종류를 보기 좋게 정리하는 함수
def get_value_types(series):
    unique_vals = series.dropna().unique()
    if series.dtype == "object" or len(unique_vals) <= 10:
        # 범주형이거나 종류가 적은 경우 -> 값들을 나열
        vals = sorted(unique_vals.astype(str))
        return ", ".join(vals)
    else:
        # 숫자형이고 값이 많은 경우 -> 최소~최대 범위로 표시
        return f"숫자 값 ({series.min()} ~ {series.max()})"

# ----- 여기 우리말 뜻만 직접 채워 넣으세요! -----
korean_meaning = {
    "id": "",
    "gender": "",
    "age": "",
    "hypertension": "",
    "heart_disease": "",
    "ever_married": "",
    "work_type": "",
    "Residence_type": "",
    "avg_glucose_level": "",
    "bmi": "",
    "smoking_status": "",
    "stroke": "",
}
# ------------------------------------------------

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [korean_meaning.get(col, "") for col in df.columns],
    "값의 종류": [get_value_types(df[col]) for col in df.columns],
    "빈 값 개수": [df[col].isnull().sum() for col in df.columns],
})

st.dataframe(column_info, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------
# 데이터 처음 5줄 미리보기
# ---------------------------------------------------
st.subheader("👀 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# ---------------------------------------------------
# 데이터 출처 (직접 작성)
# ---------------------------------------------------
st.subheader("📚 데이터 출처")
st.info("아래에 교재에 나온 데이터 출처 문장을 직접 적어보세요.")

# ----- 여기에 교재 내용을 참고해서 출처를 직접 작성하세요! -----
source_text = """
(여기에 교재에 나온 데이터 출처 문장을 적어보세요)
"""
# ---------------------------------------------------------

st.write(source_text)
