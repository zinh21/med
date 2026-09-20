import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 뇌졸중 분류 모델 만들기")
st.write("나이·혈당·체질량지수·고혈압·심장병 정보로 뇌졸중 여부를 예측하는 모델을 만들어봅니다.")
st.caption("뇌졸중(stroke=1)을 '양성'으로 둡니다.")

st.divider()

# ---------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# 열 이름 <-> 우리말 이름 매핑
col_to_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
kor_to_col = {v: k for k, v in col_to_kor.items()}
all_kor_options = list(col_to_kor.values())

# ---------------------------------------------------
# 1. 입력 속성 선택
# ---------------------------------------------------
st.subheader("1️⃣ 입력 속성 선택")

default_kor = ["나이", "평균 혈당", "고혈압", "심장병"]  # bmi 제외한 기본값

selected_kor = st.multiselect(
    "모델에 사용할 속성을 골라주세요 (기본: 체질량지수 제외 4개)",
    options=all_kor_options,
    default=default_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 두 개 이상 선택해야 합니다. 목록에서 속성을 추가로 선택해주세요.")
    st.stop()

selected_cols = [kor_to_col[k] for k in selected_kor]
use_bmi = "bmi" in selected_cols

st.success(f"선택된 속성: {', '.join(selected_kor)}")

st.divider()

# ---------------------------------------------------
# 2. 데이터 준비 (훈련/테스트 분리, bmi 결측치 처리)
# ---------------------------------------------------
st.subheader("2️⃣ 데이터 준비")

df_sorted = df.sort_values("id").reset_index(drop=True)
df_sorted["group_order"] = df_sorted.index % 10  # 10명씩 묶었을 때 그룹 내 순서

is_test = df_sorted["group_order"] < 3  # 앞 3명 -> 테스트
train_df = df_sorted[~is_test].copy()
test_df = df_sorted[is_test].copy()

st.write(f"전체 인원: {len(df_sorted):,}명 → 훈련용 {len(train_df):,}명, 테스트용 {len(test_df):,}명")

# bmi 결측치를 훈련용 중앙값으로 채우기 (선택된 경우에만)
if use_bmi:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.write(f"체질량지수(bmi) 빈 값은 훈련용 중앙값 **{bmi_median:.2f}** 로 채웠습니다.")

# ---------------------------------------------------
# 3. 훈련 데이터 크기 맞추기 (언더샘플링)
# ---------------------------------------------------
train_pos = train_df[train_df["stroke"] == 1]
train_neg = train_df[train_df["stroke"] == 0]

n_pos = len(train_pos)
rng = np.random.RandomState(42)
train_neg_sampled = train_neg.sample(n=n_pos, random_state=42)

train_balanced = pd.concat([train_pos, train_neg_sampled]).sort_index().reset_index(drop=True)

st.write(f"크기를 맞춘 훈련 데이터: 뇌졸중 있음 {n_pos}명 + 뇌졸중 없음 {n_pos}명 = 총 {len(train_balanced)}명")

X_train = train_balanced[selected_cols]
y_train = train_balanced["stroke"]
X_test = test_df[selected_cols]
y_test = test_df["stroke"]

st.divider()

# ---------------------------------------------------
# 4. 모델 학습
# ---------------------------------------------------
st.subheader("3️⃣ 모델 학습 및 정확도 비교")

# 로지스틱 회귀
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)

# 의사결정트리 (질문 3번까지, 마지막 마디 5명 미만이면 더 나누지 않음)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# 더미 모델 (훈련용에서 많은 쪽으로만 답함)
dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)

# 정확도 계산 함수
def get_accuracies(model, X_train, y_train, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model, X_train, y_train, X_test, y_test)
tree_train_acc, tree_test_acc = get_accuracies(tree_model, X_train, y_train, X_test, y_test)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model, X_train, y_train, X_test, y_test)

# 카드 3개 나란히
card1, card2, card3 = st.columns(3)

with card1:
    st.markdown("**로지스틱 회귀 (확률로 답하는 모델)**")
    st.metric(label="테스트 정확도", value=f"{log_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {log_train_acc*100:.2f} %　|　테스트 정확도: {log_test_acc*100:.2f} %")

with card2:
    st.markdown("**의사결정트리 (질문으로 답하는 모델)**")
    st.metric(label="테스트 정확도", value=f"{tree_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {tree_train_acc*100:.2f} %　|　테스트 정확도: {tree_test_acc*100:.2f} %")

with card3:
    st.markdown("**기준 모델 (많은 쪽으로만 답하는 모델)**")
    st.metric(label="테스트 정확도", value=f"{dummy_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {dummy_train_acc*100:.2f} %　|　테스트 정확도: {dummy_test_acc*100:.2f} %")

st.divider()

# ---------------------------------------------------
# 5. 산점도 + 결정 경계
# ---------------------------------------------------
st.subheader("4️⃣ 두 속성으로 보는 결정 경계")

col_x_kor, col_y_kor = st.columns(2)
with col_x_kor:
    x_kor = st.selectbox("가로축으로 사용할 속성", options=selected_kor, index=0)
with col_y_kor:
    remaining = [k for k in selected_kor if k != x_kor]
    y_kor = st.selectbox("세로축으로 사용할 속성", options=remaining, index=0)

x_col = kor_to_col[x_kor]
y_col = kor_to_col[y_kor]
other_cols = [c for c in selected_cols if c not in [x_col, y_col]]

# 나머지 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
for c in other_cols:
    fixed_values[c] = test_df[c].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_to_kor[c]} = {v:.2f}" for c, v in fixed_values.items()])
    st.write(f"📌 그림에 나타나지 않는 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 두 개뿐이라 고정할 속성이 없습니다.")

# 격자 만들기
x_min, x_max = test_df[x_col].min(), test_df[x_col].max()
y_min, y_max = test_df[y_col].min(), test_df[y_col].max()
x_range = np.linspace(x_min, x_max, 100)
y_range = np.linspace(y_min, y_max, 100)
xx, yy = np.meshgrid(x_range, y_range)

grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for c, v in fixed_values.items():
    grid_df[c] = v
grid_df = grid_df[selected_cols]  # 학습 시 순서와 동일하게 맞춤

# 트리 모델의 칸(영역) 배경색
tree_pred_grid = tree_model.predict(grid_df).reshape(xx.shape)

fig_boundary = go.Figure()

fig_boundary.add_trace(go.Contour(
    x=x_range, y=y_range, z=tree_pred_grid,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "blue"], [1, "red"]],
    contours=dict(start=0, end=1, size=1),
    name="의사결정트리 영역",
    hoverinfo="skip"
))

# 로지스틱 회귀 0.5 경계선 계산
log_proba_grid = log_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

fig_boundary.add_trace(go.Contour(
    x=x_range, y=y_range, z=log_proba_grid,
    showscale=False,
    contours=dict(
        start=0.5, end=0.5, size=0.1,
        coloring="lines"
    ),
    line=dict(width=3, color="black"),
    name="로지스틱 회귀 경계선(0.5)",
    hoverinfo="skip"
))

# 경계선이 그림 범위 안에 있는지 확인
boundary_exists = (log_proba_grid.min() < 0.5 < log_proba_grid.max())
if not boundary_exists:
    st.info("ℹ️ 로지스틱 회귀의 0.5 경계선이 이 그림의 범위 안에는 나타나지 않습니다 (그림 밖에 있음).")

# 테스트 데이터 산점도
test_df_plot = test_df.copy()
test_df_plot["실제 뇌졸중 여부"] = test_df_plot["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

for label, color in [("뇌졸중 없음", "blue"), ("뇌졸중 있음", "red")]:
    subset = test_df_plot[test_df_plot["실제 뇌졸중 여부"] == label]
    fig_boundary.add_trace(go.Scatter(
        x=subset[x_col], y=subset[y_col],
        mode="markers",
        name=label,
        marker=dict(color=color, size=6, opacity=0.7)
    ))

fig_boundary.update_layout(
    title="테스트 데이터 산점도와 결정 경계",
    xaxis_title=x_kor,
    yaxis_title=y_kor,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(fig_boundary, use_container_width=True)

st.divider()

# ---------------------------------------------------
# 6. 의사결정트리 가지 그림 (graphviz)
# ---------------------------------------------------
st.subheader("5️⃣ 의사결정트리가 던진 질문")

tree = tree_model.tree_
feature_names = selected_cols

def build_dot(tree, feature_names, kor_names):
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value  # [n_nodes, 1, n_classes] 각 클래스별 인원수

    for i in range(n_nodes):
        counts = value[i][0]  # [클래스0 인원, 클래스1 인원]
        total = int(counts.sum())
        pos_count = int(counts[1])
        ratio = pos_count / total * 100 if total > 0 else 0

        is_leaf = (children_left[i] == children_right[i])

        if is_leaf:
            pred_class = int(np.argmax(counts))
            if pred_class == 1:
                fill_color = "lightcoral"   # 뇌졸중 있음으로 답함
                pred_label = "뇌졸중 있음"
            else:
                fill_color = "lightblue"    # 뇌졸중 없음으로 답함
                pred_label = "뇌졸중 없음"
            label = f"인원 {total}명\\n뇌졸중 {pos_count}명 ({ratio:.1f}%)\\n답: {pred_label}"
        else:
            col_name = feature_names[feature[i]]
            kor_name = kor_names.get(col_name, col_name)
            thresh_val = threshold[i]
            fill_color = "lightyellow"
            label = f"{kor_name} <= {thresh_val:.2f} ?\\n인원 {total}명\\n뇌졸중 {pos_count}명 ({ratio:.1f}%)"

        dot_lines.append(f'{i} [label="{label}", fillcolor="{fill_color}"];')

        if not is_leaf:
            left = children_left[i]
            right = children_right[i]
            dot_lines.append(f'{i} -> {left} [label="예"];')
            dot_lines.append(f'{i} -> {right} [label="아니요"];')

    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_str = build_dot(tree, feature_names, col_to_kor)
st.graphviz_chart(dot_str)

# ---------------------------------------------------
# 7. 트리 요약 설명
# ---------------------------------------------------
st.subheader("6️⃣ 의사결정트리 요약")

n_nodes = tree.node_count
children_left = tree.children_left
children_right = tree.children_right
value = tree.value
feature = tree.feature

leaf_indices = [i for i in range(n_nodes) if children_left[i] == children_right[i]]
n_leaves = len(leaf_indices)

n_leaves_negative = 0
for i in leaf_indices:
    counts = value[i][0]
    pred_class = int(np.argmax(counts))
    if pred_class == 0:
        n_leaves_negative += 1

used_feature_indices = sorted(set(feature[i] for i in range(n_nodes) if children_left[i] != children_right[i]))
used_features_kor = [col_to_kor[feature_names[idx]] for idx in used_feature_indices]

st.write(f"- 답을 내는 마디(잎마디)는 모두 **{n_leaves}칸**이고, 그중 **{n_leaves_negative}칸**이 '뇌졸중 없음'이라고 답합니다.")
if used_features_kor:
    st.write(f"- 이 나무가 실제로 질문에 사용한 속성: **{', '.join(used_features_kor)}**")
else:
    st.write("- 이 나무는 어떤 속성도 실제로 사용하지 않았습니다 (뿌리마디가 곧 잎마디입니다).")
