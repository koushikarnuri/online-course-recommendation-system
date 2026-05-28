import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Hybrid Course Recommender",
    layout="wide"
)

st.title(" Course Recommendation System")


@st.cache_data
def load_data():
    return joblib.load("hybrid_recommendation_model.joblib")

data = load_data()

df = data["df"]
scaler_similarity = data["scaler_similarity"]
similarity_features = data["similarity_features"]

st.sidebar.header("Select a Course You Like")

selected_course = st.sidebar.selectbox(
    "Choose a course",
    df["course_name"].unique()
)

selected_course_id = df[df["course_name"] == selected_course]["course_id"].iloc[0]

st.success(f"Hii the course you selected is **{selected_course}**Based on this selection, the system analyzes course features and popularity to recommend relevant and high-quality courses.")

base_course = df[df["course_id"] == selected_course_id].iloc[0]
base_vector = base_course[similarity_features].values.reshape(1, -1)

base_scaled = scaler_similarity.transform(base_vector)
df_scaled = scaler_similarity.transform(df[similarity_features])

df["similarity_score"] = cosine_similarity(
    base_scaled,
    df_scaled
).flatten()

df["hybrid_score"] = (
    0.7 * df["similarity_score"] +
    0.3 * (df["enrollment_numbers"] / df["enrollment_numbers"].max())
)

st.subheader(" Recommended Courses")

recommended_courses = (
    df[df["course_id"] != selected_course_id]
    .sort_values("hybrid_score", ascending=False)
    .head(5)
)

st.dataframe(
    recommended_courses[
        [
            "course_name",
            "difficulty_level",
            "certification_offered",
            "course_price",
            "feedback_score",
            "rating",
            "enrollment_numbers",
        ]
    ],
    use_container_width=True
)


st.subheader(" Popular Courses")

df["enroll_norm"] = df["enrollment_numbers"] / df["enrollment_numbers"].max()
df["rating_norm"] = df["rating"] / df["rating"].max()

df["popular_score"] = (
    0.6 * df["enroll_norm"] +
    0.4 * df["rating_norm"]
)

popular_courses = (
    df.sort_values("popular_score", ascending=False)
    .drop_duplicates("course_id")
    .head(5)
)

st.dataframe(
    popular_courses[
        [
            "course_name",
            "difficulty_level",
            "certification_offered",
            "course_price",
            "feedback_score",
            "rating",
            "enrollment_numbers",
        ]
    ],
    use_container_width=True
)

st.subheader(" Course Summary")

course_info = df[df["course_id"] == selected_course_id].iloc[0]
recommended_names = ", ".join(recommended_courses["course_name"].tolist())

st.write(
    f"""
The selected course **{course_info['course_name']}** offers a 
**{course_info['certification_offered']}** certification and has an overall
rating of **{course_info['rating']}**. The course duration is approximately
**{course_info['course_duration_hours']} hours**, making it suitable for learners
at the **{course_info['difficulty_level']}** level.

Based on your selection, the system recommends similar courses such as
**{recommended_names}**, identified using a hybrid approach that combines
content similarity and course popularity (enrollment and ratings).
"""
)

