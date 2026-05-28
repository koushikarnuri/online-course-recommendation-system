# Online Course Recommendation System

## Overview
An intelligent course recommendation system that suggests relevant online courses
to users based on their preferences, behavior, and course attributes. Built using
**5 machine learning algorithms** on a dataset of **100,000+ records** with **14 features**,
finalized with a **Hybrid Model** combining Content-Based and Collaborative Filtering.

> 🚀 **Live Demo:** [recommednation-system-tpsknk3juvsm6apdkaknox.streamlit.app](https://recommednation-system-tpsknk3juvsm6apdkaknox.streamlit.app/)

---

## Dataset
| Attribute | Details |
|---|---|
| Total Records | 10,000 user-course interactions |
| Features | 14 attributes (numeric and categorical) |
| Missing Values | 1.8% (handled via preprocessing) |
| Average Rating | 4.05 / 5.0 |
| Difficulty Split | Beginner 33.2% / Intermediate 33.5% / Advanced 33.3% |

### Feature Categories
- **Course Attributes:** course_name, instructor, duration, difficulty_level, rating
- **Engagement Metrics:** enrollment_numbers, time_spent_hours, feedback_score
- **User Profile:** user_id, previous_courses_taken
- **Course Features:** certification_offered, study_material_available, course_price

---

## Models Built (5 Algorithms)

| # | Model | Approach | Key Detail |
|---|---|---|---|
| 1 | Content-Based KNN on PCA | Dimensionality reduction + KNN | 7 PCA components, 71% variance explained |
| 2 | User-Based Collaborative Filtering | KNN on user-item matrix | k=20 similar users, cosine similarity |
| 3 | Matrix Factorization (SVD) | Singular Value Decomposition | Handles sparse data, predicts unknown ratings |
| 4 | **Hybrid Model (Selected ✅)** | 60% Content + 40% Collaborative | Best overall performance |
| 5 | Content-Based Filtering (TF-IDF) | Text similarity | 2000 max features, cosine similarity |

---

## Finalized Model — Hybrid Approach

```
Final Score = 0.6 × Content-Based (TF-IDF) + 0.4 × Collaborative Filtering
```

- **Content-Based Rating:** 4.2
- **Collaborative Rating:** 4.1
- **Hybrid Final Rating:** 4.16
- Selected for production deployment for best accuracy and coverage

---

## Data Processing Pipeline

1. **Data Loading** — Imported dataset with 10,000 records and 14 features
2. **Missing Value Analysis** — Identified and handled 1.8% missing values
3. **Outlier Detection** — IQR method and Z-scores for anomaly identification
4. **Feature Encoding** — Label encoding for categorical variables
5. **Feature Scaling** — StandardScaler for numerical feature normalization
6. **Feature Engineering** — Created `experience_level` based on courses taken
   - 0–3 courses: Novice | 4–7: Intermediate | 8+: Expert

---

## Tech Stack
Python | scikit-learn | pandas | NumPy | Streamlit | TF-IDF | SVD | KNN | PCA | Matplotlib | Seaborn

---

## Project Structure
```
online-course-recommendation-system/
│
├── week_2_coure_recomednation_system.ipynb   ← Main Jupyter notebook
├── app.py                                     ← Streamlit web application
├── requirements.txt                           ← Required libraries
├── Online_Course_Recommendation_Dataset       ← Dataset file
├── online_course_recommendation_v2 (3) (1)   ← Updated notebook version
├── hybrid_recommendation_model.joblib         ← Saved hybrid model (production)
└── Group_2_online_course_recommedation_.pptx  ← Project presentation
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Open the main notebook**
```bash
jupyter notebook week_2_coure_recomednation_system.ipynb
```

**3. Run the Streamlit app locally**
```bash
streamlit run app.py
```

**4. Or visit the live deployment**
> [recommednation-system-tpsknk3juvsm6apdkaknox.streamlit.app](https://recommednation-system-tpsknk3juvsm6apdkaknox.streamlit.app/)

---

## Key Results
- ✅ Hybrid model outperformed all individual approaches
- ✅ Strong correlation between course rating and enrollment (r=0.73)
- ✅ Cold start problem solved using content-based fallback for new users
- ✅ System deployed and live on Streamlit Cloud
- ✅ Scalable architecture ready for production

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| Cold Start Problem | Content-based filtering fallback for new users |
| Data Imbalance | Stratified sampling and weighted features |
| Slow TF-IDF Computation | Pre-computed similarity matrices with caching |
| Model Selection | Hybrid approach with comparative evaluation framework |

---

## Future Enhancements
- Neural collaborative filtering and transformer-based embeddings
- Real-time personalization with live user behavior tracking
- Multi-modal data support (video, audio content)
- Skill gap analysis and learning path sequencing
- Distributed computing for large-scale deployment

---

## Team
**Group 2** — Koushik, Indrajeet, Dharitri, Athul, Aatlee, Vipul
**Mentor:** Karthik | **Co-Mentor:** Stephy

## Author
**Chenna Sai Mani Koushik Arnuri**
- Email: koushikarnuri@gmail.com
- LinkedIn: [linkedin.com/in/koushik-arnuri-920195387](https://www.linkedin.com/in/koushik-arnuri-920195387)
- GitHub: [github.com/koushikarnuri](https://github.com/koushikarnuri)
