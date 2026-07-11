import streamlit as st
import pandas as pd
from tech_stack_recommender import TechStackRecommender

st.set_page_config(
    page_title="Tech Stack Recommender",
    page_icon="�",
    layout="wide"
)

st.title("Tech Stack Career Recommender")
st.markdown("### Find your ideal tech career based on your skills")
st.markdown("---")

@st.cache_resource
def load_recommender():
    recommender = TechStackRecommender(use_pca=True, n_components=20)
    recommender.load_data()
    recommender.build_model()
    return recommender

with st.spinner("Loading recommendation engine..."):
    recommender = load_recommender()

st.success("System ready!")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Your Skills")
    st.markdown("Enter at least **3 skills** you possess (e.g., Python, Docker, React)")
    
    skill_input = st.text_area(
        "Skills (one per line or comma-separated)",
        height=150,
        placeholder="Python\nMachine Learning\nDocker\nAWS\nJavaScript"
    )
    
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=3)
    
    recommend_button = st.button("Get Recommendations", type="primary", use_container_width=True)

with col2:
    st.subheader("Popular Skills")
    popular_skills = [
        "Python", "JavaScript", "Java", "React", "Node.js",
        "Docker", "Kubernetes", "AWS", "Machine Learning",
        "SQL", "MongoDB", "TensorFlow", "REST API"
    ]
    st.markdown("\n".join([f"• {skill}" for skill in popular_skills[:10]]))

st.markdown("---")

if recommend_button:
    if skill_input.strip():
        if ',' in skill_input:
            skills = [s.strip() for s in skill_input.split(',') if s.strip()]
        else:
            skills = [s.strip() for s in skill_input.split('\n') if s.strip()]
        
        if len(skills) < 3:
            st.error("Please enter at least 3 skills!")
        else:
            st.info(f"Analyzing your skills: **{', '.join(skills)}**")
            
            with st.spinner("Finding best career matches..."):
                recommendations = recommender.get_recommendations(skills, top_n=top_n)
            
            if recommendations is not None and len(recommendations) > 0:
                st.success(f"Found {len(recommendations)} career matches for you!")
                st.markdown("---")
                
                for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.markdown(f"### {idx}. {row['job_role']}")
                            st.markdown(f"**Category:** {row.get('category', 'N/A')} | **Salary Range:** {row.get('salary_range', 'N/A')}")
                            
                            match_score = row['match_percentage']
                            st.progress(match_score / 100)
                            
                            if match_score >= 90:
                                match_label = "Excellent Match"
                                color = "green"
                            elif match_score >= 75:
                                match_label = "Strong Match"
                                color = "blue"
                            elif match_score >= 60:
                                match_label = "Good Match"
                                color = "orange"
                            else:
                                match_label = "Potential Match"
                                color = "gray"
                            
                            st.markdown(f"**Match Score:** :{color}[{match_score:.1f}%] - {match_label}")
                        
                        with col_b:
                            st.metric("Match Score", f"{match_score:.1f}%")
                        
                        with st.expander("Required Skills"):
                            skills_list = row['required_skills'].replace('_', ' ').split()
                            st.markdown(" • ".join(skills_list))
                        
                        st.markdown("---")
            else:
                st.warning("No recommendations found. Try different skills.")
    else:
        st.error("Please enter your skills first!")

st.markdown("---")
with st.expander("How It Works"):
    st.markdown("""
    This recommendation system uses:
    - **TF-IDF Vectorization** - Converts skills into numerical features
    - **PCA Dimensionality Reduction** - Optimizes processing efficiency
    - **Cosine Similarity** - Measures match between your skills and job requirements
    
    The system analyzes 25+ tech careers across categories like AI/ML, DevOps, Development, Cloud, and more.
    """)

with st.expander("Available Job Categories"):
    if recommender.job_data is not None:
        categories = recommender.job_data['category'].value_counts()
        st.bar_chart(categories)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Tech Stack Recommender System | Built with Streamlit & Python</p>
    </div>
    """,
    unsafe_allow_html=True
)
