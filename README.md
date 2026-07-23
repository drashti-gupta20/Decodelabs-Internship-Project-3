# Tech Stack Recommender System

A content-based recommendation system that matches users with ideal tech careers based on their skills using TF-IDF vectorization, PCA dimensionality reduction, and cosine similarity.

## Features

- **TF-IDF Vectorization** - Advanced text feature extraction
- **PCA Dimensionality Reduction** - Efficient processing (50 -> 20 dimensions)
- **Cosine Similarity** - Industry-standard similarity metric
- **User Skill Input** - Enter your skills and get matched careers
- **Minimum 3 Skills Validation** - Ensures accurate recommendations
- **Top-N Recommendations** - Get your top 3 career matches
- **Cold Start Handling** - Survey and trending options for new users
- **Visual Output** - Match bars and detailed career information

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- pandas
- numpy
- scikit-learn

## Example Output

```
Analyzing your skills: Python, Machine Learning, Docker

======================================================================
TOP CAREER RECOMMENDATIONS FOR YOU
======================================================================

1. Machine Learning Engineer
   Match Score: 89.5%
   Category: AI/ML
   Salary Range: 100k-180k
   Match Level: [#################---]

2. Data Scientist
   Match Score: 85.2%
   Category: AI/ML
   Salary Range: 90k-150k
   Match Level: [#################---]

3. MLOps Engineer
   Match Score: 78.3%
   Category: AI/ML
   Salary Range: 105k-170k
   Match Level: [###############-----]
```

## Project Structure

```
jobs_recommendation_system/
├── tech_stack_recommender.py   # Main implementation
├── app.py                       # Streamlit web interface
├── raw_skills.csv               # Tech careers dataset (25 jobs)
├── requirements.txt             # Python dependencies
├── START_HERE.md                # Detailed documentation
└── README.md                    # This file
```

### Change PCA Components
```python
rec = TechStackRecommender(use_pca=True, n_components=30)
```

### Get More Recommendations
```python
results = rec.get_recommendations(skills, top_n=5)
```
.

## Dataset

The system includes 25 pre-configured tech careers across categories:
- AI/ML (Data Scientist, ML Engineer, AI Research Scientist, MLOps Engineer)
- DevOps (DevOps Engineer, Site Reliability Engineer)
- Development (Backend, Frontend, Full Stack, API Developer)
- Cloud (Cloud Architect, Cloud Engineer)
- Security, Database, Mobile, Gaming, and more

### Adding Custom Jobs
Edit `raw_skills.csv`:
```csv
job_role,required_skills,salary_range,category
"Your Job Title","Skill1 Skill2 Skill3","100k-200k","Category"
```

**Test Coverage:**
- Data Science skills
- DevOps skills
- Full Stack skills
- Skill abbreviation normalization
- Minimum 3 skills validation
- Input rejection for less than 3 skills


## Understanding Match Scores

- **90-100 percent**: Excellent match - you have most required skills
- **75-89 percent**: Strong match - good career fit
- **60-74 percent**: Moderate match - some skill gaps
- **Below 60 percent**: Weak match - significant skill development needed

## Technical Details

- **Algorithm**: Content-Based Filtering
- **Vectorization**: TF-IDF with ngram_range=(1,2)
- **Dimensionality Reduction**: PCA (50 -> 20 components)
- **Similarity Metric**: Cosine Similarity
- **Processing Time**: approximately 1-2 seconds
- **Variance Explained**: 99.32 percent


## Get Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `python test_recommender.py`
3. Try interactive mode: `python tech_stack_recommender.py`

web link: https://techstack-recommender.streamlit.app/
