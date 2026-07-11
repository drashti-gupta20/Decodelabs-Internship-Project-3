import pandas as pd
import numpy as np
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')


class TechStackRecommender:

    
    def __init__(self, data_path='raw_skills.csv', use_pca=True, n_components=20):
        
        self.data_path = data_path
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.job_data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.pca_model = None
        self.scaler = None
        self.reduced_matrix = None
        
        self.skill_synonyms = {
            'js': 'javascript', 'py': 'python', 'ts': 'typescript',
            'ml': 'machine_learning', 'ai': 'artificial_intelligence',
            'k8s': 'kubernetes', 'aws': 'amazon_web_services',
            'gcp': 'google_cloud_platform', 'api': 'rest_api',
            'db': 'database', 'devops': 'dev_ops', 'cicd': 'ci_cd'
        }
    
    def load_data(self):
        try:
            self.job_data = pd.read_csv(self.data_path)
            print(f"[OK] Loaded {len(self.job_data)} job roles from dataset")
            
            required_cols = ['job_role', 'required_skills']
            if not all(col in self.job_data.columns for col in required_cols):
                print("Creating sample dataset...")
                self.create_sample_dataset()
            
            return True
        except FileNotFoundError:
            print(f"Dataset not found. Creating sample dataset...")
            self.create_sample_dataset()
            return True
    
    def create_sample_dataset(self):
        data = {
            'job_role': [
                'Data Scientist', 'Machine Learning Engineer', 'DevOps Engineer',
                'Backend Developer', 'Frontend Developer', 'Full Stack Developer',
                'Cloud Architect', 'Mobile Developer', 'Data Engineer',
                'AI Research Scientist', 'Security Engineer', 'Database Administrator',
                'Software Architect', 'QA Engineer', 'Blockchain Developer',
                'Game Developer', 'Systems Administrator', 'Network Engineer',
                'UI/UX Designer', 'Product Manager', 'Site Reliability Engineer',
                'MLOps Engineer', 'Cloud Engineer', 'API Developer', 'Embedded Systems Engineer'
            ],
            'required_skills': [
                'Python SQL Machine_Learning Data_Analysis Pandas NumPy Scikit-learn TensorFlow Statistics',
                'Python Machine_Learning Deep_Learning TensorFlow PyTorch Neural_Networks Computer_Vision NLP',
                'Docker Kubernetes AWS CI/CD Jenkins Terraform Ansible Linux Python Automation',
                'Java Python Node.js SQL REST_API Microservices Spring Django PostgreSQL MongoDB',
                'JavaScript React Angular Vue.js HTML CSS TypeScript Redux Webpack REST_API',
                'JavaScript Python React Node.js SQL MongoDB Express HTML CSS Docker',
                'AWS Azure GCP Cloud_Architecture Kubernetes Docker Terraform Networking Security',
                'Java Kotlin Swift React_Native Android iOS Mobile_Development REST_API SQLite',
                'Python SQL ETL Apache_Spark Apache_Kafka Data_Warehousing Airflow Hadoop BigQuery',
                'Python Machine_Learning Deep_Learning TensorFlow PyTorch Research Neural_Networks NLP',
                'Python Cybersecurity Networking Penetration_Testing Security_Tools Firewalls Linux',
                'SQL Database_Design PostgreSQL MySQL Oracle MongoDB Performance_Tuning Backup_Recovery',
                'Java Python System_Design Architecture Microservices Design_Patterns Cloud AWS',
                'Python Selenium Testing Automation Java QA API_Testing Performance_Testing CI/CD',
                'Solidity Ethereum Blockchain Smart_Contracts Web3 Cryptography JavaScript Node.js',
                'C++ C# Unity Unreal_Engine 3D_Modeling Game_Design Animation Python Scripting',
                'Linux Windows Scripting Bash PowerShell Networking Automation Monitoring Security',
                'Networking Cisco Routers Switches TCP/IP Firewalls VPN Security Linux',
                'Figma Adobe_XD UI_Design UX_Design Prototyping User_Research HTML CSS JavaScript',
                'Product_Management Agile Scrum Data_Analysis Communication SQL Python Leadership',
                'Python Linux Docker Kubernetes Monitoring Automation Terraform CI/CD AWS Prometheus',
                'Python Machine_Learning MLOps Kubernetes Docker CI/CD Airflow TensorFlow Monitoring',
                'AWS Azure GCP Cloud_Computing Kubernetes Docker Terraform CI/CD Python Networking',
                'Python Node.js REST_API GraphQL API_Design Microservices Docker Kubernetes Swagger',
                'C C++ Embedded_Systems RTOS Microcontrollers IoT Hardware Linux ARM Debugging'
            ],
            'salary_range': [
                '90k-150k', '100k-180k', '85k-140k', '80k-130k', '75k-125k', '85k-145k',
                '110k-170k', '80k-135k', '90k-155k', '120k-200k', '95k-160k', '80k-125k',
                '120k-180k', '70k-115k', '100k-170k', '85k-145k', '75k-120k', '80k-130k',
                '70k-120k', '100k-160k', '110k-175k', '105k-170k', '95k-150k', '85k-140k', '90k-145k'
            ],
            'category': [
                'AI/ML', 'AI/ML', 'DevOps', 'Development', 'Development', 'Development',
                'Cloud', 'Mobile', 'Data', 'AI/ML', 'Security', 'Database',
                'Architecture', 'QA', 'Blockchain', 'Gaming', 'Systems', 'Networking',
                'Design', 'Management', 'DevOps', 'AI/ML', 'Cloud', 'Development', 'Embedded'
            ]
        }
        
        self.job_data = pd.DataFrame(data)
        self.job_data.to_csv(self.data_path, index=False)
        print(f"Created sample dataset with {len(self.job_data)} tech jobs")
    
    def vectorize_text_feature(self, texts, max_features=50):

        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=1,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=max_features,
            lowercase=True,
            token_pattern=r'\b[A-Za-z_]+\b'
        )
        
        processed_texts = [text.lower().replace('_', ' ') for text in texts]
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        return self.tfidf_matrix
    
    def apply_pca(self, matrix, n_components=20):

        self.scaler = StandardScaler()
        scaled_matrix = self.scaler.fit_transform(matrix.toarray())
        
        self.pca_model = PCA(n_components=n_components)
        reduced_matrix = self.pca_model.fit_transform(scaled_matrix)
        
        variance_explained = self.pca_model.explained_variance_ratio_.sum()
        print(f"[OK] PCA: {matrix.shape[1]} dims -> {n_components} dims")
        print(f"[OK] Variance explained: {variance_explained:.2%}")
        
        return reduced_matrix
    
    def build_model(self):
        print("\nBuilding recommendation model...")
        
        self.tfidf_matrix = self.vectorize_text_feature(
            self.job_data['required_skills'].values,
            max_features=50
        )
        
        print(f"[OK] TF-IDF matrix: {self.tfidf_matrix.shape}")
        
        if self.use_pca and self.tfidf_matrix.shape[1] > self.n_components:
            self.reduced_matrix = self.apply_pca(self.tfidf_matrix, self.n_components)
        else:
            self.reduced_matrix = self.tfidf_matrix.toarray()
            print("INFO: Skipping PCA (not enough features or disabled)")
    
    def normalize_skills(self, skills):
        normalized = []
        for skill in skills:
            skill_lower = skill.lower().strip().replace(' ', '_')
            skill_normalized = self.skill_synonyms.get(skill_lower, skill_lower)
            normalized.append(skill_normalized)
        return normalized
    
    def validate_input(self, skills):
        if not skills or len(skills) < 3:
            print("ERROR: Minimum 3 skills required!")
            return False
        
        skills = [s.strip() for s in skills if s.strip()]
        if len(skills) < 3:
            print("ERROR: After filtering, less than 3 valid skills!")
            return False
        
        return True
    
    def get_user_input(self):
        print("\n" + "="*70)
        print("TECH STACK RECOMMENDER - Enter Your Skills")
        print("="*70)
        print("Enter at least 3 skills you possess.")
        print("Examples: Python, Cloud, Docker, JavaScript, Machine Learning")
        print("-"*70)
        
        skills = []
        while len(skills) < 3:
            skill = input(f"Skill {len(skills) + 1}: ").strip()
            if skill:
                skills.append(skill)
        
        while True:
            more = input(f"\nAdd another skill? (y/n): ").strip().lower()
            if more == 'y':
                skill = input(f"Skill {len(skills) + 1}: ").strip()
                if skill:
                    skills.append(skill)
            else:
                break
        
        return skills
    
    def calculate_similarity(self, user_skills):
        normalized_skills = self.normalize_skills(user_skills)
        user_text = ' '.join(normalized_skills)
        processed_text = user_text.lower().replace('_', ' ')
        
        user_vector = self.tfidf_vectorizer.transform([processed_text])
        
        if self.use_pca and self.pca_model is not None:
            user_scaled = self.scaler.transform(user_vector.toarray())
            user_reduced = self.pca_model.transform(user_scaled)
            similarity_scores = cosine_similarity(user_reduced, self.reduced_matrix)[0]
        else:
            similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        
        return similarity_scores
    
    def get_recommendations(self, user_skills, top_n=3):
        if not self.validate_input(user_skills):
            return None
        
        print(f"\nAnalyzing your skills: {', '.join(user_skills)}")
        similarity_scores = self.calculate_similarity(user_skills)
        
        results = self.job_data.copy()
        results['similarity_score'] = similarity_scores
        results['match_percentage'] = (similarity_scores * 100).round(2)
        results = results.sort_values('similarity_score', ascending=False)
        
        top_recommendations = results.head(top_n)
        
        return top_recommendations
    
    def display_recommendations(self, recommendations):

        if recommendations is None or len(recommendations) == 0:
            print("\nWARNING: No recommendations found.")
            return
        
        print("\n" + "="*70)
        print("TOP CAREER RECOMMENDATIONS FOR YOU")
        print("="*70)
        
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"\n{idx}. {row['job_role']}")
            print(f"   Match Score: {row['match_percentage']:.1f}%")
            print(f"   Category: {row.get('category', 'N/A')}")
            print(f"   Salary Range: {row.get('salary_range', 'N/A')}")
            
            bar_length = int(row['match_percentage'] / 5)
            bar = '#' * bar_length + '-' * (20 - bar_length)
            print(f"   Match Level: [{bar}]")
            
            skills_preview = row['required_skills'][:60]
            print(f"   Key Skills: {skills_preview}...")
        
        print("\n" + "="*70)
    
    def handle_cold_start(self):
        print("\n" + "="*70)
        print("NEW USER DETECTED")
        print("="*70)
        print("\n1. Complete skill survey")
        print("2. View trending careers")
        print("3. Enter skills manually")
        
        choice = input("\nYour choice (1-3): ").strip()
        
        if choice == '1':
            return self.skill_survey()
        elif choice == '2':
            self.show_trending()
            return None
        else:
            return None
    
    def skill_survey(self):
        print("\nSelect skills from popular options:")
        all_skills = set()
        for skills_str in self.job_data['required_skills']:
            all_skills.update(skills_str.split())
        
        popular = sorted(list(all_skills))[:15]
        for i, skill in enumerate(popular, 1):
            print(f"{i}. {skill.replace('_', ' ')}", end="  ")
            if i % 3 == 0:
                print()
        
        print("\n\nEnter numbers separated by commas (e.g., 1,5,8):")
        choices = input("Your choices: ").strip()
        
        try:
            indices = [int(x.strip()) - 1 for x in choices.split(',')]
            selected = [popular[i] for i in indices if 0 <= i < len(popular)]
            return selected if len(selected) >= 3 else None
        except:
            return None
    
    def show_trending(self):
        print("\nTRENDING TECH CAREERS:")
        for idx, (_, row) in enumerate(self.job_data.head(5).iterrows(), 1):
            print(f"{idx}. {row['job_role']} ({row['salary_range']})")
    
    def run(self, user_skills=None, top_n=3):
        self.load_data()
        
        self.build_model()
        
        if user_skills is None:
            skills = self.get_user_input()
        else:
            skills = user_skills
        
        if skills is None or len(skills) == 0:
            skills = self.handle_cold_start()
            if skills is None:
                skills = self.get_user_input()
        
        if skills:
            recommendations = self.get_recommendations(skills, top_n=top_n)
            self.display_recommendations(recommendations)
            return recommendations
        
        return None


def main():
    print("""
    ==================================================================
    
                  TECH STACK RECOMMENDER SYSTEM                    
    
    ==================================================================
    """)
    
    recommender = TechStackRecommender(use_pca=True, n_components=20)
    
    recommender.run(top_n=3)
    
    print("\n" + "-"*70)
    again = input("Would you like another recommendation? (y/n): ").strip().lower()
    if again == 'y':
        skills = recommender.get_user_input()
        recommendations = recommender.get_recommendations(skills, top_n=3)
        recommender.display_recommendations(recommendations)


if __name__ == "__main__":
    main()
