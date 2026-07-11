# How to Deploy Your Tech Stack Recommender Online

## Test Locally First

1. Install Streamlit:
```bash
pip install streamlit
```

2. Run the app locally:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

---

## Deploy Online (FREE) - Streamlit Cloud

### Step 1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and create a new repository
2. Name it something like `tech-stack-recommender`
3. Make it **Public** (required for free hosting)

### Step 2: Upload Your Files

Upload these files to your GitHub repo:
- `app.py`
- `tech_stack_recommender.py`
- `requirements.txt`
- `raw_skills.csv`

You can do this via:
- GitHub web interface (drag and drop)
- Git commands (if you have Git installed)

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Sign in with your GitHub account
4. Select your repository: `tech-stack-recommender`
5. Set main file path: `app.py`
6. Click **"Deploy"**

### Step 4: Wait 2-3 Minutes

Streamlit will:
- Install all dependencies from `requirements.txt`
- Build your app
- Give you a public URL like: `https://your-app-name.streamlit.app`

---

## Alternative Free Hosting Options

### Option 2: Render.com

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Create new "Web Service"
4. Connect GitHub repo
5. Build command: `pip install -r requirements.txt`
6. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### Option 3: Hugging Face Spaces

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space
3. Choose "Streamlit" SDK
4. Upload your files
5. Done!

---

## Quick Git Commands (If Needed)

If you want to use Git to push to GitHub:

```bash
cd c:\Users\gauta\Downloads\ChatBot\jobs_recommendation_system

git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/tech-stack-recommender.git
git push -u origin main
```

---

## Troubleshooting

### Error: "Module not found"
- Make sure `streamlit` is in `requirements.txt`
- Check all file names are correct

### App crashes on startup
- Test locally first: `streamlit run app.py`
- Check the logs in Streamlit Cloud dashboard

### CSV file not found
- Make sure `raw_skills.csv` is uploaded to GitHub
- Check file path in code matches actual file name

---

## What You'll Get

After deployment, you'll have:
- ✅ Public URL anyone can access
- ✅ Professional-looking web interface
- ✅ No server management needed
- ✅ Free forever (with usage limits)
- ✅ Auto-updates when you push to GitHub

**Example URL:** `https://tech-stack-recommender-yourname.streamlit.app`

Share this URL with anyone!
