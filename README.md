# Zavya: Autonomous Learning Companion 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![Groq](https://img.shields.io/badge/API-Groq-green.svg)
![Google AI](https://img.shields.io/badge/API-Google_Search-yellow.svg)

Zavya is an AI-powered, multi-modal learning companion designed to solve the "one-size-fits-all" problem in online education. It provides a deeply personalized learning experience by accurately assessing a user's skills and generating adaptive learning plans on the fly.


---

## ⭐ Core Features

- **Personalized Skill Assessment**  
  Uses a `RandomForestClassifier` (97.36% accuracy) with advanced feature engineering to predict a user's skill level from the main quiz.

- **Adaptive Learning Plans (RAG)**  
  Fetches real-time learning resources via **Google Search API**, then generates personalized summaries using **Groq Llama 3.1**.

- **AI Agent Tutor**  
  A smart conversational tutor that uses structured function calling to fetch resources, generate quizzes, or create a full learning plan.

- **Multi-Modal Interaction**  
  - Upload voice → transcribed using **Groq Whisper-v3**  
  - Listen to summaries → generated using **Google gTTS**

- **AI-Generated Quizzes**  
  Create quizzes dynamically on any topic and difficulty.

- **Adaptive Next-Topic Prediction**  
  A second ML model suggests the next topic based on performance.

- **Model Metrics Dashboard**  
  View confusion matrix, accuracy, and feature importance directly inside Streamlit.

---

## 🏛️ Architecture & Workflow

1. User takes **Main Quiz**  
2. `adaptive_learning.py` **re-trains RandomForest** on-the-fly  
3. Weak areas → **sent to RAG pipeline**  
4. Groq Llama 3.1 generates personalized learning plan  
5. User interacts with **Agent Tutor**  
6. User takes **Dynamic Quiz**  
7. Next-Topic model suggests what to learn next  

---

## 🛠️ Technologies Used

- **Frontend:** Streamlit  
- **ML:** scikit-learn, Pandas, NumPy, SMOTE  
- **AI APIs:**  
  - Groq (Llama 3.1, Whisper v3)  
  - Google Programmable Search  
- **TTS:** gTTS  
- **Visualization:** Matplotlib, Seaborn  

---

## ⚙️ Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/zavya-ai-companion.git
cd zavya-ai-companion
```

---

### 2. Set Up API Keys

Create a `.env` file:

```
GROQ_API_KEY="your_groq_key"
GOOGLE_API_KEY="your_google_key"
GOOGLE_CSE_ID="your_cse_id"
```

Modify your Python files to load variables:

```python
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

Add `.env` to your `.gitignore`:

```
.env
```

---

### 3. Install Requirements

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
gtts
groq
google-api-python-client
imblearn
python-dotenv
```

---

### 4. Run App

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
.
├── app.py
├── adaptive_learning.py
├── evaluate_model.py
├── run_comparison.py
├── student_quiz_dataset_v2.csv
├── student_adaptive_dataset.csv
├── requirements.txt
├── README.md
├── baseline_confusion_matrix.png
├── advanced_confusion_matrix.png
├── advanced_feature_importance.png
└── demo_screenshot.png
```

---

## 🔮 Future Improvements

- Add image/video understanding  
- Create user-based long-term memory  
- Move ML logic into a stable MLOps pipeline  

---

## 📜 License

This project is licensed under the MIT License.

