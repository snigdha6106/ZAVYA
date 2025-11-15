import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from googleapiclient.discovery import build
import groq
import json
import os
import re
# -------------------------
# 1. Dataset Paths & Setup
# -------------------------
DATA_FILE = 'student_quiz_dataset_v2.csv'      
NEXT_TOPIC_FILE = 'student_adaptive_dataset.csv' 
SUBJECTS = ['Python','OOP','Data Structures','DSA','Java','AI/ML','OS']

# --- API Keys ---
# Google Search (RAG)
API_KEY = "YOUR_API_KEY" # Google Search API Key
CSE_ID = "your_cse_id"                       # Google Search CSE ID

# Groq (Agent, Quiz Gen, Audio Transcription)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key") 

# --- Initialize Groq Client ---
try:
    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
    else:
        print("Groq API key is missing or invalid. Client not initialized.")
        groq_client = None
except Exception as e:
    print(f"Error initializing Groq client: {e}.")
    groq_client = None
    
# -------------------------
# 2. Load Datasets
# -------------------------
def load_dataset():
    if not os.path.exists(DATA_FILE):
        print(f"Warning: {DATA_FILE} not found. Returning empty DataFrame.")
        return pd.DataFrame(columns=SUBJECTS + ['Average_Score', 'Skill_Level'])
    return pd.read_csv(DATA_FILE)

def load_topic_dataset():
    if not os.path.exists(NEXT_TOPIC_FILE):
        print(f"Warning: {NEXT_TOPIC_FILE} not found. Next topic model will fail.")
        return pd.DataFrame(columns=['Subject', 'Current_Topic', 'Score', 'Next_Recommended_Topic'])
    return pd.read_csv(NEXT_TOPIC_FILE)


# -------------------------
# 3. Preprocess Skill-Level Dataset
# -------------------------
def preprocess(df):
    le = LabelEncoder()
    if 'Skill_Level_Label' not in df.columns:
        df['Skill_Level_Label'] = le.fit_transform(df['Skill_Level'])
    else:
        le.fit(df['Skill_Level'].astype(str))
        df['Skill_Level_Label'] = le.transform(df['Skill_Level'].astype(str))

    # --- Feature Engineering ---
    df['total_score'] = df[SUBJECTS].sum(axis=1)
    df['std_dev_scores'] = df[SUBJECTS].std(axis=1).fillna(0) 
    df['num_weak'] = (df[SUBJECTS] < 50).sum(axis=1)
    df['num_strong'] = (df[SUBJECTS] > 80).sum(axis=1)
    df['range'] = df[SUBJECTS].max(axis=1) - df[SUBJECTS].min(axis=1)
    
    features = SUBJECTS + ['total_score', 'std_dev_scores', 'num_weak', 'num_strong', 'range']
    
    for col in features:
        if col not in df.columns:
            df[col] = 0
            
    X = df[features]
    y = df['Skill_Level_Label']
    return X, y, le, features # Return features list

# -------------------------
# 4. Train Skill-Level Model
# -------------------------
def train_model(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=None,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_res, y_res)
    return model

# -------------------------
# 5. Weak Subjects
# -------------------------
def weak_subjects(student_scores, threshold=50):
    return [subj for subj, score in student_scores.items() if score < threshold]

# -------------------------
# 6. Google Search & Video Transcript (Multi-Modal RAG)
# -------------------------
def google_search(query: str, num_results: int = 3) -> list:
    """
    Retrieves Google search results. This is a tool for the AI agent.
    """
    print(f"[Agent Tool Call: google_search] Query: {query}")
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx=CSE_ID, num=num_results).execute()
        items = res.get('items', [])
        clean_items = [{'title': item.get('title'), 'link': item.get('link'), 'snippet': item.get('snippet')} for item in items]
        return clean_items
    except Exception as e:
        print(f"Error during Google Search: {e}")
        return [{"error": str(e)}]

def fetch_learning_resources(subject_query: str, skill_level: str, max_results: int = 3) -> list:
    """
    Fetches learning resources (articles, videos).
    """
    query = f"{skill_level} {subject_query} tutorial course"
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx=CSE_ID, num=max_results).execute()
        results = res.get('items', [])
    except Exception as e:
        print(f"Error during Google Search: {e}")
        results = []

    resources = []
    for result in results:
        res = {
            'title': result.get('title'),
            'link': result.get('link'),
            'snippet': result.get('snippet')
        }
        resources.append(res)
        
    if not resources:
        resources.append({'title': f"No {subject_query} resources found", 'link': '', 'snippet': 'Could not find resources.'})
    return resources

# -------------------------
# 7. Generate Learning Plan (RAG with GROQ)
# -------------------------
def generate_summary_with_groq(subject, skill_level, resources):
    """
    The "G" in RAG: Takes search context and generates a new summary using Groq.
    """
    if not groq_client: return "Could not generate AI summary."
    if not resources: return "No search results to summarize."

    context_parts = []
    for res in resources:
        context_parts.append(f"Source Title: {res['title']}")
        if res.get('snippet'):
            context_parts.append(f"Content (Web Snippet): {res['snippet']}\n---")
    context = "\n".join(context_parts)
    
    prompt = f"You are a learning assistant. A '{skill_level}' student is weak in '{subject}'. Based *only* on the following context, write a brief, encouraging summary.\n\nContext:\n{context}\n\nSummary:"
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful learning assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant", temperature=0.7, max_tokens=200,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during Groq generation: {e}")
        return f"Error generating summary for {subject}."

def get_rag_plan_for_subject(subject: str, skill_level: str) -> dict:
    """
    Generates a full RAG learning plan for a given subject.
    This is a tool for the AI agent.
    """
    print(f"[Agent Tool Call: get_rag_plan_for_subject] Subject: {subject}, Skill: {skill_level}")
    resources = fetch_learning_resources(subject, skill_level, max_results=3)
    summary = generate_summary_with_groq(subject, skill_level, resources)
    return {
        "Summary": summary,
        "Sources": resources 
    }

def generate_learning_plan(student_scores, model, le, features_list):
    """
    This function is for the *main quiz* page, not the agent.
    """
    scores_df = pd.DataFrame([student_scores])
    
    scores_df['total_score'] = scores_df[SUBJECTS].sum(axis=1)
    scores_df['std_dev_scores'] = scores_df[SUBJECTS].std(axis=1).fillna(0)
    scores_df['num_weak'] = (scores_df[SUBJECTS] < 50).sum(axis=1)
    scores_df['num_strong'] = (scores_df[SUBJECTS] > 80).sum(axis=1)
    scores_df['range'] = scores_df[SUBJECTS].max(axis=1) - scores_df[SUBJECTS].min(axis=1)
    
    X_pred = scores_df[features_list]
    
    skill_label = model.predict(X_pred)[0]
    skill_level = le.inverse_transform([skill_label])[0]
    
    weak = weak_subjects(student_scores)
    plan = { "Predicted_Skill_Level": skill_level, "Weak_Subjects": weak, "RAG_Resources": {} }
    for subject in weak:
        plan["RAG_Resources"][subject] = get_rag_plan_for_subject(subject, skill_level)
    return plan

# -------------------------
# 8. Append New Quiz Record
# -------------------------
def append_new_quiz(student_scores):
    try:
        df = load_dataset()
    except FileNotFoundError:
        df = pd.DataFrame(columns=SUBJECTS + ['Average_Score', 'Skill_Level'])

    scores_array = np.array(list(student_scores.values()), dtype=float)
    avg_score = scores_array.mean()
    
    if avg_score < 45: skill_level = 'beginner'
    elif avg_score < 75: skill_level = 'intermediate'
    else: skill_level = 'advanced'

    record = {**student_scores, 'Average_Score': avg_score, 'Skill_Level': skill_level}
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    
    df.to_csv(DATA_FILE, index=False)
    return df

# -------------------------
# 9. Next Recommended Topic Model
# -------------------------
def train_next_topic_model():
    df = load_topic_dataset()
    if df.empty or len(df) < 5 or len(df['Next_Recommended_Topic'].unique()) < 2:
        print("Warning: Not enough data in student_adaptive_dataset.csv to train topic model.")
        return None, None, None, None

    le_subject = LabelEncoder()
    le_topic = LabelEncoder()
    le_next = LabelEncoder()

    df['Subject_Label'] = le_subject.fit_transform(df['Subject'])
    df['Current_Topic_Label'] = le_topic.fit_transform(df['Current_Topic'])
    df['Next_Recommended_Topic_Label'] = le_next.fit_transform(df['Next_Recommended_Topic'])

    X = df[['Subject_Label', 'Current_Topic_Label', 'Score']]
    y = df['Next_Recommended_Topic_Label']

    model = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
    model.fit(X, y)

    return model, le_subject, le_topic, le_next

def get_next_topic(subject, current_topic, score, topic_model, le_subject, le_topic, le_next):
    if not all([topic_model, le_subject, le_topic, le_next]):
        return "Could not determine next topic (model not trained)."

    if subject not in le_subject.classes_:
        return "New concepts in this subject (try exploring!)"
    subj_label = le_subject.transform([subject])[0]

    if current_topic not in le_topic.classes_:
        try:
            df = load_topic_dataset()
            next_topics = df[df['Subject'] == subject]['Next_Recommended_Topic'].unique()
            if len(next_topics) > 0:
                return np.random.choice(next_topics)
        except Exception:
            pass 
        return "Advanced topics in this subject."
    topic_label = le_topic.transform([current_topic])[0]

    X_pred = pd.DataFrame([[subj_label, topic_label, score]], columns=['Subject_Label', 'Current_Topic_Label', 'Score'])
    pred_label = topic_model.predict(X_pred)[0]
    next_topic = le_next.inverse_transform([pred_label])[0]
    return next_topic

# -------------------------
# 10. Dynamic Quiz Generation (Tool)
# -------------------------
def generate_quiz_questions_with_groq(subject: str, topic: str, skill_level: str) -> list:
    """
    Generates 5 quiz questions on the fly using RAG.
    This is a tool for the AI agent AND the quiz page.
    """
    print(f"[Tool Call: generate_quiz_questions_with_groq] Subject: {subject}, Topic: {topic}")
    if not groq_client:
        return [{"q": "Error: Groq client not initialized.", "opts": ["OK"], "correct": 0}]

    # 1. Retrieve (R)
    resources = fetch_learning_resources(f"{subject} {topic}", skill_level, max_results=2)
    
    # 2. Augment (A)
    context_parts = []
    for res in resources:
        if res.get('snippet'): context_parts.append(res['snippet'])
    
    if not context_parts:
        context = f"General knowledge about {subject}: {topic}."
        print("Warning: No context found, using general knowledge prompt.")
    else:
        context = "\n".join(context_parts)
    
    # 3. Generate (G)
    json_format = """[{"q": "What is...", "opts": ["A", "B", "C"], "correct": 0}, ...]"""
    
    prompt = f"""
    You are a quiz generator. Based *only* on the provided Context,
    generate 5 multiple-choice questions for a '{skill_level}' student on the topic of '{subject}: {topic}'.
    
    RULES:
    1.  Base all questions *strictly* on the Context. If context is empty, use general knowledge.
    2.  Return *only* a valid JSON list of objects.
    3.  Do not include any other text, pre-amble, or post-amble.
    4.  The 'correct' field must be the 0-based index of the correct option.
    
    JSON Format:
    {json_format}
    
    Context:
    {context}
    
    JSON Output:
    """
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a quiz generation assistant that only outputs valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        
        response_text = chat_completion.choices[0].message.content
        
        try:
            questions = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON from Groq: {response_text}")
            match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if match:
                questions = json.loads(match.group(1))
            else:
                raise ValueError("No valid JSON found in response.")

        if isinstance(questions, dict) and 'questions' in questions:
             questions = questions['questions']
        if not isinstance(questions, list):
             raise ValueError("LLM did not return a list.")

        if not questions or not all('q' in q for q in questions):
            raise ValueError("LLM returned empty or malformed questions.")

        return questions
    
    except Exception as e:
        print(f"Error during Groq quiz generation: {e}")
        return [{"q": f"Error generating quiz: {e}. Please try again.", "opts": ["OK"], "correct": 0}]


# -------------------------
# 11. Main Adaptive Function
# -------------------------
def adaptive_learning(student_scores):
    """
    This is the main function called *once* after the initial quiz.
    """
    df = append_new_quiz(student_scores)
    X, y, le, features_list = preprocess(df) 
    model = train_model(X, y)
    
    plan = generate_learning_plan(student_scores, model, le, features_list) 
    
    return plan

# -------------------------
# 12. AGENT TOOL DEFINITIONS
# -------------------------
agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_rag_plan_for_subject",
            "description": "Get a detailed learning plan (summary and resources) for a student who is weak in a specific subject.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string", "description": "The subject the student is weak in (e.g., 'Python', 'Data Structures', 'OS')." },
                    "skill_level": { "type": "string", "description": "The student's current skill level (e.g., 'beginner', 'intermediate')." },
                },
                "required": ["subject", "skill_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz_questions_with_groq",
            "description": "Generate a 5-question multiple-choice quiz on a specific topic. Use this if the user asks for a test or quiz.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string", "description": "The main subject area (e.g., 'Python', 'DSA')." },
                    "topic": { "type": "string", "description": "The specific topic for the quiz (e.g., 'Lists', 'Recursion', 'Deadlocks')." },
                    "skill_level": { "type": "string", "description": "The student's current skill level (e.g., 'beginner', 'intermediate')." },
                },
                "required": ["subject", "topic", "skill_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search Google for recent information, definitions, or news. Use this for general knowledge or real-time questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query." },
                    "num_results": { "type": "integer", "description": "Number of results to return. Default is 3." }
                },
                "required": ["query"],
            },
        },
    }
]

available_tools = {
    "get_rag_plan_for_subject": get_rag_plan_for_subject,
    "generate_quiz_questions_with_groq": generate_quiz_questions_with_groq,
    "google_search": google_search,
}

# -------------------------
# 13. Transcribe Audio (Tool)
# -------------------------
def transcribe_audio_with_groq(audio_file_bytes, file_name="input.wav"):
    """
    Transcribes audio bytes using Groq's Whisper API.
    """
    if not groq_client:
        return "Error: Groq client not initialized."
    
    print(f"[Tool Call: transcribe_audio_with_groq] Transcribing {file_name}...")
    
    try:
        files = (file_name, audio_file_bytes)
        
        transcription = groq_client.audio.transcriptions.create(
            file=files,
            model="whisper-large-v3",
            response_format="json",  
        )
        
        print(f"Transcription successful: {transcription.text}")
        return transcription.text
    
    except Exception as e:
        print(f"Error during Groq audio transcription: {e}")
        return f"Error: Could not transcribe audio. {e}"
