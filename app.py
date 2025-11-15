import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
import io
import json

# Import functions from our learning module
from adaptive_learning import (
    adaptive_learning, 
    fetch_learning_resources,
    groq_client,
    SUBJECTS,
    agent_tools,     
    available_tools,
    train_next_topic_model, 
    get_next_topic,         
    generate_quiz_questions_with_groq, 
    transcribe_audio_with_groq, 
)
from evaluate_model import model, le, subjects, X_test, y_test, features 

st.set_page_config(page_title="Zavya: Personalized Skill Growth", layout="wide")

# --- Robust helper function to build API-compliant history ---
def build_api_history(messages):
    """
    Converts the Streamlit session history (which contains a mix of
    dicts and Groq ChatCompletionMessage objects) into a
    JSON-serializable list of dicts for the Groq API.
    
    This function explicitly filters out null values to prevent API errors.
    """
    api_history = []
    for msg in messages:
        msg_dict = None 

        if isinstance(msg, dict):
            msg_dict = msg
            if msg_dict.get("role") == "assistant" and msg_dict.get("content") is None and "tool_calls" not in msg_dict:
                continue 
            if msg_dict.get("content") is None and "tool_calls" in msg_dict:
                msg_dict.pop("content", None) 
        
        elif hasattr(msg, "role"): 
            msg_dict = {"role": msg.role}
            
            if msg.content:
                msg_dict["content"] = msg.content
                
            if msg.tool_calls:
                msg_dict["tool_calls"] = []
                for tc in msg.tool_calls:
                    msg_dict["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
        
        if msg_dict and msg_dict.get("role") == "assistant":
            if "content" not in msg_dict and "tool_calls" not in msg_dict:
                continue
        
        if msg_dict:
             api_history.append(msg_dict)
             
    return api_history

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Home",
    "Take Our Quiz",
    "View Your Result",
    "Your Personalised Recommendations",
    "The Agent Tutor",
    "Dynamic Topic Quiz", 
    "Our Metrics"
])

# --- Home / About Page ---
if page == "Home":
    st.title("🌟 Zavya: Autonomous Learning Companion")
    st.subheader("AI Agent for Personalized Skill Growth & Adaptive Education using RAG + Multi-Modal Learning")
    st.markdown("""
    **Zavya** is an intelligent platform that evaluates your skills across multiple subjects
    like Python, OOP, Data Structures, DSA, Java, AI/ML, and OS. 
    
    **Features:**
    - **Agent Tutor:** A conversational AI that can chat, search Google, generate RAG plans, and create quizzes.
    - **Dynamic Quiz Page:** A dedicated page to generate 5-question quizzes on any topic you enter.
    - **Adaptive Recommendations:** Get a personalized plan from our main quiz, and get next-topic recommendations from dynamic quizzes.
    - **Multi-Modal Learning:**
        - **Audio Input (Speech-to-Text):** Talk to the Agent Tutor by uploading an audio file.
        - **Text-to-Audio (Output):** Listen to your AI-generated summaries on the Recommendations page.
    """)

# --- Take Quiz Page ---
elif page == "Take Our Quiz":
    st.title("📝 Take the Main Quiz")
    st.markdown("This is the main, static quiz to get your baseline skill level and personalized plan.")
    
    # Static quiz questions
    subjects_dict = {
        "Python": [
            {"q":"Which of these is a Python data type?", "opts":["List","ArrayList","Struct"], "correct":0},
            {"q":"How do you create a function in Python?", "opts":["def my_func():","function my_func() {}","func my_func():"], "correct":0},
            {"q":"Which keyword is used for loops?", "opts":["for","loop","repeat"], "correct":0},
            {"q":"Which operator is used for exponentiation?", "opts":["**","^","pow"], "correct":0},
            {"q":"Which of these is mutable?", "opts":["List","Tuple","String"], "correct":0}
        ],
        "OOP": [
            {"q":"What is a class?", "opts":["A blueprint for objects","An instance of object","A type of function"], "correct":0},
            {"q":"What is encapsulation?", "opts":["Hiding internal details","Combining two classes","Using inheritance"], "correct":0},
            {"q":"Which is an example of polymorphism?", "opts":["Method overloading","Private variable","Object creation"], "correct":0},
            {"q":"What does inheritance allow?", "opts":["Reuse of code","Hiding details","Creating threads"], "correct":0},
            {"q":"What is abstraction?", "opts":["Hiding implementation","Repeating code","Sorting data"], "correct":0}
        ],
        "Data Structures": [
            {"q":"Which data structure uses LIFO?", "opts":["Stack","Queue","Tree"], "correct":0},
            {"q":"Which is fastest for lookup by key?", "opts":["Hash Table","Linked List","Stack"], "correct":0},
            {"q":"Which uses FIFO?", "opts":["Queue","Stack","Graph"], "correct":0},
            {"q":"Which is hierarchical?", "opts":["Tree","Queue","Stack"], "correct":0},
            {"q":"Which is used for adjacency representation of graphs?", "opts":["Adjacency List","Stack","Queue"], "correct":0}
        ],
        "DSA": [
            {"q":"Time complexity of binary search?", "opts":["O(log n)","O(n)","O(n^2)"], "correct":0},
            {"q":"Which algorithm is used for shortest path?", "opts":["Dijkstra's Algorithm","Merge Sort","Prim's Algorithm"], "correct":0},
            {"q":"Which is divide and conquer?", "opts":["Merge Sort","Linear Search","BFS"], "correct":0},
            {"q":"Best case for QuickSort?", "opts":["O(n log n)","O(n^2)","O(log n)"], "correct":0},
            {"q":"Which algorithm is used for minimum spanning tree?", "opts":["Prim's Algorithm","DFS","Binary Search"], "correct":0}
        ],
        "AI/ML": [
            {"q":"Supervised learning requires?", "opts":["Labeled data","Unlabeled data","No data"], "correct":0},
            {"q":"Which is a regression algorithm?", "opts":["Linear Regression","K-Means","Decision Tree Classification"], "correct":0},
            {"q":"Which algorithm is unsupervised?", "opts":["K-Means","Linear Regression","Decision Tree"], "correct":0},
            {"q":"Which is used for classification?", "opts":["Decision Tree","Linear Regression","K-Means"], "correct":0},
            {"q":"Which is used for clustering?", "opts":["K-Means","Linear Regression","Decision Tree"], "correct":0}
        ],
        "OS": [
            {"q":"What is a process?", "opts":["An executing program","A file","A thread"], "correct":0},
            {"q":"What is a deadlock?", "opts":["Two processes waiting indefinitely","CPU idle time","A terminated process"], "correct":0},
            {"q":"Which is a scheduling algorithm?", "opts":["Round Robin","File Allocation","Linked List"], "correct":0},
            {"q":"What is virtual memory?", "opts":["Memory abstraction","Cache","Queue"], "correct":0},
            {"q":"Which is a synchronization mechanism?", "opts":["Semaphore","Stack","Array"], "correct":0}
        ],
        "Java": [
            {"q":"Which is a valid Java data type?", "opts":["int","number","integer"], "correct":0},
            {"q":"How do you define a method?", "opts":["void myFunc() {}","def myFunc():","function myFunc() {}"], "correct":0},
            {"q":"Which keyword is used for inheritance?", "opts":["extends","implements","inherits"], "correct":0},
            {"q":"Which operator is used for comparison?", "opts":["==","=","!="], "correct":0},
            {"q":"What is the default value of int?", "opts":["0","null","undefined"], "correct":0}
        ]
    }
    
    st.header("Answer the following questions:")
    student_scores = {}
    
    with st.form("baseline_quiz_form"):
        for subj, questions in subjects_dict.items():
            st.subheader(subj)
            score = 0
            for idx, q in enumerate(questions):
                ans = st.radio(f"Q{idx+1}: {q['q']}", q["opts"], key=f"{subj}_Q{idx}")
                if q["opts"].index(ans) == q["correct"]:
                    score += 100
            student_scores[subj] = score / len(questions)
        
        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        with st.spinner("Analyzing your results and generating a personalized plan..."):
            plan = adaptive_learning(student_scores)
            st.session_state["plan"] = plan
            st.session_state["student_scores"] = student_scores
            
            try:
                st.write("Training recommendation model...")
                topic_model, le_subject, le_topic, le_next = train_next_topic_model()
                if topic_model:
                    st.session_state["topic_model"] = topic_model
                    st.session_state["le_subject"] = le_subject
                    st.session_state["le_topic"] = le_topic
                    st.session_state["le_next"] = le_next
                    st.write("Recommendation model trained successfully.")
                else:
                    st.warning("Could not train recommendation model (insufficient data).")
            except Exception as e:
                st.error(f"Could not train the next-topic model. Error: {e}")
                st.warning("Topic recommendations may not work.")

        st.success("Quiz submitted! Go to 'Your Personalised Recommendations' to see your learning plan.")

# --- View Results Page ---
elif page == "View Your Result":
    st.title("📊 Your Quiz Results")
    if "student_scores" not in st.session_state:
        st.warning("Take the quiz first in the 'Take Our Quiz' section.")
    else:
        student_scores = st.session_state["student_scores"]
        results_data = [{"Subject": subj, "Score": score} for subj, score in student_scores.items()]
        results_df = pd.DataFrame(results_data)
        st.subheader("Subject-wise Scores:")
        st.dataframe(results_df.style.format({"Score": "{:.2f}"}))

# --- Personalised Recommendations Page ---
elif page == "Your Personalised Recommendations":
    st.title("🎯 Your Personalized Learning Plan")
    st.markdown("This plan was generated by your main quiz. You can ask the **Agent Tutor** to generate new plans anytime!")
    if "plan" not in st.session_state:
        st.warning("Take the quiz first in the 'Take Our Quiz' section.")
    else:
        plan = st.session_state["plan"]
        st.subheader("Predicted Skill Level:")
        st.success(plan.get("Predicted_Skill_Level", "Unknown"))

        weak_subs = plan.get("Weak_Subjects", [])
        st.subheader("Weak Subjects Identified:")
        if weak_subs: st.write(", ".join(weak_subs))
        else: st.success("Great job! No weak subjects identified.")

        rag = plan.get("RAG_Resources", {})
        st.subheader("Your AI-Powered Learning Plan (RAG):")
        
        if not rag: st.markdown("No resources to recommend.")
        
        for subj, rag_package in rag.items():
            st.markdown(f"--- \n ### **{subj}**")
            st.markdown(f"**🤖 AI-Generated Summary:**")
            summary_text = rag_package.get("Summary", "No summary could be generated.")
            st.info(summary_text)
            
            # --- Multi-Modal Output: Text-to-Speech ---
            if st.button(f"🎧 Listen to {subj} Summary", key=f"tts_{subj}"):
                if summary_text and summary_text != "No summary could be generated.":
                    try:
                        tts = gTTS(text=summary_text, lang='en')
                        fp = io.BytesIO()
                        tts.write_to_fp(fp)
                        st.audio(fp, format='audio/mp3')
                    except Exception as e: st.error(f"Could not generate audio: {e}")
                else: st.warning("No summary text to read.")

            st.markdown(f"**📚 Recommended Sources:**")
            resources_list = rag_package.get("Sources", [])
            
            if not resources_list: st.markdown("- No specific resources found.")
            else:
                for r in resources_list:
                    st.markdown(f"- [{r.get('title', 'No Title')}]({r.get('link', '#')})\n  - *{r.get('snippet', 'No snippet available.')}*")

# --- Agent Tutor Page ---
elif page == "The Agent Tutor":
    st.title("🤖 Zavya: The Agent Tutor")
    st.markdown("I am an AI agent. I can answer questions, search Google, generate learning plans, and create quizzes for you.")
    st.info("Try asking: `Test me on Python lists` or `Help me with 'Data Structures'` or `What's new in AI?`")

    if not groq_client:
        st.error("Groq client not initialized. Please check your API key in `adaptive_learning.py`.")
    
    # Define the base system prompt
    system_prompt = f"""
    You are 'Zavya', a friendly and autonomous AI learning agent.
    Your job is to help a student learn. You have access to tools.
    
    *** IMPORTANT API RULE ***
    When you decide to call a tool, your response MUST contain *only* the tool call and its *full* syntax.
    Do NOT add any other text, conversation, or explanation.
    
    CORRECT EXAMPLE:
    <function=google_search>{{"query": "latest AI news"}}</function>
    
    INCORRECT:
    <function=google_search>{{"query": "latest AI news"}}  <-- (Missing closing tag)
    
    INCORRECT:
    Okay, I'll search for that! <function=google_search>{{"query": "latest AI news"}}</function>
    
    After the tool runs, you will be called again to summarize the results.
    *** END OF RULE ***
    
    - The student's known skill level is 'beginner' (you can update this if they specify).
    - Their known weak subjects are 'unknown' (use this if they don't specify a subject).
    - When a student asks to be tested, use `generate_quiz_questions_with_groq`.
    - When a student asks for help, a plan, or resources, use `get_rag_plan_for_subject`.
    - When a student asks for news, definitions, or recent facts, use `Google Search`.
    - For all other conversational questions, answer directly.
    - When you summarize a tool's output, do not just dump raw JSON. Explain what you found.
    - For quizzes, display the questions clearly.
    """

    # Get user's context from session state if it exists
    if "plan" in st.session_state:
        plan = st.session_state["plan"]
        weak_subs = ", ".join(plan.get("Weak_Subjects", ["all topics"]))
        skill_level = plan.get("Predicted_Skill_Level", "beginner")
        # Overwrite system prompt with context
        system_prompt = f"""
        You are 'Zavya', a friendly and autonomous AI learning agent.
        Your job is to help a student learn. You have access to tools.
        
        *** IMPORTANT API RULE ***
        (Same as above)
        *** END OF RULE ***
        
        - You are talking to a '{skill_level}' student.
        - Their known weak subjects are '{weak_subs}'.
        - When a student asks to be tested, use `generate_quiz_questions_with_groq`.
        - When a student asks for help, a plan, or resources, use `get_rag_plan_for_subject`.
        - When a student asks for news, definitions, or recent facts, use `Google Search`.
        - For all other conversational questions, answer directly.
        - When you summarize a tool's output, do not just dump raw JSON. Explain what you found.
        - For quizzes, display the questions clearly.
        """

    # Initialize chat history
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = [{"role": "assistant", "content": "Hello! I'm Zavya. How can I help you learn today?"}]

    # Display chat messages from history
    for message in st.session_state.agent_messages:
        
        role = None
        content = None
        tool_calls = None
        tool_name = None

        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            tool_name = message.get("name")
        
        elif hasattr(message, "role"): 
            role = message.role
            content = message.content
            tool_calls = message.tool_calls
        
        if not role: continue

        with st.chat_message(role):
            if isinstance(content, str):
                st.markdown(content)
            elif tool_calls:
                st.markdown(f"🤖 Calling tool `{tool_calls[0].function.name}`...")
            elif role == "tool":
                st.markdown(f"Tool `{tool_name}` returned data. (Agent is processing...)")


    # --- NEW MULTI-MODAL INPUT LOGIC ---
    user_prompt = None

    # Display text input first
    text_prompt = st.chat_input("Ask me to explain, search, or quiz you...")
    
    # Display audio input below
    audio_file = st.file_uploader(
        "Or, upload an audio message:", 
        type=["mp3", "wav", "m4a", "ogg"], 
        key="audio_uploader"
    )

    if text_prompt:
        user_prompt = text_prompt
    
    elif audio_file:
        with st.spinner("Transcribing audio..."):
            audio_bytes = audio_file.getvalue()
            transcribed_text = transcribe_audio_with_groq(audio_bytes, audio_file.name)
            
            if "Error:" not in transcribed_text:
                st.info(f"Transcribed: *'{transcribed_text}'*")
                user_prompt = transcribed_text # Use transcribed text as the prompt
            else:
                st.error(transcribed_text)
        # Clear the uploader after processing
    
    # --- END NEW LOGIC ---

    # React to user input (if we have one)
    if user_prompt:
        st.session_state.agent_messages.append({"role": "user", "content": user_prompt})
        
        with st.chat_message("user"):
            st.markdown(user_prompt)

        try:
            # === AGENT STEP 1: Send user message + tools to Groq ===
            messages_to_send = [{"role": "system", "content": system_prompt}]
            messages_to_send.extend(build_api_history(st.session_state.agent_messages[-10:]))
            
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages_to_send,
                tools=agent_tools,
                tool_choice="auto",
                temperature=0.0
            )

            response_message = response.choices[0].message
            st.session_state.agent_messages.append(response_message)

            # === AGENT STEP 2: Check if Groq wants to call a tool ===
            if response_message.tool_calls:
                
                # Execute all tool calls
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_tools.get(function_name)
                    
                    if function_to_call:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            tool_output = function_to_call(**function_args)
                            st.session_state.agent_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": json.dumps(tool_output),
                            })
                        except Exception as e:
                            st.error(f"Error calling tool {function_name}: {e}")
                            st.session_state.agent_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": f'{{"error": "Failed to run tool: {str(e)}"}}',
                            })
                
                # === AGENT STEP 3: Send tool output back to Groq for a final answer ===
                final_messages = [{"role": "system", "content": system_prompt}]
                final_messages.extend(build_api_history(st.session_state.agent_messages[-10:]))
                
                final_response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=final_messages,
                    stream=True
                )
                
                with st.chat_message("assistant"):
                    response_content = st.write_stream(final_response)
                
                if response_content:
                    st.session_state.agent_messages.append({"role": "assistant", "content": response_content})
                else:
                    print("Warning: Stream returned no content.")
                    st.session_state.agent_messages.append({"role": "assistant", "content": "I apologize, I encountered an empty response. Please try again."})
                
                st.rerun()

            else:
                if not response_message.content:
                    print("Warning: Groq returned empty conversational response.")
                    st.session_state.agent_messages.pop() 
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": "I apologize, I encountered an empty response. Please try again."
                    })
                
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Dynamic Topic Quiz Page ---
elif page == "Dynamic Topic Quiz":
    st.title("📝 Dynamic Topic Quiz")
    st.markdown("Test your knowledge on a specific topic. I'll generate 5 questions for you.")

    if "topic_model" not in st.session_state:
        st.warning("Please take the main quiz on the 'Take Our Quiz' page first.")
        st.info("This is needed to train the adaptive recommendation model.")
    else:
        topic_model = st.session_state["topic_model"]
        le_subject = st.session_state["le_subject"]
        le_topic = st.session_state["le_topic"]
        le_next = st.session_state["le_next"]

        with st.form("quiz_generator_form"):
            col1, col2 = st.columns(2)
            with col1:
                subject = st.selectbox("Subject", SUBJECTS)
            with col2:
                skill_level = st.selectbox("Your Skill Level", ["beginner", "intermediate", "advanced"])
            
            topic = st.text_input("Enter a specific topic (e.g., 'Lists', 'Recursion', 'Deadlocks')")
            
            generate_button = st.form_submit_button("Generate 5-Question Quiz")

        if generate_button:
            if not topic:
                st.warning("Please enter a topic.")
            else:
                with st.spinner(f"Generating a {skill_level} quiz on {subject}: {topic}..."):
                    try:
                        questions = generate_quiz_questions_with_groq(subject, topic, skill_level)
                        st.session_state.dynamic_quiz = questions
                        st.session_state.dynamic_quiz_context = {
                            "subject": subject,
                            "topic": topic,
                            "skill_level": skill_level
                        }
                    except Exception as e:
                        st.error(f"Failed to generate quiz: {e}")
        
        if "dynamic_quiz" in st.session_state:
            st.subheader(f"Quiz on {st.session_state.dynamic_quiz_context['topic']}")
            questions = st.session_state.dynamic_quiz
            
            with st.form(key="dynamic_quiz_form"):
                answers = {}
                for idx, q in enumerate(questions):
                    if 'q' not in q or 'opts' not in q or 'correct' not in q:
                        st.error(f"Invalid question format received from AI: {q}")
                        continue
                    
                    user_ans = st.radio(f"**Q{idx+1}: {q['q']}**", q["opts"], key=f"dyn_q_{idx}")
                    answers[idx] = {"user_ans": user_ans, "correct_idx": q["correct"], "opts": q["opts"]}

                submit_button = st.form_submit_button("Submit Quiz")
            
            if submit_button:
                correct_count = 0
                for idx, data in answers.items():
                    if data["opts"].index(data["user_ans"]) == data["correct_idx"]:
                        correct_count += 1
                
                score = (correct_count / len(questions)) * 100
                st.info(f"You scored **{score:.0f}%** ({correct_count} out of {len(questions)})")

                context = st.session_state.dynamic_quiz_context

                if score > 60:
                    st.success("Great job! You've passed this topic.")
                    with st.spinner("Finding your next topic..."):
                        next_topic = get_next_topic(
                            subject=context["subject"],
                            current_topic=context["topic"],
                            score=score,
                            topic_model=topic_model,
                            le_subject=le_subject,
                            le_topic=le_topic,
                            le_next=le_next
                        )
                        st.info(f"Based on your performance, a good next topic to study is: **{next_topic}**")
                else:
                    st.warning("It looks like you need a bit more practice on this topic. Here are some resources to help:")
                    with st.spinner(f"Finding resources for {context['subject']}: {context['topic']}..."):
                        resources = fetch_learning_resources(
                            subject_query=f"{context['subject']} {context['topic']}",
                            skill_level=context['skill_level']
                        )
                        for r in resources:
                            st.markdown(f"- [{r.get('title', 'No Title')}]({r.get('link', '#')})\n  - *{r.get('snippet', 'No snippet available.')}*")

                del st.session_state.dynamic_quiz
                del st.session_state.dynamic_quiz_context
            
# --- Evaluate Model Metrics Page ---
elif page == "Our Metrics":
    st.title("📈 Model Evaluation Metrics")

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import pandas as pd

        # --- First Model (Skill Level) ---
        st.header("Skill Level Prediction Model Metrics")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("Accuracy:")
        st.write(f"{acc*100:.2f}%")

        st.subheader("Classification Report:")
        report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.subheader("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        st.subheader("Feature Importance:")
        feat_importance = pd.DataFrame({
            'Feature': features, 
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(feat_importance.set_index('Feature')) 

        # --- Second Model: Next Topic Recommendation ---
        st.markdown("---")
        st.header("Next Topic Recommendation Model Metrics")
        
        try:
            df_next = pd.read_csv('student_adaptive_dataset.csv')
            if df_next.empty:
                st.warning("`student_adaptive_dataset.csv` is empty. Cannot generate Next Topic metrics.")
            else:
                from sklearn.preprocessing import LabelEncoder
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split

                le_subject = LabelEncoder()
                le_topic = LabelEncoder()
                le_next = LabelEncoder()
                df_next['Subject_Label'] = le_subject.fit_transform(df_next['Subject'])
                df_next['Current_Topic_Label'] = le_topic.fit_transform(df_next['Current_Topic'])
                df_next['Next_Recommended_Topic_Label'] = le_next.fit_transform(df_next['Next_Recommended_Topic'])

                X_next = df_next[['Subject_Label', 'Current_Topic_Label', 'Score']]
                y_next = df_next['Next_Recommended_Topic_Label']

                if len(df_next) < 5 or len(set(y_next)) < 2:
                     st.warning("Not enough data or classes in `student_adaptive_dataset.csv` for a full report.")
                else:
                    X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(
                        X_next, y_next, test_size=0.2, random_state=42, stratify=y_next
                    )

                    model_next = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
                    model_next.fit(X_train_next, y_train_next)
                    y_pred_next = model_next.predict(X_test_next)

                    acc_next = accuracy_score(y_test_next, y_pred_next)
                    st.subheader("Accuracy:")
                    st.write(f"{acc_next*100:.2f}%")

                    st.subheader("Classification Report:")
                    report_next = classification_report(y_test_next, y_pred_next, target_names=le_next.classes_, zero_division=0, output_dict=True)
                    report_df_next = pd.DataFrame(report_next).transpose()
                    st.dataframe(report_df_next)

        except FileNotFoundError:
            st.error("`student_adaptive_dataset.csv` not found. Cannot generate Next Topic metrics.")
        except Exception as e:
            st.error(f"An error occurred loading the Next Topic model: {e}")

    except ImportError:
        st.error("Required libraries not found. Please run `pip install -r requirements.txt`")
    except Exception as e:
        st.error(f"An error occurred while generating metrics: {e}")