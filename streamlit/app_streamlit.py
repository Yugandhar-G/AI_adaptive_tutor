import streamlit as st
import requests
from openai import OpenAI
import json
import os
from datetime import datetime
from typing import Dict
import plotly.express as px
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"  # FastAPI backend

# Page configuration
st.set_page_config(
    page_title="Adaptive Learning System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Card styling */
    .content-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .content-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Difficulty badges */
    .difficulty-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .difficulty-easy {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .difficulty-medium {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .difficulty-hard {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* MCQ option styling */
    .mcq-option {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        background: #f8f9fa;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .mcq-option:hover {
        background: #e9ecef;
        border-color: #667eea;
    }
    
    .mcq-option.correct {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .mcq-option.incorrect {
        background: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    /* Knowledge state progress bars */
    .progress-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 25px;
        background: #e9ecef;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Explanation box */
    .explanation-box {
        background: #f8f9fa;
        border-left: 4px solid #2196F3;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    .explanation-box h4 {
        margin-top: 0;
        color: #2196F3;
    }
    
    .explanation-box p {
        margin: 0.75rem 0;
    }
    
    .explanation-box strong {
        color: #333;
        font-weight: 600;
    }
    
    .explanation-box ul, .explanation-box ol {
        margin: 0.75rem 0;
        padding-left: 1.5rem;
    }
    
    .explanation-box li {
        margin: 0.5rem 0;
    }
    
    /* Success/Error messages */
    .feedback-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .feedback-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Loading spinner */
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 Adaptive Learning System</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Personalized Education Powered by AI</p>
</div>
""", unsafe_allow_html=True)

# ==========================
# OpenAI client (for explanations)
# ==========================
# Make sure OPENAI_API_KEY is set in your environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ==========================
# Helper functions (MUST be defined before use)
# ==========================

def submit_answer_backend(student_id: str, topic: str, concept_id: int, correct: bool):
    """Send correctness to backend to update knowledge state."""
    payload = {
        "student_id": student_id,
        "topic": topic,
        "concept_id": int(concept_id),
        "correct": bool(correct),
    }
    r = requests.post(f"{API_BASE}/submit-answer", json=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    st.session_state["knowledge_state"] = data.get("knowledge_state", {})


def refresh_knowledge_state(student_id: str):
    """Fetch latest knowledge state from backend."""
    r = requests.get(f"{API_BASE}/knowledge-state/{student_id}", timeout=10)
    r.raise_for_status()
    data = r.json()
    st.session_state["knowledge_state"] = data


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_concept_mapping():
    """Get mapping of concept IDs to concept names."""
    try:
        r = requests.get(f"{API_BASE}/concept-mapping", timeout=10)
        r.raise_for_status()
        data = r.json()
        # Validate that we got the expected structure
        if "id_to_concept" in data and "concept_to_id" in data:
            # Convert string keys to integer keys for easier lookup
            id_to_concept = {}
            for key, value in data["id_to_concept"].items():
                try:
                    id_to_concept[int(key)] = value
                except (ValueError, TypeError):
                    id_to_concept[key] = value
            return {
                "id_to_concept": id_to_concept,
                "concept_to_id": data["concept_to_id"]
            }
        else:
            return {"id_to_concept": {}, "concept_to_id": {}}
    except requests.exceptions.RequestException as e:
        # Return empty mapping if backend is not available
        return {"id_to_concept": {}, "concept_to_id": {}}


def create_mastery_charts(concept_mastery: Dict[str, float], student_id: str):
    """
    Create Plotly charts for mastery visualization.
    Returns (pie_chart, bar_chart) or (None, None) if no data.
    """
    if not concept_mastery:
        return None, None
    
    # Categorize concepts by mastery level
    mastered = sum(1 for score in concept_mastery.values() if score >= 0.7)
    learning = sum(1 for score in concept_mastery.values() if 0.4 <= score < 0.7)
    beginner = sum(1 for score in concept_mastery.values() if score < 0.4)
    
    # Create pie chart for mastery distribution
    pie_data = {
        "Category": ["Mastered (≥70%)", "Learning (40-70%)", "Beginner (<40%)"],
        "Count": [mastered, learning, beginner],
        "Color": ["#6bcf7f", "#ffd93d", "#ff6b6b"]
    }
    
    pie_fig = px.pie(
        values=pie_data["Count"],
        names=pie_data["Category"],
        title="Mastery Distribution",
        color_discrete_sequence=pie_data["Color"],
        hole=0.4  # Donut chart
    )
    pie_fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    pie_fig.update_layout(
        font=dict(size=12),
        showlegend=True,
        height=350
    )
    
    # Create bar chart for top concepts
    sorted_concepts = sorted(concept_mastery.items(), key=lambda x: x[1], reverse=True)
    top_n = min(15, len(sorted_concepts))  # Show top 15 concepts
    
    concept_names = [name[:50] + "..." if len(name) > 50 else name for name, _ in sorted_concepts[:top_n]]
    mastery_scores = [score * 100 for _, score in sorted_concepts[:top_n]]  # Convert to percentage
    
    # Color bars based on mastery level
    bar_colors = []
    for score in mastery_scores:
        if score >= 70:
            bar_colors.append("#6bcf7f")  # Green
        elif score >= 40:
            bar_colors.append("#ffd93d")  # Yellow
        else:
            bar_colors.append("#ff6b6b")  # Red
    
    bar_fig = go.Figure(data=[
        go.Bar(
            x=mastery_scores,
            y=concept_names,
            orientation='h',
            marker=dict(color=bar_colors),
            text=[f"{score:.1f}%" for score in mastery_scores],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Mastery: %{x:.1f}%<extra></extra>'
        )
    ])
    
    bar_fig.update_layout(
        title="Top Concepts by Mastery Level",
        xaxis_title="Mastery (%)",
        yaxis_title="Concept",
        height=max(400, top_n * 30),
        xaxis=dict(range=[0, 100]),
        font=dict(size=11),
        margin=dict(l=200, r=50, t=50, b=50)
    )
    
    return pie_fig, bar_fig


@st.cache_data
def fetch_topics():
    """Get topics from the KG-backed backend."""
    try:
        r = requests.get(f"{API_BASE}/topics", timeout=10)
        r.raise_for_status()
        return r.json().get("topics", [])
    except Exception as e:
        st.error(f"Error loading topics: {e}")
        return []


def call_generate_content(student_id: str, topic: str, content_type: str,
                          difficulty: str, n_items: int):
    """Call /generate-content and store results in session_state."""
    payload = {
        "student_id": student_id,
        "topic": topic,
        "content_type": content_type,
        "difficulty": difficulty,
        "n_items": n_items,
    }

    # Increase timeout based on number of items (each MCQ takes ~15-30 seconds with retries)
    # For questions: 90s base for 1 question, +30s per additional question
    # For other content: 60s is usually enough
    if content_type == "question":
        timeout_seconds = 90 + (n_items - 1) * 30  # 90s for 1, 120s for 2, 150s for 3, etc.
    else:
        timeout_seconds = 60
    
    try:
        r = requests.post(f"{API_BASE}/generate-content", json=payload, timeout=timeout_seconds)
        r.raise_for_status()
        data = r.json()

        st.session_state["generated_items"] = data.get("items", [])
        st.session_state["knowledge_state"] = data.get("knowledge_state", {})
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Request timed out after {timeout_seconds} seconds. Generating {n_items} questions can take a while. Please try with fewer questions or try again.")
        raise
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error connecting to backend: {e}")
        st.info("💡 Make sure the backend API is running on http://localhost:8000")
        raise


def explain_mcq(question: str, options: list, correct_index: int, student_index: int, was_correct: bool) -> str:
    """Ask OpenAI to explain the solution and tell the student if they were correct."""
    system_msg = (
        "You are a helpful math tutor explaining solutions clearly and concisely. "
        "Format your explanation using markdown with:\n\n"
        "1. **Clear structure**: Break into logical steps with line breaks between paragraphs\n"
        "2. **Mathematical notation**: Use clear formatting like c² = a² + b² (not [ c^2 = a^2 + b^2 ])\n"
        "3. **Emphasis**: Use **bold** for key concepts and formulas\n"
        "4. **Readability**: Keep paragraphs short (2-3 sentences max)\n"
        "5. **Steps**: Number or bullet your steps clearly\n"
        "6. **Summary**: End with a brief summary if helpful\n\n"
        "Avoid using brackets [ ] for formulas - use bold or clear text instead. "
        "Make it easy to read and understand at a glance."
    )
    user_msg = {
        "question": question,
        "options": options,
        "correct_index": int(correct_index),
        "student_index": int(student_index),
        "student_was_correct": bool(was_correct),
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
        temperature=0.7,
    )

    return resp.choices[0].message.content


def format_explanation(explanation: str) -> str:
    """
    Format explanation text for better readability.
    Converts LaTeX-style notation to markdown and improves structure.
    """
    import re
    
    # Replace LaTeX-style brackets with proper formatting
    # [ formula ] -> **formula** for emphasis
    explanation = re.sub(r'\[([^\]]+)\]', r'**\1**', explanation)
    
    # Replace single letter variables in parentheses with italic
    # (c) -> *c* but keep longer text like (the side) as is
    explanation = re.sub(r'\(([a-zA-Z])\)', r'*\1*', explanation)
    
    # Fix common mathematical notation for better display
    explanation = explanation.replace('^2', '²').replace('^3', '³')
    explanation = explanation.replace('\\sqrt', '√')
    explanation = explanation.replace('\\times', '×')
    explanation = explanation.replace('\\div', '÷')
    
    # Split into paragraphs and format
    paragraphs = explanation.split('\n\n')
    formatted_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If paragraph starts with "Step", "Substitute", etc., make it bold
        if re.match(r'^(Step|Substitute|Calculate|Find|Solve|To find|In this)', para, re.IGNORECASE):
            formatted_paragraphs.append(f"**{para}**")
        # If it's a summary, format it nicely
        elif para.startswith('Summary:'):
            formatted_paragraphs.append(f"\n---\n\n**Summary:**\n\n{para.replace('Summary:', '').strip()}")
        else:
            formatted_paragraphs.append(para)
    
    return '\n\n'.join(formatted_paragraphs)


# ==========================
# Sidebar: Student + Controls
# ==========================

st.sidebar.markdown("### 👤 Student Information")
student_id = st.sidebar.text_input(
    "Student ID", 
    value="student_001",
    help="Enter your unique student identifier"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Learning Configuration")

topics = fetch_topics()
if not topics:
    st.sidebar.warning("⚠️ No topics loaded. Check backend connection.")
    topics = ["exponents"]  # Fallback

topic = st.sidebar.selectbox(
    "Select Topic",
    topics,
    help="Choose a topic from the knowledge graph",
    index=0 if topics else None
)

st.sidebar.markdown("---")

content_type_label_to_value = {
    "📝 Explanation": "explanation",
    "💡 Example": "example",
    "❓ Multiple Choice Questions": "question",
}
content_type_label = st.sidebar.radio(
    "Content Type",
    list(content_type_label_to_value.keys()),
    index=2,  # default to MCQ
    help="Choose the type of content you want to learn"
)
content_type = content_type_label_to_value[content_type_label]

st.sidebar.markdown("---")

difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
difficulty_labels = [f"{difficulty_emoji[d]} {d.capitalize()}" for d in ["easy", "medium", "hard"]]
difficulty_index = st.sidebar.radio(
    "Difficulty Level",
    difficulty_labels,
    index=1,
    help="Select the difficulty level for your learning"
)
difficulty = difficulty_index.split()[-1].lower()

st.sidebar.markdown("---")

# number of items (only used for explanation/example)
n_items = 1
if content_type != "question":
    n_items = st.sidebar.number_input(
        "Number of Items",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="How many explanations/examples to generate"
    )

st.sidebar.markdown("---")

generate_btn = st.sidebar.button(
    "🚀 Generate Content",
    type="primary",
    use_container_width=True
)

# ==========================
# Session state init
# ==========================

if "generated_items" not in st.session_state:
    st.session_state["generated_items"] = []
if "knowledge_state" not in st.session_state:
    st.session_state["knowledge_state"] = {}
if "explanations" not in st.session_state:
    st.session_state["explanations"] = {}
if "answered_questions" not in st.session_state:
    st.session_state["answered_questions"] = {}  # question_index -> selected_index
if "stats" not in st.session_state:
    st.session_state["stats"] = {"total": 0, "correct": 0, "incorrect": 0}

# ==========================
# Layout: Content + Knowledge State
# ==========================

if generate_btn:
    if not student_id:
        st.error("❌ Please enter a Student ID.")
    else:
        # Show progress for MCQ generation
        if content_type == "question":
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"🔄 Generating {n_items} questions... This may take 1-3 minutes. Please wait...")
        
        with st.spinner("🔄 Generating personalized content..."):
            # reset old explanations when generating new content
            st.session_state["explanations"] = {}
            st.session_state["answered_questions"] = {}  # Reset to empty dict, not set
            
            try:
                call_generate_content(student_id, topic, content_type, difficulty, n_items)
                if content_type == "question":
                    progress_bar.progress(1.0)
                    status_text.text("✅ All questions generated!")
                st.success("✅ Content generated successfully!")
            except requests.exceptions.Timeout:
                st.error("⏱️ Generation timed out. The backend is still processing. You can try again in a moment.")
                if content_type == "question":
                    st.info("💡 Tip: Try generating fewer questions (5 instead of 10) for faster results.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                if content_type == "question":
                    st.info("💡 Tip: Generating many questions can take time. Try with fewer questions first.")

# Main layout
content_col, state_col = st.columns([2.5, 1])

# ---------- Content display ----------
with content_col:
    st.markdown("### 📚 Generated Content")
    
    items = st.session_state.get("generated_items", [])
    if not items:
        st.info("👆 Click **Generate Content** in the sidebar to create personalized learning material.")
    else:
        # Show progress for MCQs
        if content_type == "question":
            total_q = len(items)
            answered = len(st.session_state["answered_questions"])
            progress = answered / total_q if total_q > 0 else 0
            st.progress(progress)
            st.caption(f"📊 Progress: {answered}/{total_q} questions answered ({progress*100:.0f}%)")
        
        for i, item in enumerate(items, start=1):
            q_topic = item["topic"]
            q_concept_id = item["concept_id"]
            q_type = item["content_type"]
            q_diff = item["difficulty"]
            q_text = item["text"]
            options = item.get("options", None)
            correct_index = item.get("correct_option_index", None)
            
            # Difficulty badge
            diff_class = f"difficulty-{q_diff}"
            diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[q_diff]
            
            # Determine content type label
            content_labels = {
                "question": "Question",
                "explanation": "Explanation",
                "example": "Example"
            }
            content_label = content_labels.get(q_type, "Content")
            
            # Card container
            st.markdown(f"""
            <div class="content-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #333;">{content_label} {i}</h3>
                    <span class="difficulty-badge {diff_class}">{diff_emoji} {q_diff.capitalize()}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # MCQ case: question + 4 options
            if q_type == "question" and options and correct_index is not None:
                is_answered = i in st.session_state["answered_questions"]
                previous_selection = st.session_state["answered_questions"].get(i, None)
                
                # Format question text for better readability
                formatted_question = format_explanation(q_text)
                st.markdown(formatted_question)
                st.markdown("---")
                
                radio_key = f"q{i}_choice"
                btn_key = f"q{i}_check"
                
                # Show options with better styling
                if not is_answered:
                    selected_index = st.radio(
                        "Select your answer:",
                        options=list(range(len(options))),
                        format_func=lambda idx: f"**{chr(65+idx)}.** {options[idx]}",
                        key=radio_key,
                    )
                    
                    if st.button("✅ Submit Answer", key=btn_key, type="primary", use_container_width=True):
                        is_correct = int(selected_index) == int(correct_index)
                        
                        # Update stats
                        st.session_state["stats"]["total"] += 1
                        if is_correct:
                            st.session_state["stats"]["correct"] += 1
                        else:
                            st.session_state["stats"]["incorrect"] += 1
                        
                        # Mark as answered and store selection
                        st.session_state["answered_questions"][i] = selected_index
                        
                        # 1) Update knowledge state via backend
                        try:
                            submit_answer_backend(student_id, q_topic, q_concept_id, is_correct)
                            # The graph will refresh automatically on rerun
                        except Exception as e:
                            st.error(f"❌ Error submitting answer: {e}")
                            st.rerun()
                        
                        # 2) Get explanation from OpenAI
                        try:
                            with st.spinner("🤔 Generating explanation..."):
                                explanation = explain_mcq(
                                    q_text,
                                    options,
                                    correct_index,
                                    selected_index,
                                    is_correct,
                                )
                                st.session_state["explanations"][i] = explanation
                        except Exception as e:
                            st.error(f"Error generating explanation: {e}")
                            st.session_state["explanations"][i] = (
                                f"The correct answer is option {chr(65+correct_index)}: {options[correct_index]}"
                            )
                        
                        st.rerun()
                else:
                    # Show results for answered questions
                    user_selected = st.session_state["answered_questions"][i]
                    was_correct = user_selected == correct_index
                    
                    # Show feedback banner
                    if was_correct:
                        st.success("✅ **Correct!** Great job!")
                    else:
                        st.error("❌ **Incorrect.** The correct answer is highlighted below.")
                    
                    st.markdown("**Options:**")
                    for idx, option in enumerate(options):
                        option_class = ""
                        label = ""
                        if idx == correct_index:
                            option_class = "correct"
                            label = " ✅ Correct Answer"
                        elif idx == user_selected and idx != correct_index:
                            option_class = "incorrect"
                            label = " ❌ Your Answer"
                        
                        st.markdown(f"""
                        <div class="mcq-option {option_class}">
                            <strong>{chr(65+idx)}.</strong> {option}{label}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show explanation
                    if i in st.session_state["explanations"]:
                        st.markdown("---")
                        st.markdown("### 💡 Explanation")
                        
                        # Format the explanation for better readability
                        explanation_text = st.session_state["explanations"][i]
                        formatted_explanation = format_explanation(explanation_text)
                        
                        # Display in a styled box with proper markdown rendering
                        st.markdown("""
                        <div class="explanation-box">
                        """, unsafe_allow_html=True)
                        st.markdown(formatted_explanation)
                        st.markdown("</div>", unsafe_allow_html=True)
            
            else:
                # Explanation / example or non-MCQ content
                # Format the content for better readability
                formatted_content = format_explanation(q_text)
                
                # Display in a styled box with proper markdown rendering (same as explanations)
                st.markdown("""
                <div class="explanation-box">
                """, unsafe_allow_html=True)
                st.markdown(formatted_content)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

# ---------- Knowledge state display ----------
with state_col:
    st.markdown("### 📊 Learning Dashboard")
    
    # Statistics cards
    stats = st.session_state.get("stats", {"total": 0, "correct": 0, "incorrect": 0})
    if stats["total"] > 0:
        col1, col2 = st.columns(2)
        with col1:
            accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            st.metric("Accuracy", f"{accuracy:.1f}%", delta=f"{stats['correct']}/{stats['total']}")
        with col2:
            st.metric("Total Questions", stats["total"])
    
    st.markdown("---")
    
    # Knowledge State for Current Topic
    st.markdown("### 🎯 Mastery for Current Topic")
    
    if not student_id:
        st.info("👆 Enter a Student ID to see your mastery")
    elif not topic:
        st.info("👆 Select a topic to see your mastery")
    else:
        # Get concept mapping
        concept_mapping = get_concept_mapping()
        id_to_concept = concept_mapping.get("id_to_concept", {})
        concept_to_id = concept_mapping.get("concept_to_id", {})
        
        # Get concept_id for current topic
        current_concept_id = concept_to_id.get(topic)
        
        ks = st.session_state.get("knowledge_state", {})
        mastery = ks.get("mastery", {})

        if not mastery:
            st.info("📝 No mastery data yet. Answer some questions to see your knowledge state!")
        elif current_concept_id is None:
            st.info(f"📝 No mastery data for '{topic}' yet. Answer some questions on this topic!")
        else:
            # Get mastery for current topic only
            cid_str = str(current_concept_id)
            if cid_str in mastery:
                score = float(mastery[cid_str])
                
                # Show mastery level with stats
                st.markdown(f"**Topic:** {topic}")
                
                # Determine mastery level
                if score >= 0.7:
                    level = "🟢 Mastered"
                    level_color = "#6bcf7f"
                elif score >= 0.4:
                    level = "🟡 Learning"
                    level_color = "#ffd93d"
                else:
                    level = "🔴 Beginner"
                    level_color = "#ff6b6b"
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Mastery Level:** <span style='color: {level_color}; font-size: 1.1rem;'>{level}</span>", unsafe_allow_html=True)
                with col2:
                    st.metric("Mastery Score", f"{score:.1%}")
                
                # Progress bar
                st.markdown(f"""
                <div class="progress-container" style="margin-top: 1rem;">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {score*100}%; background: {level_color};">
                            {score:.0%}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Mastery breakdown
                st.markdown("---")
                st.markdown("**Mastery Breakdown:**")
                mastered_threshold = 0.7
                learning_threshold = 0.4
                
                if score >= mastered_threshold:
                    st.success(f"✅ **Mastered** (≥70%): You have a strong understanding of this topic!")
                elif score >= learning_threshold:
                    st.warning(f"📚 **Learning** (40-70%): Keep practicing to master this topic!")
                else:
                    st.error(f"📖 **Beginner** (<40%): Continue learning to improve your mastery!")
            else:
                st.info(f"📝 No mastery data for '{topic}' yet. Answer some questions on this topic!")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Knowledge State", use_container_width=True):
            try:
                with st.spinner("Refreshing..."):
                    refresh_knowledge_state(student_id)
                st.success("✅ Knowledge state updated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error refreshing knowledge state: {e}")
    
    # Additional info
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This adaptive learning system:
    - 📚 Generates personalized content
    - 🎯 Tracks your knowledge state
    - 📈 Adapts difficulty to your level

    """)
