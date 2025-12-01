"""
Adaptive Learning System - Streamlit Application

This application integrates:
1. Knowledge Graph for topic selection and concept relationships
2. Content Generation using OpenAI GPT-3.5 Turbo
3. Transformer DKT Model for knowledge state evaluation
4. Difficulty adaptation based on student performance

Student Flow:
1. Enter Student ID
2. Choose Topic from Knowledge Graph
3. Select Difficulty Level
4. Learn Content (explanation)
5. View Generated Examples
6. Answer Multiple Choice Questions
7. Evaluate Performance and Show Knowledge State
"""

import streamlit as st
import torch
import numpy as np
import pickle
import json
import os
from datetime import datetime
from openai import OpenAI
import networkx as nx
import random

# Page configuration
st.set_page_config(
    page_title="Adaptive Learning System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION AND CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the Adaptive Learning System"""
    def __init__(self):
        # Paths (these should be updated based on where files are stored)
        self.knowledge_graph_path = 'khan_knowledge_graph.pkl'
        self.embeddings_path = 'khan_embeddings.pkl'
        self.dkt_model_path = 'dkt_output/dkt_model.pt'
        
        # OpenAI API Key (should be set as environment variable in production)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Difficulty levels
        self.difficulty_levels = ["easy", "medium", "hard"]
        
        # Number of MCQs to generate
        self.num_mcqs = 5
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def initialize_system():
    """Initialize the adaptive learning system components"""
    config = Config()
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=config.openai_api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        client = None
    
    # Load Knowledge Graph
    knowledge_graph = None
    if os.path.exists(config.knowledge_graph_path):
        try:
            with open(config.knowledge_graph_path, 'rb') as f:
                knowledge_graph = pickle.load(f)
            st.success(f"✅ Loaded Knowledge Graph with {knowledge_graph.number_of_nodes()} topics")
        except Exception as e:
            st.warning(f"⚠️ Could not load Knowledge Graph: {e}")
    else:
        st.warning(f"⚠️ Knowledge Graph not found at {config.knowledge_graph_path}")
        # Create a simple demo knowledge graph
        knowledge_graph = create_demo_knowledge_graph()
    
    # Load DKT Model
    dkt_model = None
    num_skills = 100  # default
    if os.path.exists(config.dkt_model_path):
        try:
            checkpoint = torch.load(config.dkt_model_path, map_location=config.device)
            num_skills = checkpoint.get('num_skills', 100)
            st.success(f"✅ Loaded DKT Model for {num_skills} skills")
        except Exception as e:
            st.warning(f"⚠️ Could not load DKT Model: {e}")
    else:
        st.warning(f"⚠️ DKT Model not found at {config.dkt_model_path}")
    
    return config, client, knowledge_graph, num_skills


def create_demo_knowledge_graph():
    """Create a demo knowledge graph with common math topics"""
    G = nx.DiGraph()
    
    # Basic topics
    topics = [
        "Counting", "Addition", "Subtraction", "Multiplication", "Division",
        "Fractions", "Decimals", "Percentages", "Integers", "Exponents",
        "Variables", "Linear Equations", "Quadratic Equations", "Functions",
        "Graphing", "Geometry", "Pythagorean Theorem", "Trigonometry",
        "Derivatives", "Integrals"
    ]
    
    # Add nodes with complexity levels
    for i, topic in enumerate(topics):
        complexity = min(10, (i // 3) + 1)
        G.add_node(topic, complexity=complexity, category="math")
    
    # Add some prerequisite edges
    edges = [
        ("Counting", "Addition"),
        ("Addition", "Subtraction"),
        ("Addition", "Multiplication"),
        ("Multiplication", "Division"),
        ("Division", "Fractions"),
        ("Fractions", "Decimals"),
        ("Decimals", "Percentages"),
        ("Subtraction", "Integers"),
        ("Multiplication", "Exponents"),
        ("Integers", "Variables"),
        ("Variables", "Linear Equations"),
        ("Linear Equations", "Quadratic Equations"),
        ("Variables", "Functions"),
        ("Functions", "Graphing"),
        ("Geometry", "Pythagorean Theorem"),
        ("Pythagorean Theorem", "Trigonometry"),
        ("Functions", "Derivatives"),
        ("Derivatives", "Integrals"),
    ]
    
    for source, target in edges:
        G.add_edge(source, target, relation='prerequisite')
    
    return G


# ============================================================================
# CONTENT GENERATION FUNCTIONS
# ============================================================================

def generate_explanation(client, topic, difficulty):
    """Generate educational explanation for a topic"""
    if difficulty == "easy":
        prompt = f"""You are a math teacher explaining {topic} to beginners. Write a clear, simple explanation of {topic} that includes:
1) A basic definition
2) The key formula or concept
3) A simple example showing how to use it

Keep your explanation under 200 words and avoid complex terminology."""
    elif difficulty == "medium":
        prompt = f"""You are a math teacher explaining {topic} to intermediate students. Write a detailed explanation of {topic} that includes:
1) A precise definition
2) The key formulas and concepts
3) Step-by-step examples
4) Common applications

Use appropriate mathematical terminology but explain any complex terms."""
    else:  # hard
        prompt = f"""You are a university professor explaining {topic} to advanced students. Write a comprehensive explanation of {topic} that includes:
1) Formal definitions and theorems
2) Proofs or derivations where applicable
3) Complex examples demonstrating mastery
4) Connections to other mathematical concepts
5) Advanced applications"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating explanation: {e}"


def generate_example(client, topic, difficulty):
    """Generate a practical example for the topic"""
    if difficulty == "easy":
        prompt = f"""Provide a simple, real-world example of {topic} in action. Your response should include:
1) A concrete situation where {topic} is used
2) Step-by-step explanation of how {topic} applies
3) The calculations involved, written in a way beginners can understand

Make it relatable and easy to follow."""
    elif difficulty == "medium":
        prompt = f"""Provide a practical application of {topic} in a real-world context. Your response should include:
1) A specific scenario where {topic} is applied
2) Detailed explanation of how {topic} is used
3) The mathematical process involved
4) Why this application is important or useful"""
    else:  # hard
        prompt = f"""Provide an advanced, real-world application of {topic}. Your response should include:
1) A complex scenario from science, engineering, or advanced mathematics
2) Detailed mathematical analysis using {topic}
3) The reasoning and problem-solving approach
4) Why this application is significant"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating example: {e}"


def generate_mcq(client, topic, difficulty, question_num):
    """Generate a multiple choice question"""
    if difficulty == "easy":
        prompt = f"""Create a beginner-level multiple choice question about {topic}.

Format your response EXACTLY as follows:
Question: [Write the question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A, B, C, or D]
Explanation: [Brief explanation of why this is correct]

The question should test basic understanding of {topic}."""
    elif difficulty == "medium":
        prompt = f"""Create an intermediate-level multiple choice question about {topic}.

Format your response EXACTLY as follows:
Question: [Write the question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A, B, C, or D]
Explanation: [Brief explanation of why this is correct]

The question should require applying {topic} concepts."""
    else:  # hard
        prompt = f"""Create an advanced multiple choice question about {topic}.

Format your response EXACTLY as follows:
Question: [Write the question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A, B, C, or D]
Explanation: [Brief explanation of why this is correct]

The question should require sophisticated understanding and problem-solving."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.8
        )
        return parse_mcq(response.choices[0].message.content.strip())
    except Exception as e:
        return None


def parse_mcq(text):
    """Parse the MCQ text into structured format"""
    lines = text.strip().split('\n')
    mcq = {
        'question': '',
        'options': {},
        'correct_answer': '',
        'explanation': ''
    }
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('Question:'):
            mcq['question'] = line.replace('Question:', '').strip()
        elif line.startswith('A)'):
            mcq['options']['A'] = line.replace('A)', '').strip()
        elif line.startswith('B)'):
            mcq['options']['B'] = line.replace('B)', '').strip()
        elif line.startswith('C)'):
            mcq['options']['C'] = line.replace('C)', '').strip()
        elif line.startswith('D)'):
            mcq['options']['D'] = line.replace('D)', '').strip()
        elif line.startswith('Correct Answer:'):
            answer = line.replace('Correct Answer:', '').strip()
            # Extract just the letter
            mcq['correct_answer'] = answer[0] if answer else 'A'
        elif line.startswith('Explanation:'):
            mcq['explanation'] = line.replace('Explanation:', '').strip()
    
    return mcq if mcq['question'] and mcq['options'] else None


# ============================================================================
# KNOWLEDGE STATE EVALUATION
# ============================================================================

def calculate_knowledge_state(student_responses, topic, difficulty):
    """Calculate student's knowledge state based on their responses"""
    total_questions = len(student_responses)
    correct_answers = sum(1 for r in student_responses if r['is_correct'])
    
    # Calculate accuracy
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Calculate difficulty-weighted score
    difficulty_weights = {'easy': 1.0, 'medium': 1.5, 'hard': 2.0}
    weighted_score = accuracy * difficulty_weights[difficulty]
    
    # Estimate mastery level (0-1 scale)
    mastery_level = min(1.0, weighted_score / 2.0)
    
    # Determine performance category
    if accuracy >= 0.8:
        performance = "Excellent"
        recommendation = f"Great job! You have mastered {topic} at {difficulty} level. Consider trying harder difficulty or moving to advanced topics."
    elif accuracy >= 0.6:
        performance = "Good"
        recommendation = f"Good understanding of {topic}. Review the questions you missed and try a few more practice problems."
    elif accuracy >= 0.4:
        performance = "Fair"
        recommendation = f"You have basic understanding but need more practice with {topic}. Review the explanations and try easier questions first."
    else:
        performance = "Needs Improvement"
        recommendation = f"You may need to review the fundamentals of {topic}. Consider starting with easier difficulty and reviewing prerequisite concepts."
    
    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'mastery_level': mastery_level,
        'performance': performance,
        'recommendation': recommendation,
        'weighted_score': weighted_score
    }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Initialize system
    config, client, knowledge_graph, num_skills = initialize_system()
    
    # Header
    st.markdown('<h1 class="main-header">📚 Adaptive Learning System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'student_id' not in st.session_state:
        st.session_state.student_id = ''
    if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = None
    if 'difficulty' not in st.session_state:
        st.session_state.difficulty = 'medium'
    if 'explanation' not in st.session_state:
        st.session_state.explanation = ''
    if 'example' not in st.session_state:
        st.session_state.example = ''
    if 'mcqs' not in st.session_state:
        st.session_state.mcqs = []
    if 'student_responses' not in st.session_state:
        st.session_state.student_responses = []
    if 'current_mcq_index' not in st.session_state:
        st.session_state.current_mcq_index = 0
    if 'knowledge_state' not in st.session_state:
        st.session_state.knowledge_state = None
    
    # Sidebar - Progress tracking
    with st.sidebar:
        st.markdown("### 📊 Learning Progress")
        
        steps = [
            "1️⃣ Student ID",
            "2️⃣ Select Topic",
            "3️⃣ Choose Difficulty",
            "4️⃣ Learn Content",
            "5️⃣ View Example",
            "6️⃣ Answer Questions",
            "7️⃣ View Results"
        ]
        
        for i, step in enumerate(steps, 1):
            if i < st.session_state.step:
                st.markdown(f"✅ {step}")
            elif i == st.session_state.step:
                st.markdown(f"🔵 **{step}**")
            else:
                st.markdown(f"⚪ {step}")
        
        st.markdown("---")
        
        if st.session_state.student_id:
            st.markdown(f"**Student ID:** {st.session_state.student_id}")
        if st.session_state.selected_topic:
            st.markdown(f"**Topic:** {st.session_state.selected_topic}")
        if st.session_state.difficulty:
            st.markdown(f"**Difficulty:** {st.session_state.difficulty.title()}")
        
        st.markdown("---")
        
        if st.button("🔄 Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    
    # STEP 1: Enter Student ID
    if st.session_state.step == 1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 👤 Step 1: Enter Your Student ID")
        st.markdown("Please enter your unique student identifier to begin your learning session.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            student_id = st.text_input(
                "Student ID",
                value=st.session_state.student_id,
                placeholder="e.g., STU12345",
                key="student_id_input"
            )
            
            if st.button("Continue ➡️", key="step1_continue"):
                if student_id.strip():
                    st.session_state.student_id = student_id.strip()
                    st.session_state.step = 2
                    st.rerun()
                else:
                    st.error("Please enter a valid Student ID")
    
    # STEP 2: Select Topic
    elif st.session_state.step == 2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📖 Step 2: Choose Your Topic")
        st.markdown("Select a topic you want to learn about from our knowledge graph.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if knowledge_graph and knowledge_graph.number_of_nodes() > 0:
            # Get topics sorted by complexity
            topics = sorted(
                knowledge_graph.nodes(),
                key=lambda x: knowledge_graph.nodes[x].get('complexity', 5)
            )
            
            # Group topics by complexity for better UX
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_topic = st.selectbox(
                    "Available Topics",
                    options=topics,
                    index=topics.index(st.session_state.selected_topic) if st.session_state.selected_topic in topics else 0,
                    key="topic_select"
                )
                
                # Show topic information
                if selected_topic:
                    complexity = knowledge_graph.nodes[selected_topic].get('complexity', 5)
                    st.markdown(f"**Complexity Level:** {complexity}/10")
                    
                    # Show prerequisites
                    prerequisites = list(knowledge_graph.predecessors(selected_topic))
                    if prerequisites:
                        st.markdown(f"**Prerequisites:** {', '.join(prerequisites[:3])}")
            
            with col2:
                st.markdown("#### 💡 Tip")
                st.info("Start with topics that match your current knowledge level. Prerequisites are shown for each topic.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Back", key="step2_back"):
                    st.session_state.step = 1
                    st.rerun()
            with col2:
                if st.button("Continue ➡️", key="step2_continue"):
                    st.session_state.selected_topic = selected_topic
                    st.session_state.step = 3
                    st.rerun()
        else:
            st.error("Knowledge graph not available. Please ensure the graph file exists.")
    
    # STEP 3: Choose Difficulty
    elif st.session_state.step == 3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Step 3: Select Difficulty Level")
        st.markdown(f"Choose the difficulty level for learning **{st.session_state.selected_topic}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Easy\n\nBeginner-friendly\nexplanations", key="diff_easy", use_container_width=True):
                st.session_state.difficulty = 'easy'
                st.session_state.step = 4
                st.rerun()
        
        with col2:
            if st.button("🟡 Medium\n\nIntermediate\nconcepts", key="diff_medium", use_container_width=True):
                st.session_state.difficulty = 'medium'
                st.session_state.step = 4
                st.rerun()
        
        with col3:
            if st.button("🔴 Hard\n\nAdvanced\nproblems", key="diff_hard", use_container_width=True):
                st.session_state.difficulty = 'hard'
                st.session_state.step = 4
                st.rerun()
        
        if st.button("⬅️ Back", key="step3_back"):
            st.session_state.step = 2
            st.rerun()
    
    # STEP 4: Learn Content (Explanation)
    elif st.session_state.step == 4:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"### 📚 Step 4: Learn About {st.session_state.selected_topic}")
        st.markdown(f"**Difficulty Level:** {st.session_state.difficulty.title()}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate explanation if not already generated
        if not st.session_state.explanation:
            with st.spinner("Generating personalized explanation..."):
                st.session_state.explanation = generate_explanation(
                    client,
                    st.session_state.selected_topic,
                    st.session_state.difficulty
                )
        
        # Display explanation
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.explanation)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back", key="step4_back"):
                st.session_state.step = 3
                st.session_state.explanation = ''
                st.rerun()
        with col2:
            if st.button("Continue to Example ➡️", key="step4_continue"):
                st.session_state.step = 5
                st.rerun()
    
    # STEP 5: View Example
    elif st.session_state.step == 5:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"### 💡 Step 5: Practical Example - {st.session_state.selected_topic}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate example if not already generated
        if not st.session_state.example:
            with st.spinner("Generating practical example..."):
                st.session_state.example = generate_example(
                    client,
                    st.session_state.selected_topic,
                    st.session_state.difficulty
                )
        
        # Display example
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.example)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Explanation", key="step5_back"):
                st.session_state.step = 4
                st.rerun()
        with col2:
            if st.button("Continue to Questions ➡️", key="step5_continue"):
                st.session_state.step = 6
                st.rerun()
    
    # STEP 6: Answer Questions
    elif st.session_state.step == 6:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### ✍️ Step 6: Answer Multiple Choice Questions")
        st.markdown(f"Test your understanding of **{st.session_state.selected_topic}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate MCQs if not already generated
        if not st.session_state.mcqs:
            with st.spinner(f"Generating {config.num_mcqs} questions..."):
                mcqs = []
                for i in range(config.num_mcqs):
                    mcq = generate_mcq(
                        client,
                        st.session_state.selected_topic,
                        st.session_state.difficulty,
                        i + 1
                    )
                    if mcq:
                        mcqs.append(mcq)
                st.session_state.mcqs = mcqs
                st.session_state.student_responses = []
                st.session_state.current_mcq_index = 0
        
        # Display current question
        if st.session_state.current_mcq_index < len(st.session_state.mcqs):
            mcq = st.session_state.mcqs[st.session_state.current_mcq_index]
            
            st.markdown(f"#### Question {st.session_state.current_mcq_index + 1} of {len(st.session_state.mcqs)}")
            st.markdown(f"**{mcq['question']}**")
            
            # Show options
            selected_answer = st.radio(
                "Choose your answer:",
                options=list(mcq['options'].keys()),
                format_func=lambda x: f"{x}) {mcq['options'][x]}",
                key=f"mcq_{st.session_state.current_mcq_index}"
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("Submit Answer ✅", key=f"submit_{st.session_state.current_mcq_index}", use_container_width=True):
                    is_correct = (selected_answer == mcq['correct_answer'])
                    
                    # Store response
                    st.session_state.student_responses.append({
                        'question_num': st.session_state.current_mcq_index + 1,
                        'question': mcq['question'],
                        'student_answer': selected_answer,
                        'correct_answer': mcq['correct_answer'],
                        'is_correct': is_correct,
                        'explanation': mcq['explanation']
                    })
                    
                    # Show feedback
                    if is_correct:
                        st.success("✅ Correct! " + mcq['explanation'])
                    else:
                        st.error(f"❌ Incorrect. The correct answer is {mcq['correct_answer']}. " + mcq['explanation'])
                    
                    # Move to next question or finish
                    if st.session_state.current_mcq_index < len(st.session_state.mcqs) - 1:
                        if st.button("Next Question ➡️", key="next_question"):
                            st.session_state.current_mcq_index += 1
                            st.rerun()
                    else:
                        if st.button("View Results ➡️", key="view_results"):
                            st.session_state.step = 7
                            st.rerun()
        
        # Show progress
        progress = (st.session_state.current_mcq_index + 1) / len(st.session_state.mcqs) if st.session_state.mcqs else 0
        st.progress(progress)
        
        if st.button("⬅️ Back to Example", key="step6_back"):
            st.session_state.step = 5
            st.rerun()
    
    # STEP 7: View Results and Knowledge State
    elif st.session_state.step == 7:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### 🎓 Step 7: Your Learning Results")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate knowledge state if not already calculated
        if not st.session_state.knowledge_state:
            st.session_state.knowledge_state = calculate_knowledge_state(
                st.session_state.student_responses,
                st.session_state.selected_topic,
                st.session_state.difficulty
            )
        
        ks = st.session_state.knowledge_state
        
        # Display overall performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Score",
                value=f"{ks['correct_answers']}/{ks['total_questions']}",
                delta=f"{ks['accuracy']*100:.1f}%"
            )
        
        with col2:
            st.metric(
                label="Performance",
                value=ks['performance']
            )
        
        with col3:
            st.metric(
                label="Mastery Level",
                value=f"{ks['mastery_level']*100:.1f}%"
            )
        
        # Show knowledge state visualization
        st.markdown("#### 📊 Knowledge State")
        st.progress(ks['mastery_level'])
        
        # Display detailed results
        st.markdown("#### 📝 Detailed Results")
        
        for i, response in enumerate(st.session_state.student_responses, 1):
            with st.expander(f"Question {i}: {'✅ Correct' if response['is_correct'] else '❌ Incorrect'}"):
                st.markdown(f"**Question:** {response['question']}")
                st.markdown(f"**Your Answer:** {response['student_answer']}")
                st.markdown(f"**Correct Answer:** {response['correct_answer']}")
                st.markdown(f"**Explanation:** {response['explanation']}")
        
        # Show recommendation
        st.markdown("#### 💡 Recommendation")
        st.info(ks['recommendation'])
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Try Another Topic", key="new_topic"):
                # Reset for new topic but keep student ID
                student_id = st.session_state.student_id
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.student_id = student_id
                st.session_state.step = 2
                st.rerun()
        
        with col2:
            if st.button("🎯 Try Different Difficulty", key="new_difficulty"):
                # Keep topic but reset difficulty
                student_id = st.session_state.student_id
                topic = st.session_state.selected_topic
                for key in list(st.session_state.keys()):
                    if key not in ['student_id', 'selected_topic']:
                        del st.session_state[key]
                st.session_state.student_id = student_id
                st.session_state.selected_topic = topic
                st.session_state.step = 3
                st.rerun()
        
        with col3:
            if st.button("🏁 Complete Session", key="complete"):
                # Save session data (could be saved to database in production)
                session_data = {
                    'student_id': st.session_state.student_id,
                    'topic': st.session_state.selected_topic,
                    'difficulty': st.session_state.difficulty,
                    'timestamp': datetime.now().isoformat(),
                    'knowledge_state': st.session_state.knowledge_state,
                    'responses': st.session_state.student_responses
                }
                
                # In production, save to database
                # For demo, just show success message
                st.balloons()
                st.success("✅ Session completed! Your progress has been saved.")
                
                # Reset everything
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                if st.button("Start New Session"):
                    st.rerun()


if __name__ == "__main__":
    main()