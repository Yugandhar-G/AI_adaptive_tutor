# backend_api.py
import os
import sys

# Get the parent directory of this file (the capstone folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# backend_api.py

# backend_api.py

from typing import Literal, Optional, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
import random
import networkx as nx
import json

from adaptive_content_generator import AdaptiveContentGenerator
from mcq_generator import MCQGenerator
from openai import OpenAI  # make sure openai is installed: pip install openai

# =======================================
# Paths & Knowledge Graph loading
# =======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_PATH = os.path.join(BASE_DIR, "khan_knowledge_graph_nlp.pkl")

print(f"[INFO] Loading knowledge graph from: {KG_PATH}")
with open(KG_PATH, "rb") as f:
    KG: nx.DiGraph = pickle.load(f)

print(f"[INFO] KG loaded: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges")

concept_nodes: List[str] = [
    n for n, data in KG.nodes(data=True)
    if isinstance(data, dict) and data.get("type") == "concept"
]
concept_nodes_sorted = sorted(concept_nodes)

ID_TO_CONCEPT: Dict[int, str] = {
    i + 1: name for i, name in enumerate(concept_nodes_sorted)
}
CONCEPT_TO_ID: Dict[str, int] = {name: cid for cid, name in ID_TO_CONCEPT.items()}


def get_node_data(concept_name: str) -> Dict:
    return KG.nodes[concept_name] if concept_name in KG.nodes else {}


def infer_difficulty_from_complexity(concept_name: str) -> str:
    data = get_node_data(concept_name)
    c = data.get("complexity", 3)
    try:
        c = int(c)
    except Exception:
        c = 3

    if c <= 2:
        return "easy"
    elif c <= 4:
        return "medium"
    else:
        return "hard"


def choose_random_concept() -> str:
    return random.choice(concept_nodes_sorted)


# =======================================
# FastAPI app & models
# =======================================

app = FastAPI(title="Adaptive Learning")

DKT_MODEL_PATH = None
EMBEDDINGS_PATH = None

adaptive_gen = AdaptiveContentGenerator(
    dkt_model_path=DKT_MODEL_PATH,
    embeddings_path=EMBEDDINGS_PATH,
)

# Initialize MCQ generator
mcq_gen = MCQGenerator(seed=42)

print("[INFO] AdaptiveContentGenerator initialized in backend_api.")
print("[INFO] MCQGenerator initialized in backend_api.")

# OpenAI client (set OPENAI_API_KEY in your environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


# =======================================
# Student session
# =======================================

class StudentSession:
    def __init__(self, student_id: str):
        self.student_id = student_id
        self.history: List[Dict] = []
        self.mastery: Dict[int, float] = {}

    def update_with_answer(self, concept_id: int, correct: bool):
        self.history.append({"concept_id": concept_id, "correct": bool(correct)})
        old = self.mastery.get(concept_id, 0.0)  # Default to 0.0 instead of 0.5
        alpha = 0.3
        target = 1.0 if correct else 0.0
        new = (1 - alpha) * old + alpha * target
        self.mastery[concept_id] = new

    def knowledge_state_dict(self) -> Dict[str, float]:
        return {str(cid): float(m) for cid, m in self.mastery.items()}


student_sessions: Dict[str, StudentSession] = {}


def get_or_create_session(student_id: str) -> StudentSession:
    if student_id not in student_sessions:
        student_sessions[student_id] = StudentSession(student_id)
    return student_sessions[student_id]


# =======================================
# Pydantic models
# =======================================

Difficulty = Literal["easy", "medium", "hard"]
ContentType = Literal["explanation", "example", "question"]

class ContentItem(BaseModel):
    topic: str
    concept_id: int
    content_type: ContentType
    difficulty: Difficulty
    text: str
    # MCQ-specific fields (None for explanation/example)
    options: Optional[List[str]] = None
    correct_option_index: Optional[int] = None


class KnowledgeState(BaseModel):
    student_id: str
    mastery: Dict[str, float]


class GenerateContentRequest(BaseModel):
    student_id: str
    topic: str
    content_type: ContentType
    difficulty: Optional[Difficulty] = None
    n_items: int = 1


class GenerateContentResponse(BaseModel):
    items: List[ContentItem]
    knowledge_state: KnowledgeState


class KnowledgeStateResponse(KnowledgeState):
    pass


class SubmitAnswerRequest(BaseModel):
    student_id: str
    topic: str
    concept_id: int
    correct: bool


class SubmitAnswerResponse(BaseModel):
    success: bool
    knowledge_state: KnowledgeState


class SimulateSessionRequest(BaseModel):
    student_id: str
    topic: Optional[str] = None
    n_steps: int = 10


class SimulatedInteraction(BaseModel):
    step: int
    topic: str
    concept_id: int
    content_type: ContentType
    difficulty: Difficulty
    content: str
    correct: bool
    mastery_after: Dict[str, float]


class SimulateSessionResponse(BaseModel):
    student_id: str
    interactions: List[SimulatedInteraction]
    final_knowledge_state: KnowledgeState


# =======================================
# MCQ generation helper - Now uses MCQGenerator
# =======================================

def generate_mcq_for_api(topic: str, difficulty: Difficulty, concept_id: int, question_num: int = 0) -> Dict:
    """
    Generate a properly formatted MCQ using the MCQGenerator module.
    Returns a dictionary with all MCQ data including randomized options.
    
    Args:
        topic: The topic for the question
        difficulty: Difficulty level
        concept_id: Concept identifier
        question_num: Question number (used to vary question types)
    """
    mcq_data = mcq_gen.generate_mcq(
        topic=topic,
        difficulty=difficulty,
        concept_id=concept_id,
        question_num=question_num,
    )
    return mcq_data


# =======================================
# API endpoints
# =======================================

@app.get("/topics")
def list_topics():
    return {
        "topics": concept_nodes_sorted,
        "concept_count": len(concept_nodes_sorted),
    }


@app.post("/generate-content", response_model=GenerateContentResponse)
def generate_content(req: GenerateContentRequest):
    session = get_or_create_session(req.student_id)

    topic = req.topic
    if topic not in CONCEPT_TO_ID:
        topic = choose_random_concept()

    concept_id_int = CONCEPT_TO_ID[topic]

    if req.difficulty is not None:
        difficulty = req.difficulty
    else:
        difficulty = infer_difficulty_from_complexity(topic)
    difficulty = difficulty  # type: ignore

    items: List[ContentItem] = []

    n_items = req.n_items
    if req.content_type == "question" and n_items <= 1:
        n_items = 10

    for i in range(n_items):
        if req.content_type == "question":
            # Generate proper MCQ with randomized options
            # Pass question number to ensure variety
            mcq_data = generate_mcq_for_api(topic, difficulty, concept_id_int, question_num=i)
            items.append(
                ContentItem(
                    topic=topic,
                    concept_id=concept_id_int,
                    content_type="question",
                    difficulty=difficulty,  # type: ignore
                    text=mcq_data["question"],
                    options=mcq_data["options"],
                    correct_option_index=mcq_data["correct_index"],
                )
            )
        else:
            # Use your existing adaptive generator for explanation/example
            gen_content, gen_topic, gen_ctype, gen_diff, gen_concept_id = \
                adaptive_gen.generate_adaptive_content(
                    student_id=req.student_id,
                    topic=topic,
                    content_type=req.content_type,
                )

            final_topic = topic
            final_concept_id = concept_id_int
            final_type: ContentType = req.content_type
            final_diff: Difficulty = difficulty  # type: ignore

            items.append(
                ContentItem(
                    topic=final_topic,
                    concept_id=final_concept_id,
                    content_type=final_type,
                    difficulty=final_diff,
                    text=gen_content,
                    options=None,
                    correct_option_index=None,
                )
            )

    ks = KnowledgeState(
        student_id=req.student_id,
        mastery=session.knowledge_state_dict(),
    )

    return GenerateContentResponse(items=items, knowledge_state=ks)


@app.get("/knowledge-state/{student_id}", response_model=KnowledgeStateResponse)
def get_knowledge_state(student_id: str):
    session = get_or_create_session(student_id)
    return KnowledgeStateResponse(
        student_id=student_id,
        mastery=session.knowledge_state_dict(),
    )


@app.get("/concept-mapping")
def get_concept_mapping():
    """Get mapping of concept IDs to concept names."""
    return {
        "id_to_concept": ID_TO_CONCEPT,
        "concept_to_id": CONCEPT_TO_ID,
        "total_concepts": len(ID_TO_CONCEPT)
    }


@app.get("/knowledge-graph-data/{student_id}")
def get_knowledge_graph_data(student_id: str):
    """Get knowledge graph data with student mastery for visualization."""
    session = get_or_create_session(student_id)
    mastery = session.mastery
    
    # Convert concept IDs to names with mastery scores
    concept_mastery = {}
    for concept_id, score in mastery.items():
        concept_name = ID_TO_CONCEPT.get(int(concept_id), f"Concept_{concept_id}")
        concept_mastery[concept_name] = float(score)
    
    # Get graph structure (nodes and edges)
    nodes = []
    edges = []
    
    # Determine which nodes to include
    if concept_mastery:
        # Include concepts with mastery and their neighbors
        concepts_to_include = set(concept_mastery.keys())
        # Add neighbors (prerequisites and dependents)
        for concept_name in concept_mastery.keys():
            if concept_name in KG:
                # Add prerequisites (concepts that this depends on)
                concepts_to_include.update(KG.predecessors(concept_name))
                # Add dependents (concepts that depend on this)
                concepts_to_include.update(KG.successors(concept_name))
    else:
        # If no mastery data, show a sample of central concepts
        if len(concept_nodes_sorted) > 30:
            # Get most connected nodes
            degrees = dict(KG.degree())
            top_concepts = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:30]
            concepts_to_include = {node for node, _ in top_concepts if node in concept_nodes_sorted}
        else:
            concepts_to_include = set(concept_nodes_sorted)
    
    # Add nodes with mastery scores
    for concept_name in concepts_to_include:
        if concept_name in concept_nodes_sorted:  # Only include actual concept nodes
            mastery_score = concept_mastery.get(concept_name, 0.0)
            node_data = KG.nodes.get(concept_name, {})
            nodes.append({
                "id": concept_name,
                "label": concept_name,
                "mastery": mastery_score,
                "complexity": node_data.get("complexity", 5),
                "category": node_data.get("category", "general")
            })
    
    # Add edges between included nodes
    for source, target, data in KG.edges(data=True):
        if source in concepts_to_include and target in concepts_to_include:
            edges.append({
                "source": source,
                "target": target,
                "relation": data.get("relation", "prerequisite"),
                "method": data.get("method", "unknown")
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "concept_mastery": concept_mastery,
        "total_concepts": len(concept_mastery)
    }


@app.post("/submit-answer", response_model=SubmitAnswerResponse)
def submit_answer(req: SubmitAnswerRequest):
    session = get_or_create_session(req.student_id)

    concept_id = req.concept_id
    if concept_id not in ID_TO_CONCEPT:
        return SubmitAnswerResponse(
            success=False,
            knowledge_state=KnowledgeState(
                student_id=req.student_id,
                mastery=session.knowledge_state_dict(),
            ),
        )

    session.update_with_answer(concept_id, req.correct)

    ks = KnowledgeState(
        student_id=req.student_id,
        mastery=session.knowledge_state_dict(),
    )

    return SubmitAnswerResponse(success=True, knowledge_state=ks)


@app.post("/simulate-session", response_model=SimulateSessionResponse)
def simulate_session(req: SimulateSessionRequest):
    session = get_or_create_session(req.student_id)
    interactions: List[SimulatedInteraction] = []

    for step in range(1, req.n_steps + 1):
        topic = req.topic or choose_random_concept()
        if topic not in CONCEPT_TO_ID:
            topic = choose_random_concept()

        concept_id_int = CONCEPT_TO_ID[topic]
        difficulty = infer_difficulty_from_complexity(topic)

        # here you could also use MCQs, but we'll just use generator
        gen_content, gen_topic, gen_ctype, gen_diff, gen_concept_id = \
            adaptive_gen.generate_adaptive_content(
                student_id=req.student_id,
                topic=topic,
                content_type=None,
            )

        mastery_before = session.mastery.get(concept_id_int, 0.0)
        base_prob = 0.5
        prob_correct = 0.3 * base_prob + 0.7 * mastery_before
        correct = random.random() < prob_correct

        session.update_with_answer(concept_id_int, correct)

        interactions.append(
            SimulatedInteraction(
                step=step,
                topic=topic,
                concept_id=concept_id_int,
                content_type=(gen_ctype or "question"),
                difficulty=difficulty,  # type: ignore
                content=gen_content,
                correct=correct,
                mastery_after=session.knowledge_state_dict(),
            )
        )

    final_state = KnowledgeState(
        student_id=req.student_id,
        mastery=session.knowledge_state_dict(),
    )

    return SimulateSessionResponse(
        student_id=req.student_id,
        interactions=interactions,
        final_knowledge_state=final_state,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
