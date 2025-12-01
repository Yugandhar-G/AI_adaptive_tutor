# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
from functools import lru_cache
import os, pickle, uuid

# === Your artifact paths ===
GRAPH_PICKLE = "khan_knowledge_graph_nlp.pkl"
EMB_PICKLE   = "khan_embeddings_nlp.pkl"
TRANS_DKT    = "dkt_output/dkt_model.pt"
LSTM_DIR     = "/Users/yugandhargopu/Desktop/capstone/models"
LSTM_IDXMAP  = os.path.join(LSTM_DIR, "problem_to_idx.pkl")

app = FastAPI(title="LARIA 4-endpoint API", version="1.0")

# -------- state --------
SESSIONS: Dict[str, Dict[str, Any]] = {}   # student_id -> mastery/history
QUESTIONS: Dict[str, Dict[str, Any]] = {}  # qid -> record

def ensure_session(student_id: str):
    if student_id not in SESSIONS:
        SESSIONS[student_id] = {"mastery": {}, "history": []}
    return SESSIONS[student_id]

# -------- loaders --------
@lru_cache(maxsize=1)
def load_graph_nodes() -> List[str]:
    if not os.path.exists(GRAPH_PICKLE):
        return []
    with open(GRAPH_PICKLE, "rb") as f:
        G = pickle.load(f)
    return [str(n) for n in list(G.nodes)]

@lru_cache(maxsize=1)
def load_embeddings() -> Dict[str, Any]:
    if not os.path.exists(EMB_PICKLE):
        return {}
    with open(EMB_PICKLE, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "title_embeddings" in obj:
        return obj["title_embeddings"]
    return obj if isinstance(obj, dict) else {}

# -------- stubs (replace with your generator + DKT) --------
def gen_content(topic: str, difficulty: str) -> str:
    return f"[{difficulty.upper()}] Main content for **{topic}**.\n- Core idea\n- Why it matters\n- Tip tailored for {difficulty} level."

def gen_example(topic: str, difficulty: str) -> str:
    return f"[{difficulty.upper()}] Worked example on **{topic}**:\n1) Step A\n2) Step B\n3) Result + brief reasoning."

def gen_questions(topic: str, difficulty: str, n: int) -> List[Dict[str, str]]:
    out = []
    for i in range(n):
        qid = str(uuid.uuid4())
        text = f"[{difficulty.upper()}] Q{i+1} on {topic}: Solve and explain your steps."
        QUESTIONS[qid] = {"topic": topic, "difficulty": difficulty}
        out.append({"question_id": qid, "text": text})
    return out

def dkt_update(student_id: str, concept_id: str, was_correct: bool) -> float:
    s = ensure_session(student_id)
    cur = s["mastery"].get(concept_id, 0.55)
    target = 0.9 if was_correct else 0.1
    new_val = round(0.8 * cur + 0.2 * target, 4)
    s["mastery"][concept_id] = new_val
    return new_val

# -------- models --------
class TopicsOut(BaseModel):
    ok: bool
    topics: List[str]
    total: int

class ContentIn(BaseModel):
    student_id: str
    topic: str
    difficulty: Literal["easy","medium","hard"]

class ContentOut(BaseModel):
    ok: bool
    content: str

class ExampleIn(BaseModel):
    student_id: str
    topic: str
    difficulty: Literal["easy","medium","hard"]

class ExampleOut(BaseModel):
    ok: bool
    example: str

class QuestionsIn(BaseModel):
    student_id: str
    topic: str
    difficulty: Literal["easy","medium","hard"]
    n: int = 3

class QuestionItem(BaseModel):
    question_id: str
    text: str

class QuestionsOut(BaseModel):
    ok: bool
    questions: List[QuestionItem]

class SubmitAns(BaseModel):
    student_id: str
    question_id: str
    was_correct: bool
    time_on_task: Optional[float] = None
    attempts: Optional[int] = 1

# -------- ENDPOINTS (exactly 4) --------

# 1) Topics from Knowledge Graph (for dropdown)
@app.get("/topics", response_model=TopicsOut)
def topics(limit: int = 500):
    nodes = load_graph_nodes()
    if not nodes:
        raise HTTPException(status_code=404, detail="Knowledge graph not found or empty.")
    return TopicsOut(ok=True, topics=nodes[:limit], total=len(nodes))

# 2) Generate difficulty-aware MAIN CONTENT
@app.post("/content", response_model=ContentOut)
def content(req: ContentIn):
    nodes = load_graph_nodes()
    if req.topic not in nodes:
        raise HTTPException(status_code=400, detail="Topic must be selected from /topics.")
    ensure_session(req.student_id)
    text = gen_content(req.topic, req.difficulty)
    return ContentOut(ok=True, content=text)

# 3) Generate WORKED EXAMPLE for same topic/difficulty
@app.post("/example", response_model=ExampleOut)
def example(req: ExampleIn):
    nodes = load_graph_nodes()
    if req.topic not in nodes:
        raise HTTPException(status_code=400, detail="Topic must be selected from /topics.")
    ensure_session(req.student_id)
    text = gen_example(req.topic, req.difficulty)
    return ExampleOut(ok=True, example=text)

# 4) QUESTIONS (POST generate; PUT submit + DKT update)
@app.post("/questions", response_model=QuestionsOut)
def questions(req: QuestionsIn):
    nodes = load_graph_nodes()
    if req.topic not in nodes:
        raise HTTPException(status_code=400, detail="Topic must be selected from /topics.")
    ensure_session(req.student_id)
    qs = gen_questions(req.topic, req.difficulty, req.n)
    return QuestionsOut(ok=True, questions=[QuestionItem(**q) for q in qs])

@app.put("/questions")
def submit_answer(req: SubmitAns):
    if req.question_id not in QUESTIONS:
        raise HTTPException(status_code=404, detail="Unknown question_id")
    topic = QUESTIONS[req.question_id]["topic"]
    # use topic as concept_id placeholder; swap to your concept-id mapping if needed
    new_mastery = dkt_update(req.student_id, topic, req.was_correct)
    return {"ok": True, "concept_id": topic, "new_mastery": new_mastery}
