"""
Adaptive Content Generator

This script integrates:
1. OpenAI GPT-3.5 Turbo API for generating educational content
2. Transformer DKT model for predicting student performance
3. PEBG Knowledge Graph embeddings for understanding concept relationships
4. Content difficulty adaptation based on student performance

CHANGES:
- Replaced Flan-T5-Base model with OpenAI GPT-3.5 Turbo API for content generation
- Hard-coded OpenAI API key for simplicity
- Improved prompt engineering for better educational content generation
- FIX: Return and capture `concept_id` so it's defined in interactions

REQUIREMENTS:
- pip install openai torch numpy tqdm
"""

import os
import torch
import numpy as np
import json
import argparse
import random
import datetime
from openai import OpenAI
from tqdm import tqdm

# -----------------------------
# Config & Initialization
# -----------------------------
# Import transformer DKT model (expected to exist on your path)
# If your class name is different (e.g., TransformerDKTModel), adjust the import.
from transformer_dkt_model import TransformerDKT  # noqa: F401 (referenced later as optional)

# Initialize OpenAI client with API key (not recommended for production use)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------------
# Helper
# -----------------------------
def _pretty_print_block(title: str, text: str, width: int = 80):
    bar = "-" * width
    print(f"\n{bar}\n{title}\n{bar}\n{text}\n")


# -----------------------------
# Core Class
# -----------------------------
class AdaptiveContentGenerator:
    """
    Adaptive Content Generator that integrates OpenAI GPT-3.5 Turbo, DKT, and Knowledge Graph
    to generate personalized educational content with adaptive difficulty.
    """

    def __init__(
        self,
        dkt_model_path: str | None = None,
        embeddings_path: str | None = None,
        difficulty_levels=None,
        content_types=None,
        log_file: str = "generation_log.txt",
        seed: int = 42,
    ):
        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        self.content_types = content_types or ["explanation", "question", "example"]

        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize logging
        self.log_file = log_file
        with open(self.log_file, "a") as f:
            f.write(f"\n\n--- New Session: {datetime.datetime.now()} ---\n\n")

        # Load Knowledge Graph embeddings (optional)
        self.embeddings = None
        self.problem_embeddings = None
        self.skill_embeddings = None
        if embeddings_path and os.path.exists(embeddings_path):
            try:
                print(f"[INFO] Loading Knowledge Graph embeddings from {embeddings_path} ...")
                self.embeddings = torch.load(embeddings_path, map_location=device)

                print("[INFO] Embeddings summary:")
                for key, tensor in self.embeddings.items():
                    if isinstance(tensor, torch.Tensor):
                        print(f"  - {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
                    else:
                        print(f"  - {key}: (non-tensor) type={type(tensor)}")

                # Extract known keys if present
                if isinstance(self.embeddings, dict):
                    self.problem_embeddings = self.embeddings.get("pro_final_repre", None)
                    self.skill_embeddings = self.embeddings.get("skill_repre", None)
                    if self.problem_embeddings is not None:
                        print(f"[INFO] Using problem embeddings with shape: {tuple(self.problem_embeddings.shape)}")
                    else:
                        print("[WARN] 'pro_final_repre' not found in embeddings.")
                    if self.skill_embeddings is not None:
                        print(f"[INFO] Using skill embeddings with shape: {tuple(self.skill_embeddings.shape)}")
                    else:
                        print("[WARN] 'skill_repre' not found in embeddings.")
            except Exception as e:
                print(f"[ERROR] Error loading Knowledge Graph embeddings: {e}")

        # Load DKT model (optional)
        self.dkt_model = None
        if dkt_model_path and os.path.exists(dkt_model_path):
            try:
                print(f"[INFO] Loading DKT model from {dkt_model_path} ...")
                # If you saved via torch.save(model.state_dict()), you’ll need to reconstruct the class, then load_state_dict.
                # For simplicity, this assumes torch.save(model) was used; adjust as needed for your artifact.
                self.dkt_model = torch.load(dkt_model_path, map_location=device)
                print("[INFO] DKT model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Error loading DKT model: {e}")
                self.dkt_model = None

        # Initialize student state
        self.reset_student_state()

        print("[INFO] Adaptive Content Generator initialized.")

    def reset_student_state(self):
        """Reset the student state for a new student."""
        self.student_state = {
            "concept_history": [],          # List of (concept_id, correct) tuples
            "performance_predictions": {},  # Concept ID -> predicted performance
            "difficulty_levels": {},        # Concept ID -> current difficulty level
            "mastery_levels": {},           # Concept ID -> mastery level (0-1)
            "recommended_concepts": [],     # List of recommended concept IDs
        }

    # -----------------------------
    # Content Generation
    # -----------------------------
    def generate_content(
        self,
        topic: str,
        content_type: str | None = None,
        difficulty: str | None = None,
        max_tokens: int = 500,
        generate_variations: bool = False,
    ):
        """
        Generate educational content for a topic using OpenAI GPT-3.5 Turbo.
        """
        # Defaults
        content_type = content_type or random.choice(self.content_types)
        difficulty = difficulty if (difficulty in self.difficulty_levels) else "medium"

        # Build prompt template
        if content_type == "explanation":
            if difficulty == "easy":
                prompt = (
                    f"You are a math teacher explaining {topic} to beginners. "
                    "Write a clear, simple explanation of {topic} that includes: "
                    "1) A basic definition, 2) The key formula or concept, 3) A simple example showing how to use it. "
                    "Keep your explanation under 200 words and avoid complex terminology."
                )
            elif difficulty == "medium":
                prompt = (
                    f"You are a math teacher explaining {topic} to intermediate students. "
                    "Write a detailed explanation that includes: 1) A precise definition, "
                    "2) Key formulas and concepts, 3) Step-by-step examples, 4) Common applications. "
                    "Use appropriate mathematical terminology but explain any complex terms."
                )
            else:
                prompt = (
                    f"You are a university professor explaining {topic} to advanced students. "
                    "Write a comprehensive explanation including: 1) Formal definitions and theorems, "
                    "2) Proofs or derivations where applicable, 3) Complex examples demonstrating mastery, "
                    "4) Connections to other concepts, 5) Advanced applications."
                )
        elif content_type == "question":
            if difficulty == "easy":
                prompt = (
                    f"Create a beginner-level math question about {topic}. The question should: "
                    "1) Test basic understanding, 2) Include all necessary information, "
                    "3) Have a clear, straightforward solution process, 4) Include the answer with steps."
                )
            elif difficulty == "medium":
                prompt = (
                    f"Create an intermediate-level math problem about {topic}. The problem should: "
                    "1) Require applying the concept in a slightly complex scenario, "
                    "2) Test deeper understanding beyond formulas, 3) Require multiple steps, "
                    "4) Include the answer and detailed solution steps."
                )
            else:
                prompt = (
                    f"Create an advanced math problem about {topic}. The problem should: "
                    "1) Require sophisticated understanding, 2) Involve a complex or novel situation, "
                    "3) Potentially combine with other concepts, 4) Require creative reasoning, "
                    "5) Include a detailed solution with explanation."
                )
        else:  # "example"
            if difficulty == "easy":
                prompt = (
                    f"Provide a simple, real-world example of {topic} in action. Include: "
                    "1) A concrete situation, 2) Step-by-step explanation of how the concept applies, "
                    "3) The calculations or reasoning, suitable for beginners."
                )
            elif difficulty == "medium":
                prompt = (
                    f"Provide a practical application of {topic}. Include: "
                    "1) A specific scenario, 2) Detailed explanation of how the concept is used, "
                    "3) The mathematical process, 4) Why the application is useful."
                )
            else:
                prompt = (
                    f"Describe an advanced real-world application of {topic} in science or engineering. Include: "
                    "1) A sophisticated scenario, 2) How the concept enables this application, "
                    "3) Implementation details, 4) Benefits and insights."
                )

        # Log the prompt
        with open(self.log_file, "a") as f:
            f.write(f"PROMPT [{content_type}|{difficulty}|{topic}]: {prompt}\n")

        # Call OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful educational assistant. Generate clear, accurate, student-friendly content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.0,
            )

            content = response.choices[0].message.content.strip()

            # Log the generation
            with open(self.log_file, "a") as f:
                f.write(f"GENERATED: {content}\n\n")

            if generate_variations:
                variations = []
                for i in range(2):
                    variation_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful educational assistant. Generate clear, accurate, student-friendly content.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.9,
                        max_tokens=min(max_tokens * 2, 2000),
                        top_p=0.95,
                        frequency_penalty=0.3,
                        presence_penalty=0.1,
                    )
                    variation = variation_response.choices[0].message.content.strip()
                    with open(self.log_file, "a") as f:
                        f.write(f"VARIATION {i+1}: {variation}\n\n")
                    variations.append(variation)

                return {"main": content, "variations": variations}

            return content

        except Exception as e:
            error_msg = f"Error generating content with OpenAI API: {str(e)}"
            print(f"[ERROR] {error_msg}")
            with open(self.log_file, "a") as f:
                f.write(f"ERROR: {error_msg}\n\n")
            return f"Error generating content: {str(e)}"

    # -----------------------------
    # Adaptive Wrapper
    # -----------------------------
    def generate_adaptive_content(
        self,
        student_id: str,
        topic: str | None = None,
        content_type: str | None = None,
        max_tokens: int = 500,
    ):
        """
        Generate adaptive content for a student.
        (Currently uses a simple heuristic for difficulty; wire to DKT later.)
        """
        topic = topic or random.choice(
            [
                "Pythagorean theorem",
                "quadratic equations",
                "linear equations",
                "prime numbers",
                "fractions",
                "derivatives",
                "integrals",
                "logarithms",
                "vectors",
                "matrices",
            ]
        )

        # Map topic to concept ID (simple deterministic mapping for demo)
        concept_id = (hash(topic) % 100) + 1  # 1..100

        # TODO: If DKT available, compute performance and pick difficulty from prediction
        difficulty = "medium"

        # Generate content
        content = self.generate_content(
            topic=topic,
            content_type=content_type,
            difficulty=difficulty,
            max_tokens=max_tokens,
        )

        # Return concept_id as well (FIX)
        return content, topic, content_type, difficulty, concept_id

    def simulate_student_interaction(self, num_interactions: int = 10):
        """
        Simulate student interactions to demonstrate adaptive content generation.
        """
        self.reset_student_state()
        student_id = f"student_{random.randint(1000, 9999)}"
        interactions = []

        for i in range(num_interactions):
            _pretty_print_block("Interaction", f"{i+1}/{num_interactions}")

            # Capture concept_id (FIX)
            content, topic, content_type, difficulty, concept_id = self.generate_adaptive_content(
                student_id
            )

            _pretty_print_block(
                "Generated",
                f"Topic: {topic}\nContent Type: {content_type or 'auto'}\nDifficulty: {difficulty}\n\nContent:\n{content}",
            )

            # Simulate correctness by difficulty
            if difficulty == "easy":
                correct = random.random() < 0.8
            elif difficulty == "medium":
                correct = random.random() < 0.6
            else:
                correct = random.random() < 0.4

            print(f"[STUDENT RESPONSE] {'Correct' if correct else 'Incorrect'}")

            interactions.append(
                {
                    "student_id": student_id,
                    "interaction_id": i + 1,
                    "topic": topic,
                    "content_type": content_type,
                    "difficulty": difficulty,
                    "content": content,
                    "correct": correct,
                    "concept_id": concept_id,  # now defined
                }
            )

            # Update simple state (optional demo)
            self.student_state["concept_history"].append((concept_id, correct))

        return interactions


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Adaptive Content Generator")

    parser.add_argument(
        "--dkt_model_path",
        type=str,
        default="/Users/yugandhargopu/Desktop/capstone/dkt_model.pt",
        help="Path to the DKT model (optional)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="/Users/yugandhargopu/Desktop/capstone/embedding_200.pt",
        help="Path to the Knowledge Graph embeddings (optional)",
    )
    parser.add_argument("--topic", type=str, default=None, help="Topic to generate content for")
    parser.add_argument(
        "--content_type",
        type=str,
        default=None,
        choices=["explanation", "question", "example"],
        help="Type of content to generate",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard"],
        help="Difficulty level (overrides adaptive choice)",
    )
    parser.add_argument("--simulate", action="store_true", help="Simulate student interactions")
    parser.add_argument("--num_interactions", type=int, default=5, help="Number of interactions")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save outputs (JSON)")
    parser.add_argument(
        "--generate_variations",
        action="store_true",
        help="Also generate 2 variations of the content",
    )
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens for generation")

    args = parser.parse_args()

    generator = AdaptiveContentGenerator(
        dkt_model_path=args.dkt_model_path,
        embeddings_path=args.embeddings_path,
    )

    if args.simulate:
        interactions = generator.simulate_student_interaction(args.num_interactions)
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(interactions, f, indent=2)
            print(f"[INFO] Saved interactions to {args.output_file}")
    else:
        topic = args.topic or "Pythagorean theorem"
        content = generator.generate_content(
            topic=topic,
            content_type=args.content_type,
            difficulty=args.difficulty,
            max_tokens=args.max_tokens,
            generate_variations=args.generate_variations,
        )

        _pretty_print_block(
            "Single Generation",
            f"Topic: {topic}\nContent Type: {args.content_type or 'auto'}\n"
            f"Difficulty: {args.difficulty or 'auto'}\n\nGenerated Content:\n{content}",
        )

        if args.output_file:
            payload = {
                "topic": topic,
                "content_type": args.content_type,
                "difficulty": args.difficulty,
                "content": content,
            }
            with open(args.output_file, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[INFO] Saved content to {args.output_file}")


if __name__ == "__main__":
    main()
