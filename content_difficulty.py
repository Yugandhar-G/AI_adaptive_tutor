"""
Transformer-based Content Difficulty Adaptation

This module provides adaptive difficulty selection based on student knowledge state
predicted by the Transformer DKT model.

Key Features:
- Student knowledge prediction using Transformer DKT
- Difficulty recommendation (easy/medium/hard)
- Batch processing support
- Knowledge state caching

Author: Yugandhar
Date: 2025
"""

import numpy as np
import os
import torch
import torch.nn as nn
import sys
from typing import Dict, List, Optional, Tuple

# Import transformer model
from transformer_dkt_model import TransformerDKT


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DKT_MODEL_PATH = os.path.expanduser(
    "/Users/yugandhargopu/Desktop/capstone/dkt_model.pt"
)
DEFAULT_EDNET_DATA_PATH = os.path.expanduser(
    "/Users/yugandhargopu/Desktop/capstone/ednet/ednet.npz"
)
DEFAULT_EMBEDDINGS_PATH = os.path.expanduser(
    "/Users/yugandhargopu/Desktop/capstone/embedding_200.pt"
)

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_student_data_from_ednet(
    ednet_npz_path: str,
    max_students: Optional[int] = None
) -> Tuple[Dict, Dict]:
    """
    Load student interaction data from EdNet npz file.
    
    Args:
        ednet_npz_path: Path to ednet.npz file
        max_students: Maximum number of students to load
    
    Returns:
        student_data: Dictionary containing student interaction data
        concept_map: Dictionary mapping concept IDs to indices
    """
    print(f"Loading EdNet data from: {ednet_npz_path}")
    
    if not os.path.exists(ednet_npz_path):
        raise FileNotFoundError(f"EdNet file not found: {ednet_npz_path}")
    
    # Load data
    data = np.load(ednet_npz_path)
    
    # Extract arrays
    skills = data['skill']        # [num_students, max_seq_len]
    answers = data['y']           # [num_students, max_seq_len]
    real_lens = data['real_len']  # [num_students]
    skill_num = int(data['skill_num'])
    
    print(f"Loaded data: {len(skills)} students, {skill_num} skills")
    
    # Limit students if requested
    if max_students is not None and max_students < len(skills):
        skills = skills[:max_students]
        answers = answers[:max_students]
        real_lens = real_lens[:max_students]
    
    # Create concept map (skill_id = concept_id)
    unique_skills = np.unique(skills[skills > 0])
    concept_map = {
        int(skill): int(skill) 
        for skill in unique_skills
    }
    
    print(f"Created concept map with {len(concept_map)} concepts")
    
    # Store student data
    student_data = {
        'skills': skills,
        'answers': answers,
        'real_lens': real_lens,
        'num_students': len(skills)
    }
    
    return student_data, concept_map


# =============================================================================
# DIFFICULTY ADAPTER
# =============================================================================

class TransformerDifficultyAdapter:
    """
    Adapter that uses Transformer DKT model to predict student knowledge
    and recommend appropriate difficulty levels.
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_DKT_MODEL_PATH,
        ednet_data_path: str = DEFAULT_EDNET_DATA_PATH,
        embeddings_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Transformer Difficulty Adapter.
        
        Args:
            model_path: Path to trained Transformer DKT model
            ednet_data_path: Path to EdNet data
            embeddings_path: Path to pretrained embeddings (optional)
            device: PyTorch device to use
        """
        self.device = device if device is not None else DEVICE
        self.model = None
        self.student_data = None
        self.concept_map = None
        self.knowledge_cache = {}  # Cache for student knowledge states
        
        print("="*60)
        print("Initializing Transformer Difficulty Adapter")
        print("="*60)
        
        # Load student data
        self._load_student_data(ednet_data_path)
        
        # Load pretrained embeddings (optional)
        self.pretrained_embeddings = None
        if embeddings_path and os.path.exists(embeddings_path):
            self._load_embeddings(embeddings_path)
        
        # Load DKT model
        self._load_dkt_model(model_path)
        
        print("Adapter initialized successfully!")
        print("="*60 + "\n")
    
    def _load_student_data(self, ednet_data_path: str):
        """Load student data from EdNet."""
        self.student_data, self.concept_map = load_student_data_from_ednet(
            ednet_data_path
        )
    
    def _load_embeddings(self, embeddings_path: str):
        """Load pretrained embeddings."""
        try:
            embeddings = torch.load(embeddings_path, map_location=self.device)
            if 'pro_final_repre' in embeddings:
                self.pretrained_embeddings = embeddings['pro_final_repre']
                print(f"Loaded pretrained embeddings: {self.pretrained_embeddings.shape}")
        except Exception as e:
            print(f"Warning: Could not load embeddings: {e}")
            self.pretrained_embeddings = None
    
    def _load_dkt_model(self, model_path: str):
        """Load the pre-trained Transformer DKT model."""
        print(f"Loading DKT model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model parameters
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine number of skills from model weights
        if 'skill_embed.weight' in state_dict:
            num_skills = state_dict['skill_embed.weight'].shape[0] - 1
        else:
            raise ValueError("Cannot determine num_skills from checkpoint")
        
        print(f"Detected {num_skills} skills in the model")
        
        # Create model with matching architecture
        self.model = TransformerDKT(
            num_skills=num_skills,
            hidden_size=128,
            num_layers=3,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Model loaded successfully")
        if 'val_auc' in checkpoint:
            print(f"Model validation AUC: {checkpoint['val_auc']:.4f}")
    
    def predict_student_knowledge(
        self,
        student_id: int,
        concept_ids: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Predict student's knowledge state for specific concepts.
        
        Args:
            student_id: Student ID (index in EdNet data)
            concept_ids: List of concept IDs (None = all concepts)
        
        Returns:
            Dictionary mapping concept_id -> mastery_probability
        """
        if self.model is None or self.student_data is None:
            print("Error: Model or data not loaded")
            return {}
        
        # Check cache
        if student_id in self.knowledge_cache:
            cached = self.knowledge_cache[student_id]
            if concept_ids is None:
                return cached
            return {cid: cached.get(cid, 0.5) for cid in concept_ids}
        
        # Validate student ID
        if student_id >= self.student_data['num_students']:
            print(f"Warning: Student ID {student_id} out of range")
            return {}
        
        # Get student's interaction history
        seq_len = int(self.student_data['real_lens'][student_id])
        skills = self.student_data['skills'][student_id, :seq_len]
        answers = self.student_data['answers'][student_id, :seq_len]
        
        # Filter out padding (skill_id = 0)
        valid_idx = skills > 0
        skills = skills[valid_idx]
        answers = answers[valid_idx]
        
        if len(skills) == 0:
            print(f"Warning: Student {student_id} has no valid interactions")
            return {}
        
        # Convert to tensors
        skills_tensor = torch.LongTensor(skills).unsqueeze(0).to(self.device)
        answers_tensor = torch.FloatTensor(answers).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(skills_tensor, answers_tensor)
        
        # Extract knowledge state from last timestep
        last_pred = predictions[0, -1, :].cpu().numpy()
        
        # Create knowledge state dictionary
        knowledge_state = {}
        for concept_id in range(1, len(last_pred)):
            knowledge_state[concept_id] = float(last_pred[concept_id])
        
        # Cache the result
        self.knowledge_cache[student_id] = knowledge_state
        
        # Return requested concepts
        if concept_ids is None:
            return knowledge_state
        return {cid: knowledge_state.get(cid, 0.5) for cid in concept_ids}
    
    def get_appropriate_difficulty(
        self,
        student_id: int,
        concept_ids: List[int]
    ) -> str:
        """
        Determine appropriate difficulty level for a student.
        
        Args:
            student_id: Student ID
            concept_ids: List of relevant concept IDs
        
        Returns:
            Difficulty level: "easy", "medium", or "hard"
        """
        if not concept_ids:
            return "medium"
        
        # Get knowledge state
        knowledge_state = self.predict_student_knowledge(student_id, concept_ids)
        
        if not knowledge_state:
            return "medium"
        
        # Calculate average mastery
        avg_mastery = np.mean(list(knowledge_state.values()))
        
        # Determine difficulty based on thresholds
        if avg_mastery < 0.4:
            return "easy"
        elif avg_mastery < 0.7:
            return "medium"
        else:
            return "hard"
    
    def update_student_knowledge(
        self,
        student_id: int,
        concept_id: int,
        correct: bool
    ):
        """
        Update student's knowledge state after answering a question.
        
        This clears the cache to force re-prediction on next access.
        
        Args:
            student_id: Student ID
            concept_id: Concept that was tested
            correct: Whether answer was correct
        """
        # Clear cache for this student
        if student_id in self.knowledge_cache:
            del self.knowledge_cache[student_id]
        
        print(f"Updated knowledge for student {student_id}, concept {concept_id}")
    
    def batch_predict_knowledge_states(
        self,
        student_ids: List[int],
        concept_ids: Optional[List[int]] = None
    ) -> Dict[int, Dict[int, float]]:
        """
        Predict knowledge states for multiple students.
        
        Args:
            student_ids: List of student IDs
            concept_ids: List of concept IDs (None = all)
        
        Returns:
            Dictionary mapping student_id -> knowledge_state
        """
        knowledge_states = {}
        for student_id in student_ids:
            knowledge_states[student_id] = self.predict_student_knowledge(
                student_id, concept_ids
            )
        return knowledge_states
    
    def batch_get_appropriate_difficulties(
        self,
        student_ids: List[int],
        concept_ids: List[int]
    ) -> Dict[int, str]:
        """
        Determine appropriate difficulties for multiple students.
        
        Args:
            student_ids: List of student IDs
            concept_ids: List of relevant concept IDs
        
        Returns:
            Dictionary mapping student_id -> difficulty
        """
        difficulties = {}
        for student_id in student_ids:
            difficulties[student_id] = self.get_appropriate_difficulty(
                student_id, concept_ids
            )
        return difficulties
    
    def get_concept_mastery_report(
        self,
        student_id: int,
        top_k: int = 10
    ) -> Optional[Dict]:
        """
        Get a report of student's mastery levels.
        
        Args:
            student_id: Student ID
            top_k: Number of top/bottom concepts to show
        
        Returns:
            Dictionary with mastery statistics
        """
        knowledge_state = self.predict_student_knowledge(student_id)
        
        if not knowledge_state:
            return None
        
        # Sort by mastery level
        sorted_concepts = sorted(
            knowledge_state.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'average_mastery': np.mean(list(knowledge_state.values())),
            'strongest_concepts': sorted_concepts[:top_k],
            'weakest_concepts': sorted_concepts[-top_k:],
            'total_concepts': len(knowledge_state)
        }
    
    def clear_cache(self, student_id: Optional[int] = None):
        """
        Clear knowledge state cache.
        
        Args:
            student_id: Specific student to clear (None = clear all)
        """
        if student_id is None:
            self.knowledge_cache.clear()
        elif student_id in self.knowledge_cache:
            del self.knowledge_cache[student_id]


# =============================================================================
# DEMO & TESTING
# =============================================================================

def demo_difficulty_adapter():
    """Demo function to show how to use the difficulty adapter."""
    print("\n" + "="*60)
    print("Transformer Difficulty Adapter Demo")
    print("="*60 + "\n")
    
    # Initialize adapter
    adapter = TransformerDifficultyAdapter()
    
    # Test with first 5 students
    for student_id in range(5):
        print(f"\n--- Student {student_id} ---")
        
        # Get knowledge state
        knowledge = adapter.predict_student_knowledge(student_id)
        
        if knowledge:
            avg_mastery = np.mean(list(knowledge.values()))
            print(f"Average mastery: {avg_mastery:.3f}")
            print(f"Concepts tracked: {len(knowledge)}")
            
            # Get difficulty for sample concepts
            sample_concepts = list(knowledge.keys())[:5]
            difficulty = adapter.get_appropriate_difficulty(
                student_id, sample_concepts
            )
            print(f"Recommended difficulty: {difficulty}")
            
            # Show mastery report
            report = adapter.get_concept_mastery_report(student_id, top_k=3)
            if report:
                print(f"\nStrongest concepts:")
                for cid, mastery in report['strongest_concepts']:
                    print(f"  Concept {cid}: {mastery:.3f}")
                print(f"Weakest concepts:")
                for cid, mastery in report['weakest_concepts']:
                    print(f"  Concept {cid}: {mastery:.3f}")


def main():
    """Main function for testing."""
    demo_difficulty_adapter()


if __name__ == "__main__":
    main()