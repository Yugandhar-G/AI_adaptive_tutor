"""
MCQ Generator for Adaptive Learning System

This module generates well-formatted multiple-choice questions with:
1. Proper MCQ structure (question + 4 options)
2. Randomized correct option placement
3. Plausible distractors
4. Answer evaluation
5. Integration with DKT for knowledge state updates
6. Robust validation to ensure correct answer is always present

Author: Capstone Project
Date: 2025
"""

import json
import random
import re
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_MODEL_AVAILABLE = True
    print("[INFO] Loading semantic similarity model...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("[INFO] Semantic similarity model loaded successfully")
except ImportError:
    SEMANTIC_MODEL_AVAILABLE = False
    semantic_model = None
    print("[WARNING] sentence-transformers not available. Install with: pip install sentence-transformers")
    print("[WARNING] Falling back to OpenAI embeddings for semantic similarity")
except Exception as e:
    SEMANTIC_MODEL_AVAILABLE = False
    semantic_model = None
    print(f"[WARNING] Could not load semantic model: {e}. Using OpenAI embeddings instead.")


class MCQGenerator:
    """
    Generate and manage multiple-choice questions with randomized options.
    """

    def __init__(self, seed: int = None):
        """
        Initialize the MCQ Generator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)

        self.generation_history = []
        self.recent_questions = []  # Track recent questions to avoid repetition
        self.recent_topics = {}  # Track questions by topic to avoid repetition
        self.question_embeddings = []  # Store semantic embeddings for similarity checking
        self.topic_embeddings = {}  # Store embeddings by topic

    def generate_mcq(
        self,
        topic: str,
        difficulty: str = "medium",
        concept_id: int = 0,
        max_tokens: int = 800,
        max_retries: int = 3,
        question_num: int = 0,
    ) -> Dict:
        """
        Generate a multiple-choice question with 4 options.
        Includes retry logic and validation to ensure correct answer is present.

        Args:
            topic: The mathematical topic/concept
            difficulty: One of "easy", "medium", "hard"
            concept_id: Identifier for the concept (for DKT tracking)
            max_tokens: Maximum tokens for GPT generation
            max_retries: Maximum number of retry attempts if generation fails
            question_num: Question number (used to vary question types)

        Returns:
            Dictionary containing:
                - question: The question text
                - options: List of 4 option strings
                - correct_index: Index (0-3) of the correct answer
                - explanation: Explanation of the correct answer
                - topic: The topic
                - difficulty: The difficulty level
                - concept_id: The concept ID
        """

        # Build the prompt based on difficulty with variations
        prompt = self._build_mcq_prompt(topic, difficulty, question_num)
        
        # Add strong instruction to avoid similar questions from history
        if topic in self.recent_topics and len(self.recent_topics[topic]) > 0:
            recent_for_topic = self.recent_topics[topic][-15:]  # Last 15 questions for this topic
            import re
            all_used_numbers = set()
            prompt += f"\n\n🚫🚫🚫 CRITICAL - DO NOT REPEAT: You have already generated these {len(recent_for_topic)} questions about {topic}:\n"
            for idx, prev_q in enumerate(recent_for_topic, 1):
                # Extract numbers from each question
                numbers = re.findall(r'\b\d+\b', prev_q)
                all_used_numbers.update(numbers)
                numbers_str = f" [uses: {', '.join(numbers[:5])}]" if numbers else ""
                prompt += f"{idx}. {prev_q[:90]}{numbers_str}...\n"
            
            # Show all used numbers to avoid
            if all_used_numbers:
                prompt += f"\n📊 NUMBERS TO AVOID: {', '.join(sorted(all_used_numbers, key=int)[:20])}\n"
                prompt += "DO NOT use any of these numbers in your new question! Use COMPLETELY DIFFERENT numbers!\n"
            
            prompt += "\n\n⚠️⚠️⚠️ YOUR NEW QUESTION MUST:\n"
            prompt += "1. Use COMPLETELY DIFFERENT numbers - NONE of the numbers listed above!\n"
            prompt += "2. Have a COMPLETELY DIFFERENT scenario/context\n"
            prompt += "3. Use a COMPLETELY DIFFERENT question format/structure\n"
            prompt += "4. Test a COMPLETELY DIFFERENT aspect or application of the concept\n"
            prompt += "5. Be UNIQUE - if it uses the same numbers OR similar wording to ANY question above, it's WRONG\n"
            prompt += "6. AVOID classic textbook examples\n"
            prompt += "\n🔴 REJECTION CRITERIA: Your question will be REJECTED if:\n"
            prompt += "- It has 75% or higher SEMANTIC SIMILARITY to any previous question (same meaning, different words)\n"
            prompt += "- It uses 2 or more of the same numbers as any previous question\n"
            prompt += "- It has 75% or more word overlap with any previous question\n"
            prompt += "- It follows the same structure/format as any previous question\n"
            prompt += "- It tests the SAME aspect of the concept as any previous question\n"
            prompt += "\nIf you generate a question that's similar to any above, you have FAILED. Generate something COMPLETELY DIFFERENT.\n"
        
        # Also check general recent questions
        if len(self.recent_questions) > 0:
            recent_summary = "\n".join([f"{idx}. {q}" for idx, q in enumerate(self.recent_questions[-10:], 1)])
            prompt += f"\n\n⚠️⚠️⚠️ Also avoid these last 10 questions (any topic):\n{recent_summary}\n"
            prompt += "\nYour question must be COMPLETELY DIFFERENT from ALL of these. Check each one carefully.\n"
        
        # Add general warning about avoiding classic textbook examples for ALL topics
        prompt += "\n\n🚨 GENERAL WARNING - APPLIES TO ALL TOPICS:\n"
        prompt += "DO NOT use classic textbook examples or common numerical examples!\n"
        prompt += "DO NOT repeat the same numbers, scenarios, or problem structures!\n"
        prompt += "If you've used specific numbers before (like 3, 4, 5 or any other combination), use COMPLETELY DIFFERENT numbers!\n"
        prompt += "Be creative and use unique scenarios, different contexts, and varied numerical values!\n"

        # Retry logic for robust generation
        current_prompt = prompt  # Keep original prompt
        for attempt in range(max_retries):
            content = None
            try:
                # Vary temperature more significantly for more diversity
                # Higher temperature = more creative/varied outputs
                import random
                base_temp = 0.9  # Higher base temperature for more creativity
                variation = (question_num % 5) * 0.15 + random.random() * 0.1  # More variation
                temperature = min(base_temp + variation, 1.0)  # Cap at 1.0, but allow more variation
                
                # Use current_prompt which may be updated on retries
                use_prompt = current_prompt
                
                # Add timeout to prevent hanging (30 seconds per API call)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": self._build_system_prompt(),
                        },
                        {"role": "user", "content": use_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    timeout=30.0,  # 30 second timeout per API call
                )

                # Parse the response
                content = response.choices[0].message.content.strip()
                
                # Clean up any markdown code blocks if present
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()
                
                mcq_data = json.loads(content)

                # Validate and extract components with strict checking
                validated_mcq = self._validate_and_extract_mcq(mcq_data, topic, difficulty, concept_id)
                
                if validated_mcq:
                    question_text = validated_mcq["question"]
                    
                    # Check for similarity with recent questions - reject if too similar
                    # First check semantic similarity (more accurate)
                    is_semantically_similar, semantic_score = self._check_semantic_similarity(question_text, topic, threshold=0.75)
                    
                    # Also check structural similarity (numbers, words, etc.)
                    is_structurally_similar = self._check_similarity(question_text, topic)
                    
                    # Reject if either check fails
                    is_too_similar = is_semantically_similar or is_structurally_similar
                    
                    if is_too_similar:
                        if attempt < max_retries - 1:
                            reason = []
                            if is_semantically_similar:
                                reason.append(f"semantic similarity ({semantic_score:.3f})")
                            if is_structurally_similar:
                                reason.append("structural similarity (numbers/words)")
                            
                            print(f"[WARNING] Question too similar to previous ones, retrying (attempt {attempt + 1})...")
                            print(f"[WARNING] Reason: {', '.join(reason)}")
                            print(f"[DEBUG] Question: {question_text[:100]}...")
                            
                            # Update prompt with rejection feedback for next attempt
                            current_prompt += f"\n\n🚨🚨🚨 REJECTED: The question you just generated is too similar to previous ones!\n"
                            current_prompt += f"Rejection reason: {', '.join(reason)}\n"
                            if is_semantically_similar:
                                current_prompt += f"Semantic similarity score: {semantic_score:.3f} (threshold: 0.75) - too high!\n"
                            current_prompt += f"Rejected question: {question_text[:150]}...\n"
                            current_prompt += f"Generate something COMPLETELY DIFFERENT with:\n"
                            current_prompt += f"- DIFFERENT semantic meaning (test a different aspect of the concept)\n"
                            current_prompt += f"- DIFFERENT numbers (avoid any numbers you've seen before)\n"
                            current_prompt += f"- DIFFERENT scenario/context\n"
                            current_prompt += f"- DIFFERENT question structure\n"
                            current_prompt += f"- DIFFERENT approach to testing the concept\n"
                            time.sleep(1.0)  # Longer delay before retry
                            continue
                        else:
                            print(f"[ERROR] Failed to generate unique question after {max_retries} attempts")
                            print(f"[ERROR] Returning question anyway, but it may be similar to previous ones")
                            # Still return it but log the issue - better than failing completely
                    
                    # Store in history
                    self.generation_history.append(validated_mcq)
                    
                    # Get and store semantic embedding
                    question_embedding = self._get_question_embedding(question_text)
                    question_embedding = question_embedding / np.linalg.norm(question_embedding)  # Normalize
                    
                    # Track recent questions to avoid repetition
                    self.recent_questions.append(question_text)
                    self.question_embeddings.append((question_embedding, question_text))
                    if len(self.recent_questions) > 30:  # Keep last 30 questions
                        self.recent_questions.pop(0)
                        self.question_embeddings.pop(0)
                    
                    # Track by topic
                    if topic not in self.recent_topics:
                        self.recent_topics[topic] = []
                        self.topic_embeddings[topic] = []
                    self.recent_topics[topic].append(question_text)
                    self.topic_embeddings[topic].append((question_embedding, question_text))
                    if len(self.recent_topics[topic]) > 20:  # Keep last 20 per topic
                        self.recent_topics[topic].pop(0)
                        self.topic_embeddings[topic].pop(0)
                    
                    return validated_mcq
                else:
                    print(f"[WARNING] Attempt {attempt + 1} failed validation, retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Brief delay before retry
                        continue

            except json.JSONDecodeError as e:
                print(f"[ERROR] Attempt {attempt + 1}: Failed to parse JSON: {e}")
                if content:
                    print(f"[DEBUG] Response content: {content[:200]}...")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1}: Error generating MCQ: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue

        # If all retries failed, return fallback
        print(f"[WARNING] All {max_retries} attempts failed. Using fallback MCQ.")
        return self._generate_fallback_mcq(topic, difficulty, concept_id)

    def _build_system_prompt(self) -> str:
        """Build the system prompt with explicit JSON structure requirements."""
        return (
            "You are an expert math educator creating diverse, engaging multiple-choice questions. "
            "You MUST respond with valid JSON only, no additional text.\n\n"
            "Required JSON structure (all fields are mandatory):\n"
            "{\n"
            '  "question": "The question text (must be a complete, clear question)",\n'
            '  "correct_answer": "The correct answer (must be a specific value or statement)",\n'
            '  "distractor_1": "First incorrect option (plausible but wrong)",\n'
            '  "distractor_2": "Second incorrect option (plausible but wrong)",\n'
            '  "distractor_3": "Third incorrect option (plausible but wrong)",\n'
            '  "explanation": "Step-by-step explanation of why the correct answer is right"\n'
            "}\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. All 5 fields (question, correct_answer, distractor_1, distractor_2, distractor_3, explanation) MUST be present\n"
            "2. All fields must be non-empty strings\n"
            "3. The correct_answer must be a specific, concrete answer (not vague)\n"
            "4. Distractors should be plausible but clearly incorrect\n"
            "5. For numerical answers, provide exact numbers or expressions\n"
            "6. Ensure mathematical notation is clear and correct\n"
            "7. The explanation should clearly justify the correct answer\n"
            "8. VARY your question formats, contexts, and approaches - avoid repetitive patterns\n"
            "9. Use creative scenarios, different problem structures, and varied question styles"
        )

    def _get_question_type_variations(self) -> List[str]:
        """Get different question type variations to ensure diversity."""
        return [
            "a word problem with a real-world scenario (like sharing items, calculating costs, measuring distances)",
            "a conceptual question about properties, definitions, or relationships",
            "a problem requiring multiple steps or combining operations",
            "a comparison question (which is larger/smaller, more/less, etc.)",
            "a problem involving estimation or approximation",
            "a question about identifying patterns or sequences",
            "a problem requiring interpretation of results in context",
            "a question about when or how to apply the concept in different situations",
            "a problem involving units, measurements, or conversions",
            "a question testing understanding of inverse operations or relationships",
            "a problem with a visual or geometric context",
            "a question about common mistakes or misconceptions",
            "a problem requiring problem-solving strategy selection",
            "a question about applications in science, engineering, or daily life",
            "a problem involving fractions, decimals, or percentages related to the concept"
        ]
    
    def _get_question_format_variations(self) -> List[str]:
        """Get different question formats to ensure variety."""
        return [
            "What is...",
            "Which of the following...",
            "Solve the following...",
            "Identify the...",
            "Determine...",
            "Find...",
            "Calculate...",
            "Evaluate...",
            "Choose the correct...",
            "Select the best...",
            "If..., then what is...",
            "Given that..., what is...",
            "Which statement is true about...",
            "What would happen if...",
            "How would you...",
            "In which situation would you use..."
        ]

    def _build_mcq_prompt(self, topic: str, difficulty: str, question_num: int = 0) -> str:
        """Build the appropriate prompt based on difficulty level with variations."""
        
        # Add variety based on question number
        import random
        question_types = self._get_question_type_variations()
        question_formats = self._get_question_format_variations()
        
        # Cycle through different question types
        question_type = question_types[question_num % len(question_types)]
        question_format_hint = question_formats[question_num % len(question_formats)]
        
        # Vary the approach and context with more specific scenarios
        contexts = [
            "a cooking/recipe scenario (dividing ingredients, scaling recipes)",
            "a construction/building scenario (materials, measurements, planning)",
            "a sports scenario (scores, statistics, game situations)",
            "a shopping/budgeting scenario (prices, discounts, splitting costs)",
            "a science/experiment scenario (measurements, data analysis)",
            "a time/scheduling scenario (hours, days, planning)",
            "a sharing/distribution scenario (fair division, groups)",
            "a measurement/conversion scenario (units, scales, ratios)"
        ]
        context = contexts[question_num % len(contexts)]
        
        # Add explicit anti-repetition instructions
        anti_repetition = [
            "CRITICAL: DO NOT create a simple 'What is X divided by Y?' question. Instead, embed the division in a word problem or real scenario.",
            "CRITICAL: AVOID using common division pairs like 63÷7, 27÷3, 48÷6, 35÷7. Use different numbers and contexts.",
            "CRITICAL: DO NOT repeat similar question structures. Each question must have a UNIQUE format, context, and approach.",
            "CRITICAL: CREATE a word problem or scenario question, NOT a direct calculation question.",
            "CRITICAL: VARY the numbers significantly - avoid using similar dividend/divisor pairs in consecutive questions."
        ]
        anti_repetition_instruction = anti_repetition[question_num % len(anti_repetition)]
        
        # Add specific examples of what to avoid - more diverse examples
        bad_examples = [
            "BAD: 'What is 63 divided by 7?' → GOOD: 'Sarah has 63 cookies and wants to share them equally among 7 friends. How many cookies does each friend get?'",
            "BAD: 'Calculate 27 ÷ 3' → GOOD: 'A teacher needs to divide 27 students into 3 equal groups for a project. How many students will be in each group?'",
            "BAD: 'What is the quotient of 48 and 6?' → GOOD: 'A construction worker has 48 bricks and needs to stack them in piles of 6. How many piles can they make?'",
            "BAD: 'Find 35 ÷ 7' → GOOD: 'If a week has 7 days and you have 35 tasks to complete, how many tasks should you do per day to finish on time?'",
            "BAD: 'What is X divided by Y?' → GOOD: 'A recipe calls for 24 eggs to make 8 batches of cookies. How many eggs are needed per batch?'",
            "BAD: 'Calculate A ÷ B' → GOOD: 'A farmer has 90 apples and wants to pack them into boxes of 15. How many boxes will be filled?'",
            "BAD: 'Find the quotient' → GOOD: 'A marathon runner completes 42 kilometers in 6 hours. What is their average speed in km per hour?'",
            "BAD: 'What is X/Y?' → GOOD: 'A store receives 120 items and needs to display them on 8 shelves equally. How many items per shelf?'"
        ]
        example_instruction = bad_examples[question_num % len(bad_examples)]
        
        # Add random seed variation to prompt to force different outputs
        import random
        random_seed_hint = random.randint(1000, 9999)
        prompt_variation = f"\n\n🎲 Variation seed: {random_seed_hint} - Use this to create a UNIQUE question that hasn't been seen before."

        if difficulty == "easy":
            return (
                f"Create a beginner-level multiple-choice question about {topic}.\n\n"
                f"Question Type: Create {question_type}.\n"
                f"Question Format: Use a format like '{question_format_hint}' to start your question.\n"
                f"Context: Use {context}.\n\n"
                f"CRITICAL: {anti_repetition_instruction}\n\n"
                f"Example of what to AVOID vs what to DO:\n"
                f"{example_instruction}\n\n"
                f"{prompt_variation}\n\n"
                f"Requirements:\n"
                f"- MUST be a word problem or scenario question, NOT a direct calculation\n"
                f"- Test basic understanding of {topic} through application\n"
                f"- Use simple, clear language\n"
                f"- Include straightforward calculations embedded in a real scenario\n"
                f"- Provide 3 plausible but incorrect distractors\n"
                f"- Give a brief step-by-step explanation\n"
                f"- Use DIFFERENT numbers than common examples (avoid 63÷7, 27÷3, 48÷6, 35÷7, 42÷6, 56÷8, etc.)\n"
                f"- Use numbers that are NOT multiples of common division pairs\n"
                f"- Make this question COMPLETELY UNIQUE - no similar phrasing, numbers, or structure to typical questions\n"
                f"- If generating multiple questions, each MUST be significantly different from the others"
            )
        elif difficulty == "hard":
            return (
                f"Create an advanced multiple-choice question about {topic}.\n\n"
                f"Question Type: Create {question_type}.\n"
                f"Question Format: Use a format like '{question_format_hint}' to start your question.\n"
                f"Context: Use {context}.\n\n"
                f"CRITICAL: {anti_repetition_instruction}\n\n"
                f"Example of what to AVOID vs what to DO:\n"
                f"{example_instruction}\n\n"
                f"Requirements:\n"
                f"- MUST be a word problem or scenario question, NOT a direct calculation\n"
                f"- Test deep understanding of {topic} through complex real-world applications\n"
                f"- Require multi-step reasoning or complex analysis\n"
                f"- May involve combining multiple concepts or real-world scenarios\n"
                f"- Provide 3 challenging distractors (common mistakes or misconceptions)\n"
                f"- Give a detailed step-by-step explanation\n"
                f"- Use DIFFERENT numbers than common examples (avoid 63÷7, 27÷3, 48÷6, 35÷7, etc.)\n"
                f"- Make this question COMPLETELY UNIQUE and different from standard textbook questions\n\n"
                f"Be creative: Use novel scenarios, unexpected applications, or non-standard problem formulations."
            )
        else:  # medium
            return (
                f"Create an intermediate-level multiple-choice question about {topic}.\n\n"
                f"Question Type: Create {question_type}.\n"
                f"Question Format: Use a format like '{question_format_hint}' to start your question.\n"
                f"Context: Use {context}.\n\n"
                f"CRITICAL: {anti_repetition_instruction}\n\n"
                f"Example of what to AVOID vs what to DO:\n"
                f"{example_instruction}\n\n"
                f"{prompt_variation}\n\n"
                f"Requirements:\n"
                f"- MUST be a word problem or scenario question, NOT a direct calculation\n"
                f"- Test application of {topic} concepts through real-world situations\n"
                f"- Require understanding beyond memorization\n"
                f"- Include 2-3 step problems or moderate complexity\n"
                f"- Provide 3 plausible distractors\n"
                f"- Give a clear step-by-step explanation\n"
                f"- Use DIFFERENT numbers than common examples (avoid 63÷7, 27÷3, 48÷6, 35÷7, 42÷6, 56÷8, 72÷9, etc.)\n"
                f"- Use numbers that are NOT multiples of common division pairs\n"
                f"- Make this question COMPLETELY DISTINCTIVE - avoid generic or repetitive formats\n"
                f"- Each question must have a UNIQUE scenario, different numbers, and different structure\n"
                f"- If this is part of a set, ensure it's COMPLETELY DIFFERENT from any other questions in the set"
            )

    def _get_question_embedding(self, question_text: str) -> np.ndarray:
        """
        Get semantic embedding for a question.
        Uses sentence transformers if available, otherwise OpenAI embeddings.
        """
        if SEMANTIC_MODEL_AVAILABLE and semantic_model is not None:
            # Use sentence transformers (faster, free)
            embedding = semantic_model.encode(question_text, convert_to_numpy=True)
            return embedding
        else:
            # Fallback to OpenAI embeddings
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=question_text
                )
                embedding = np.array(response.data[0].embedding)
                return embedding
            except Exception as e:
                print(f"[WARNING] Could not get embedding: {e}")
                # Return zero vector as fallback
                return np.zeros(384)  # Default dimension
    
    def _check_semantic_similarity(self, question_text: str, topic: str, threshold: float = 0.75) -> Tuple[bool, float]:
        """
        Check semantic similarity using embeddings.
        Returns (is_too_similar, max_similarity_score)
        """
        try:
            new_embedding = self._get_question_embedding(question_text)
            new_embedding = new_embedding / np.linalg.norm(new_embedding)  # Normalize
            
            max_similarity = 0.0
            most_similar_q = None
            
            # Check against questions for this topic
            if topic in self.topic_embeddings and len(self.topic_embeddings[topic]) > 0:
                for prev_embedding, prev_question in self.topic_embeddings[topic][-15:]:
                    # Cosine similarity
                    similarity = np.dot(new_embedding, prev_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_q = prev_question
            
            # Also check against all recent questions
            for prev_embedding, prev_question in self.question_embeddings[-20:]:
                similarity = np.dot(new_embedding, prev_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_q = prev_question
            
            is_too_similar = max_similarity >= threshold
            
            if is_too_similar:
                print(f"[SEMANTIC CHECK] High similarity detected: {max_similarity:.3f}")
                if most_similar_q:
                    print(f"[SEMANTIC CHECK] Most similar to: {most_similar_q[:80]}...")
            
            return is_too_similar, max_similarity
            
        except Exception as e:
            print(f"[WARNING] Error in semantic similarity check: {e}")
            return False, 0.0
    
    def _check_similarity(self, question_text: str, topic: str) -> bool:
        """
        Check if a question is too similar to recent questions.
        Returns True if too similar, False otherwise.
        """
        import re
        question_lower = question_text.lower()
        numbers_in_new = set(re.findall(r'\b\d+\b', question_lower))
        
        # Check against recent questions for this topic
        if topic in self.recent_topics and len(self.recent_topics[topic]) > 0:
            for prev_q in self.recent_topics[topic][-10:]:
                prev_lower = prev_q.lower()
                numbers_in_prev = set(re.findall(r'\b\d+\b', prev_lower))
                
                # STRICT: If 2 or more numbers match, it's too similar
                matching_numbers = numbers_in_new & numbers_in_prev
                if len(matching_numbers) >= 2:
                    print(f"[SIMILARITY CHECK] {len(matching_numbers)} matching numbers: {matching_numbers}")
                    print(f"[SIMILARITY CHECK] New: {question_text[:80]}...")
                    print(f"[SIMILARITY CHECK] Prev: {prev_q[:80]}...")
                    return True
                
                # Check for very similar wording (75% word overlap)
                words_new = set(re.findall(r'\b\w+\b', question_lower))
                words_prev = set(re.findall(r'\b\w+\b', prev_lower))
                # Remove common stop words for better comparison
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                             'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'if'}
                words_new = words_new - stop_words
                words_prev = words_prev - stop_words
                
                if len(words_new) > 0 and len(words_prev) > 0:
                    overlap = len(words_new & words_prev) / max(len(words_new), len(words_prev))
                    if overlap > 0.75:  # 75% word overlap
                        print(f"[SIMILARITY CHECK] Too similar wording (overlap: {overlap:.2f})")
                        print(f"[SIMILARITY CHECK] New: {question_text[:80]}...")
                        print(f"[SIMILARITY CHECK] Prev: {prev_q[:80]}...")
                        return True
                
                # Check for same question structure (same key phrases)
                key_phrases_new = set(re.findall(r'\b\w+\s+\w+\b', question_lower))  # Bigrams
                key_phrases_prev = set(re.findall(r'\b\w+\s+\w+\b', prev_lower))
                if len(key_phrases_new & key_phrases_prev) >= 5:  # 5+ matching phrases
                    print(f"[SIMILARITY CHECK] Too many matching phrases")
                    return True
        
        # Also check against all recent questions (cross-topic)
        if len(self.recent_questions) > 0:
            for prev_q in self.recent_questions[-10:]:
                if prev_q == question_text:  # Exact duplicate
                    print(f"[SIMILARITY CHECK] Exact duplicate detected")
                    return True
                
                prev_lower = prev_q.lower()
                numbers_in_prev = set(re.findall(r'\b\d+\b', prev_lower))
                # If same topic and 2+ numbers match, reject
                if len(numbers_in_new & numbers_in_prev) >= 2:
                    # Check if it's the same topic by checking question content
                    if any(word in question_lower and word in prev_lower for word in topic.lower().split()):
                        print(f"[SIMILARITY CHECK] Cross-check: matching numbers in same topic context")
                        return True
        
        return False
    
    def _validate_and_extract_mcq(
        self, 
        mcq_data: Dict, 
        topic: str, 
        difficulty: str, 
        concept_id: int
    ) -> Optional[Dict]:
        """
        Validate and extract MCQ components, ensuring all required fields are present.
        
        Args:
            mcq_data: Raw dictionary from OpenAI response
            topic: The topic
            difficulty: The difficulty level
            concept_id: The concept ID
            
        Returns:
            Validated MCQ dictionary or None if validation fails
        """
        # Extract and validate all required fields
        question = mcq_data.get("question", "").strip()
        correct_answer = mcq_data.get("correct_answer", "").strip()
        distractor_1 = mcq_data.get("distractor_1", "").strip()
        distractor_2 = mcq_data.get("distractor_2", "").strip()
        distractor_3 = mcq_data.get("distractor_3", "").strip()
        explanation = mcq_data.get("explanation", "").strip()

        # Strict validation: all fields must be non-empty
        if not question:
            print("[VALIDATION ERROR] Question is empty")
            return None
        
        if not correct_answer:
            print("[VALIDATION ERROR] Correct answer is empty")
            return None
        
        if not distractor_1:
            print("[VALIDATION ERROR] Distractor 1 is empty")
            return None
        
        if not distractor_2:
            print("[VALIDATION ERROR] Distractor 2 is empty")
            return None
        
        if not distractor_3:
            print("[VALIDATION ERROR] Distractor 3 is empty")
            return None

        # Create options list with correct answer and distractors
        options = [correct_answer, distractor_1, distractor_2, distractor_3]
        
        # Validate that all options are unique (to avoid confusion)
        if len(set(options)) < 4:
            print("[VALIDATION ERROR] Options are not all unique")
            # Still proceed, but log warning
        
        # Randomize the order and track the correct index
        correct_index = self._randomize_options(options)
        
        # Final validation: ensure correct answer is in the options after randomization
        if correct_index < 0 or correct_index >= len(options):
            print(f"[VALIDATION ERROR] Invalid correct_index: {correct_index}")
            return None
        
        # Verify the correct answer is actually in the options
        if correct_answer not in options:
            print(f"[VALIDATION ERROR] Correct answer '{correct_answer}' not found in options after randomization")
            return None

        # Build the result dictionary
        result = {
            "question": question,
            "options": options,
            "correct_index": correct_index,
            "explanation": explanation if explanation else f"The correct answer is {correct_answer}.",
            "topic": topic,
            "difficulty": difficulty,
            "concept_id": concept_id,
        }
        
        return result

    def _randomize_options(self, options: List[str]) -> int:
        """
        Randomize the order of options and return the new index of the correct answer.
        Includes validation to ensure correct answer is preserved.

        Args:
            options: List where first element is the correct answer

        Returns:
            The new index (0-3) of the correct answer after shuffling
        """
        if not options or len(options) != 4:
            raise ValueError(f"Expected 4 options, got {len(options)}")
        
        # Store the correct answer
        correct_answer = options[0]
        
        # Validate correct answer is not empty
        if not correct_answer:
            raise ValueError("Correct answer cannot be empty")

        # Shuffle the options in place
        random.shuffle(options)

        # Find and return the new index of the correct answer
        try:
            correct_index = options.index(correct_answer)
        except ValueError:
            raise ValueError(f"Correct answer '{correct_answer}' not found in options after shuffling")
        
        # Final check
        if correct_index < 0 or correct_index >= len(options):
            raise ValueError(f"Invalid correct_index: {correct_index}")
        
        return correct_index

    def _generate_fallback_mcq(self, topic: str, difficulty: str, concept_id: int) -> Dict:
        """Generate a simple fallback MCQ if OpenAI call fails."""
        
        # Create a simple but valid MCQ
        question = f"Which statement is true about {topic}?"
        correct_answer = f"{topic} is a fundamental concept in mathematics"
        distractor_1 = f"{topic} is not used in real-world applications"
        distractor_2 = f"{topic} cannot be learned systematically"
        distractor_3 = f"{topic} has no practical value"
        
        options = [correct_answer, distractor_1, distractor_2, distractor_3]
        
        # Randomize and get correct index
        try:
            correct_index = self._randomize_options(options)
        except Exception as e:
            print(f"[FALLBACK ERROR] Error randomizing options: {e}")
            # If randomization fails, keep correct answer at index 0
            correct_index = 0

        result = {
            "question": question,
            "options": options,
            "correct_index": correct_index,
            "explanation": f"The correct answer is: '{correct_answer}'. This statement correctly describes {topic} as a fundamental mathematical concept.",
            "topic": topic,
            "difficulty": difficulty,
            "concept_id": concept_id,
        }
        
        # Validate fallback MCQ
        if result["options"][result["correct_index"]] != correct_answer:
            print(f"[FALLBACK WARNING] Correct answer mismatch in fallback MCQ")
            # Fix it
            result["correct_index"] = options.index(correct_answer)
        
        return result

    def generate_multiple_mcqs(
        self,
        topic: str,
        difficulty: str,
        n_questions: int,
        concept_id: int = 0,
    ) -> List[Dict]:
        """
        Generate multiple MCQs for a topic.

        Args:
            topic: The mathematical topic
            difficulty: Difficulty level
            n_questions: Number of questions to generate
            concept_id: Concept ID for tracking

        Returns:
            List of MCQ dictionaries
        """
        mcqs = []
        for i in range(n_questions):
            mcq = self.generate_mcq(topic, difficulty, concept_id)
            mcqs.append(mcq)

        return mcqs

    def evaluate_answer(
        self,
        mcq: Dict,
        selected_index: int,
    ) -> Tuple[bool, str]:
        """
        Evaluate a student's answer to an MCQ.
        Includes validation to ensure indices are valid.

        Args:
            mcq: The MCQ dictionary (from generate_mcq)
            selected_index: The index (0-3) selected by the student

        Returns:
            Tuple of (is_correct: bool, feedback: str)
        """
        # Validate inputs
        if "correct_index" not in mcq:
            raise ValueError("MCQ missing 'correct_index' field")
        if "options" not in mcq:
            raise ValueError("MCQ missing 'options' field")
        
        correct_index = mcq["correct_index"]
        options = mcq["options"]
        
        # Validate indices
        if correct_index < 0 or correct_index >= len(options):
            raise ValueError(f"Invalid correct_index: {correct_index} (options length: {len(options)})")
        if selected_index < 0 or selected_index >= len(options):
            raise ValueError(f"Invalid selected_index: {selected_index} (options length: {len(options)})")
        
        is_correct = (selected_index == correct_index)
        
        # Get the selected and correct options
        selected_option = options[selected_index]
        correct_option = options[correct_index]
        
        # Validate that correct option exists
        if not correct_option:
            raise ValueError("Correct option is empty - MCQ validation failed")

        if is_correct:
            feedback = (
                f"✅ **Correct!**\n\n"
                f"**Your Answer:** {selected_option}\n\n"
                f"**Explanation:** {mcq.get('explanation', 'Well done!')}"
            )
        else:
            feedback = (
                f"❌ **Incorrect.**\n\n"
                f"**Your Answer:** {selected_option}\n\n"
                f"**Correct Answer:** {correct_option}\n\n"
                f"**Explanation:** {mcq.get('explanation', 'Review the concept and try again.')}"
            )

        return is_correct, feedback

    def get_statistics(self) -> Dict:
        """Get statistics about generated MCQs."""
        if not self.generation_history:
            return {"total_generated": 0}

        difficulty_counts = {}
        topic_counts = {}

        for mcq in self.generation_history:
            diff = mcq.get("difficulty", "unknown")
            topic = mcq.get("topic", "unknown")

            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return {
            "total_generated": len(self.generation_history),
            "by_difficulty": difficulty_counts,
            "by_topic": topic_counts,
        }


# ============================================================================
# Standalone Testing & Demo
# ============================================================================

def demo_mcq_generator():
    """Demonstrate the MCQ generator functionality."""

    print("=" * 80)
    print("MCQ Generator Demo")
    print("=" * 80)

    # Initialize generator
    generator = MCQGenerator(seed=42)

    # Test topics and difficulties
    test_cases = [
        ("quadratic equations", "easy"),
        ("derivatives", "medium"),
        ("integration by parts", "hard"),
    ]

    for topic, difficulty in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Topic: {topic} | Difficulty: {difficulty}")
        print("=" * 80)

        # Generate MCQ
        mcq = generator.generate_mcq(topic=topic, difficulty=difficulty, concept_id=1)

        # Display the question
        print(f"\n**Question:** {mcq['question']}\n")
        print("**Options:**")
        for i, option in enumerate(mcq['options']):
            print(f"  {chr(65+i)}. {option}")

        print(f"\n**Correct Answer:** Option {chr(65 + mcq['correct_index'])}")
        print(f"**Explanation:** {mcq['explanation']}")

        # Simulate student answer
        student_choice = random.randint(0, 3)
        is_correct, feedback = generator.evaluate_answer(mcq, student_choice)

        print(f"\n**Student selected:** Option {chr(65 + student_choice)}")
        print(f"**Result:** {'Correct' if is_correct else 'Incorrect'}")
        print(f"\n{feedback}")

    # Show statistics
    print("\n" + "=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    stats = generator.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    # Run demo
    demo_mcq_generator()
