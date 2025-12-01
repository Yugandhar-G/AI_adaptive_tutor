#!/usr/bin/env python3
"""
Comprehensive Demo: Adaptive Content Generator Capabilities

This script demonstrates ALL the features and concepts that the
Adaptive Content Generator can handle.
"""

import subprocess
import json
import os

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title):
    """Print a section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")

def print_subsection(title):
    """Print a subsection header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'-'*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'-'*80}{Colors.END}\n")

def run_command(cmd, description):
    """Run a command and display the output"""
    print(f"{Colors.GREEN}▶ {description}{Colors.END}")
    print(f"{Colors.YELLOW}Command: {cmd}{Colors.END}\n")
    
    os.system(cmd)
    print()

def main():
    """Main demo function"""
    
    print_section("🎓 ADAPTIVE CONTENT GENERATOR - COMPLETE CAPABILITIES DEMO 🎓")
    
    # ========================================================================
    # PART 1: CONTENT TYPES
    # ========================================================================
    print_section("📚 PART 1: CONTENT TYPES")
    
    content_types = [
        ("explanation", "Generates detailed explanations of mathematical concepts"),
        ("question", "Creates practice questions with solutions"),
        ("example", "Provides real-world examples and applications")
    ]
    
    print(f"{Colors.BOLD}The system can generate 3 types of educational content:{Colors.END}\n")
    for ct, desc in content_types:
        print(f"  {Colors.GREEN}✓{Colors.END} {Colors.BOLD}{ct.upper()}{Colors.END}: {desc}")
    
    print_subsection("Demo: Explanation")
    run_command(
        'python adaptive_content_generator.py --topic "quadratic equations" --content_type explanation --difficulty medium',
        "Generating an EXPLANATION about quadratic equations"
    )
    
    print_subsection("Demo: Question")
    run_command(
        'python adaptive_content_generator.py --topic "derivatives" --content_type question --difficulty medium',
        "Generating a QUESTION about derivatives"
    )
    
    print_subsection("Demo: Example")
    run_command(
        'python adaptive_content_generator.py --topic "vectors" --content_type example --difficulty medium',
        "Generating an EXAMPLE about vectors"
    )
    
    # ========================================================================
    # PART 2: DIFFICULTY LEVELS
    # ========================================================================
    print_section("📊 PART 2: ADAPTIVE DIFFICULTY LEVELS")
    
    difficulties = [
        ("easy", "For beginners - simple definitions, basic examples, clear terminology"),
        ("medium", "For intermediate learners - detailed explanations, multi-step problems"),
        ("hard", "For advanced students - formal proofs, complex applications, deep insights")
    ]
    
    print(f"{Colors.BOLD}The system adapts content across 3 difficulty levels:{Colors.END}\n")
    for diff, desc in difficulties:
        print(f"  {Colors.GREEN}✓{Colors.END} {Colors.BOLD}{diff.upper()}{Colors.END}: {desc}")
    
    topic = "Pythagorean theorem"
    
    print_subsection(f"Demo: {topic} at EASY Level")
    run_command(
        f'python adaptive_content_generator.py --topic "{topic}" --content_type explanation --difficulty easy',
        f"Generating EASY explanation of {topic}"
    )
    
    print_subsection(f"Demo: {topic} at MEDIUM Level")
    run_command(
        f'python adaptive_content_generator.py --topic "{topic}" --content_type explanation --difficulty medium',
        f"Generating MEDIUM explanation of {topic}"
    )
    
    print_subsection(f"Demo: {topic} at HARD Level")
    run_command(
        f'python adaptive_content_generator.py --topic "{topic}" --content_type explanation --difficulty hard',
        f"Generating HARD explanation of {topic}"
    )
    
    # ========================================================================
    # PART 3: MATHEMATICAL TOPICS COVERED
    # ========================================================================
    print_section("🔢 PART 3: MATHEMATICAL TOPICS")
    
    topics = [
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
        "trigonometry",
        "probability",
        "statistics",
        "geometry",
        "algebra",
        "calculus",
        "complex numbers",
        "sequences and series",
        "limits",
        "graph theory"
    ]
    
    print(f"{Colors.BOLD}The system can generate content for ANY mathematical topic!{Colors.END}\n")
    print(f"{Colors.BOLD}Here are some examples:{Colors.END}\n")
    
    for i, topic in enumerate(topics, 1):
        print(f"  {i:2d}. {topic}")
    
    print(f"\n{Colors.CYAN}Note: You can use ANY mathematical topic, not just these examples!{Colors.END}")
    
    # Demo a few different topics
    sample_topics = ["linear equations", "probability", "trigonometry"]
    
    for topic in sample_topics:
        print_subsection(f"Demo: {topic}")
        run_command(
            f'python adaptive_content_generator.py --topic "{topic}" --content_type explanation --difficulty medium',
            f"Generating content for {topic}"
        )
    
    # ========================================================================
    # PART 4: CONTENT VARIATIONS
    # ========================================================================
    print_section("🔄 PART 4: CONTENT VARIATIONS")
    
    print(f"{Colors.BOLD}The system can generate multiple variations of the same content!{Colors.END}\n")
    print(f"This is useful for:")
    print(f"  {Colors.GREEN}✓{Colors.END} Providing different explanations for diverse learning styles")
    print(f"  {Colors.GREEN}✓{Colors.END} Creating practice sets with different questions")
    print(f"  {Colors.GREEN}✓{Colors.END} Offering alternative examples and approaches")
    
    print_subsection("Demo: Generating Variations")
    run_command(
        'python adaptive_content_generator.py --topic "fractions" --content_type explanation --difficulty easy --generate_variations --output_file variations_demo.json',
        "Generating VARIATIONS of content about fractions"
    )
    
    # ========================================================================
    # PART 5: STUDENT INTERACTION SIMULATION
    # ========================================================================
    print_section("👤 PART 5: ADAPTIVE STUDENT INTERACTION")
    
    print(f"{Colors.BOLD}The system can simulate adaptive learning sessions!{Colors.END}\n")
    print(f"Features:")
    print(f"  {Colors.GREEN}✓{Colors.END} Tracks student performance over time")
    print(f"  {Colors.GREEN}✓{Colors.END} Adjusts difficulty based on correctness")
    print(f"  {Colors.GREEN}✓{Colors.END} Generates personalized learning paths")
    print(f"  {Colors.GREEN}✓{Colors.END} Records interaction history")
    
    print_subsection("Demo: Simulating 5 Student Interactions")
    run_command(
        'python adaptive_content_generator.py --simulate --num_interactions 5 --output_file simulation_demo.json',
        "Running adaptive learning simulation"
    )
    
    # ========================================================================
    # PART 6: OUTPUT OPTIONS
    # ========================================================================
    print_section("💾 PART 6: OUTPUT & LOGGING")
    
    print(f"{Colors.BOLD}The system provides comprehensive output options:{Colors.END}\n")
    print(f"  {Colors.GREEN}✓{Colors.END} Console output for immediate viewing")
    print(f"  {Colors.GREEN}✓{Colors.END} JSON file export for data analysis")
    print(f"  {Colors.GREEN}✓{Colors.END} Automatic logging to generation_log.txt")
    print(f"  {Colors.GREEN}✓{Colors.END} Structured data for integration with other systems")
    
    print_subsection("Demo: Saving Output to File")
    run_command(
        'python adaptive_content_generator.py --topic "logarithms" --content_type question --difficulty hard --output_file logarithms_question.json',
        "Generating and saving content to JSON file"
    )
    
    # Display the saved file
    print(f"\n{Colors.BLUE}Content of logarithms_question.json:{Colors.END}\n")
    try:
        with open('logarithms_question.json', 'r') as f:
            content = json.load(f)
            print(json.dumps(content, indent=2))
    except:
        print("(File not found - generate it by running the command above)")
    
    # ========================================================================
    # PART 7: ADVANCED FEATURES
    # ========================================================================
    print_section("🚀 PART 7: ADVANCED FEATURES")
    
    features = [
        ("DKT Model Integration", "Deep Knowledge Tracing to predict student performance"),
        ("Knowledge Graph Embeddings", "Uses concept relationships to enhance content generation"),
        ("Personalization", "Adapts to individual student learning patterns"),
        ("Multi-modal Learning", "Combines explanations, questions, and examples"),
        ("Progressive Difficulty", "Gradually increases challenge as student improves"),
        ("Real-time Adaptation", "Adjusts content based on immediate feedback"),
        ("OpenAI GPT-3.5 Turbo", "Leverages state-of-the-art language model for content quality")
    ]
    
    print(f"{Colors.BOLD}Advanced AI-powered features:{Colors.END}\n")
    for feature, desc in features:
        print(f"  {Colors.GREEN}✓{Colors.END} {Colors.BOLD}{feature}{Colors.END}: {desc}")
    
    # ========================================================================
    # PART 8: USE CASES
    # ========================================================================
    print_section("💡 PART 8: PRACTICAL USE CASES")
    
    use_cases = [
        ("Intelligent Tutoring Systems", "Provide personalized learning experiences"),
        ("Homework Help Platforms", "Generate custom practice problems and explanations"),
        ("Educational Games", "Create dynamic content that adapts to player skill"),
        ("MOOC Platforms", "Offer differentiated content for diverse learners"),
        ("Test Preparation", "Generate practice tests at appropriate difficulty levels"),
        ("Special Education", "Adapt content for students with different learning needs"),
        ("Flipped Classrooms", "Create pre-class materials tailored to student readiness"),
        ("Research in Education", "Study adaptive learning and content generation")
    ]
    
    print(f"{Colors.BOLD}Real-world applications:{Colors.END}\n")
    for i, (use_case, desc) in enumerate(use_cases, 1):
        print(f"  {i}. {Colors.BOLD}{use_case}{Colors.END}")
        print(f"     {desc}\n")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_section("✨ SUMMARY: WHAT CAN THIS SYSTEM DO?")
    
    summary = f"""
{Colors.BOLD}The Adaptive Content Generator is a comprehensive educational AI system that:{Colors.END}

{Colors.GREEN}✓{Colors.END} Generates 3 types of content: explanations, questions, and examples
{Colors.GREEN}✓{Colors.END} Adapts across 3 difficulty levels: easy, medium, and hard
{Colors.GREEN}✓{Colors.END} Covers unlimited mathematical topics (not restricted to a predefined list)
{Colors.GREEN}✓{Colors.END} Creates content variations for diverse learning styles
{Colors.GREEN}✓{Colors.END} Simulates adaptive learning sessions with student tracking
{Colors.GREEN}✓{Colors.END} Integrates with Deep Knowledge Tracing (DKT) models
{Colors.GREEN}✓{Colors.END} Uses Knowledge Graph embeddings for concept understanding
{Colors.GREEN}✓{Colors.END} Exports data in JSON format for further analysis
{Colors.GREEN}✓{Colors.END} Maintains comprehensive logs of all generated content
{Colors.GREEN}✓{Colors.END} Powered by OpenAI GPT-3.5 Turbo for high-quality output

{Colors.CYAN}{Colors.BOLD}Key Innovation:{Colors.END}
Unlike static educational systems, this generator DYNAMICALLY ADAPTS to student
performance, ensuring each learner receives content at their optimal challenge level.

{Colors.CYAN}{Colors.BOLD}Getting Started:{Colors.END}
Try generating content with different combinations of topics, types, and difficulties
to explore the full capabilities of this system!
"""
    
    print(summary)
    
    # ========================================================================
    # QUICK REFERENCE
    # ========================================================================
    print_section("📖 QUICK REFERENCE: COMMAND LINE OPTIONS")
    
    commands = f"""
{Colors.BOLD}Basic Content Generation:{Colors.END}
  python adaptive_content_generator.py --topic "TOPIC" --content_type TYPE --difficulty LEVEL

{Colors.BOLD}Generate with Variations:{Colors.END}
  python adaptive_content_generator.py --topic "TOPIC" --generate_variations

{Colors.BOLD}Simulate Student Learning:{Colors.END}
  python adaptive_content_generator.py --simulate --num_interactions N

{Colors.BOLD}Save Output to File:{Colors.END}
  python adaptive_content_generator.py --topic "TOPIC" --output_file output.json

{Colors.BOLD}With DKT Model and Embeddings:{Colors.END}
  python adaptive_content_generator.py --dkt_model_path PATH --embeddings_path PATH

{Colors.BOLD}Available Options:{Colors.END}
  --topic              : Any mathematical topic (e.g., "calculus", "algebra")
  --content_type       : explanation | question | example
  --difficulty         : easy | medium | hard
  --generate_variations: Create multiple versions of content
  --simulate           : Run adaptive learning simulation
  --num_interactions   : Number of interactions in simulation (default: 5)
  --output_file        : Save results to JSON file
  --dkt_model_path     : Path to DKT model for predictions
  --embeddings_path    : Path to Knowledge Graph embeddings

{Colors.CYAN}💡 Tip: Try mixing and matching options to explore different combinations!{Colors.END}
"""
    
    print(commands)
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}Demo Complete! Try the commands above to explore further. 🎉{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")

if __name__ == "__main__":
    main()