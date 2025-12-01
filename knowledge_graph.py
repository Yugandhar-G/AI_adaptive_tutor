"""
Build Knowledge Graph from Khan Academy Titles with REAL NLP

This script uses genuine NLP techniques:
1. Sentence embeddings for semantic similarity
2. Named Entity Recognition for mathematical concepts
3. Dependency parsing for relationship extraction
4. TF-IDF for keyword extraction
5. Cosine similarity for finding related concepts

Install requirements:
pip install sentence-transformers spacy scikit-learn
python -m spacy download en_core_web_sm
"""

import os
import pickle
import networkx as nx
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# NLP imports
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Configuration
KHAN_DIR = 'khan-exercises/exercises'
OUTPUT_GRAPH = 'khan_knowledge_graph_nlp.pkl'
OUTPUT_EMBEDDINGS = 'khan_embeddings_nlp.pkl'

# Mathematical concept hierarchy (for validation/augmentation)
MATH_HIERARCHY = {
    'counting': (1, ['count', 'number'], []),
    'addition': (2, ['add', 'sum', 'plus'], ['counting']),
    'subtraction': (2, ['subtract', 'minus', 'difference'], ['addition']),
    'multiplication': (3, ['multiply', 'times', 'product'], ['addition']),
    'division': (3, ['divide', 'quotient'], ['multiplication']),
    'fractions': (4, ['fraction', 'numerator', 'denominator'], ['division']),
    'decimals': (4, ['decimal', 'point'], ['fractions']),
    'percentages': (5, ['percent', '%'], ['decimals', 'fractions']),
    'integers': (4, ['integer', 'negative', 'positive'], ['subtraction']),
    'exponents': (5, ['exponent', 'power', 'square', 'cube'], ['multiplication']),
    'roots': (6, ['root', 'sqrt', 'radical'], ['exponents']),
    'variables': (5, ['variable', 'unknown', 'x', 'y'], ['integers']),
    'expressions': (6, ['expression', 'term', 'coefficient'], ['variables']),
    'equations': (6, ['equation', 'solve', 'equal'], ['expressions']),
    'inequalities': (7, ['inequality', 'greater', 'less'], ['equations']),
    'linear': (7, ['linear', 'slope', 'line'], ['equations']),
    'quadratic': (8, ['quadratic', 'parabola', 'square'], ['linear']),
    'polynomial': (8, ['polynomial', 'degree'], ['quadratic']),
    'rational': (8, ['rational', 'ratio'], ['polynomial', 'fractions']),
    'functions': (7, ['function', 'f(x)', 'domain', 'range'], ['expressions']),
    'graphing': (7, ['graph', 'plot', 'coordinate'], ['functions']),
    'geometry': (5, ['angle', 'shape', 'triangle', 'circle'], []),
    'trigonometry': (9, ['sin', 'cos', 'tan', 'trig'], ['geometry', 'functions']),
    'calculus': (10, ['derivative', 'integral', 'limit'], ['functions', 'trigonometry']),
}


class NLPKnowledgeGraphBuilder:
    """Build knowledge graph using real NLP techniques."""
    
    def __init__(self, khan_dir):
        self.khan_dir = khan_dir
        self.concepts = []
        self.graph = nx.DiGraph()
        
        # Initialize NLP models
        print("🤖 Initializing NLP models...")
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("  ✓ Loaded sentence embedding model")
        except:
            print("  ⚠️  Sentence transformers not available")
            self.sentence_model = None
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("  ✓ Loaded spaCy model")
        except:
            print("  ⚠️  spaCy not available. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        print("  ✓ Initialized TF-IDF vectorizer")
        
        self.title_embeddings = {}
        self.title_tfidf = None
        
    def extract_all_titles(self):
        """Extract all exercise titles from Khan Academy."""
        print("\n" + "="*80)
        print("STEP 1: Extracting Titles from Khan Academy")
        print("="*80)
        
        if not os.path.exists(self.khan_dir):
            print(f"❌ Khan Academy directory not found: {self.khan_dir}")
            return
        
        concepts = []
        
        for root, dirs, files in tqdm(list(os.walk(self.khan_dir))):
            for fname in files:
                if fname.endswith('.html'):
                    filepath = os.path.join(root, fname)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            soup = BeautifulSoup(f.read(), 'lxml')
                        
                        # Extract title
                        if soup.title and soup.title.string:
                            title = soup.title.string.strip()
                        else:
                            title = fname.replace('.html', '').replace('_', ' ')
                        
                        # Get category from directory
                        rel_path = os.path.relpath(root, self.khan_dir)
                        category = rel_path.split(os.sep)[0] if rel_path != '.' else 'general'
                        
                        concepts.append({
                            'title': title,
                            'category': category,
                            'file': fname,
                            'path': filepath
                        })
                    except Exception as e:
                        continue
        
        self.concepts = concepts
        print(f"\n✓ Extracted {len(concepts)} concepts")
        
        # Show sample
        print("\nSample concepts:")
        for concept in concepts[:5]:
            print(f"  - {concept['title']} (category: {concept['category']})")
    
    def compute_nlp_features(self):
        """Compute NLP features for all concepts."""
        print("\n" + "="*80)
        print("STEP 2: Computing NLP Features")
        print("="*80)
        
        titles = [c['title'] for c in self.concepts]
        
        # 1. Sentence Embeddings (semantic similarity)
        if self.sentence_model:
            print("\n1️⃣ Computing sentence embeddings...")
            embeddings = self.sentence_model.encode(titles, show_progress_bar=True)
            for i, concept in enumerate(self.concepts):
                self.title_embeddings[concept['title']] = embeddings[i]
            print(f"✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        # 2. TF-IDF (keyword importance)
        print("\n2️⃣ Computing TF-IDF features...")
        self.title_tfidf = self.tfidf_vectorizer.fit_transform(titles)
        print(f"✓ Generated TF-IDF matrix: {self.title_tfidf.shape}")
        
        # Show top keywords
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"  Top keywords: {', '.join(feature_names[:20])}")
        
        # 3. spaCy linguistic analysis
        if self.nlp:
            print("\n3️⃣ Analyzing linguistic features with spaCy...")
            for concept in tqdm(self.concepts):
                doc = self.nlp(concept['title'])
                
                # Extract key information
                concept['tokens'] = [token.text for token in doc]
                concept['lemmas'] = [token.lemma_ for token in doc]
                concept['pos_tags'] = [token.pos_ for token in doc]
                concept['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
                
                # Extract main concepts (nouns and proper nouns)
                concept['main_concepts'] = [token.text.lower() for token in doc 
                                           if token.pos_ in ['NOUN', 'PROPN']]
            
            print(f"✓ Analyzed linguistic features for {len(self.concepts)} concepts")
    
    def infer_concept_complexity(self, concept):
        """
        Use NLP to infer complexity level.
        
        Strategies:
        1. Check domain knowledge
        2. Count mathematical terms
        3. Analyze linguistic complexity
        """
        title = concept['title'].lower()
        
        # Strategy 1: Domain knowledge
        for concept_type, (level, keywords, _) in MATH_HIERARCHY.items():
            for keyword in keywords:
                if keyword in title:
                    return level
        
        # Strategy 2: Linguistic complexity
        if self.nlp and 'tokens' in concept:
            # More tokens = potentially more complex
            token_count = len(concept['tokens'])
            
            # Check for complexity indicators
            complexity_markers = {
                'basic': -2, 'simple': -2, 'intro': -2, 'elementary': -1,
                'advanced': 3, 'complex': 3, 'multi': 2, 'compound': 2,
                'apply': 1, 'solve': 1, 'analyze': 2, 'evaluate': 2
            }
            
            complexity_boost = sum(complexity_markers.get(token.lower(), 0) 
                                  for token in concept['tokens'])
            
            base_complexity = min(token_count // 2, 5)
            return max(1, min(10, base_complexity + complexity_boost))
        
        # Default
        return 5
    
    def find_semantic_prerequisites(self, concept, similarity_threshold=0.7):
        """
        Use semantic similarity to find prerequisite concepts.
        
        Logic: If concept A is semantically similar to B but simpler,
        A is likely a prerequisite for B.
        """
        if not self.sentence_model or concept['title'] not in self.title_embeddings:
            return []
        
        prereqs = []
        current_embedding = self.title_embeddings[concept['title']]
        current_complexity = self.infer_concept_complexity(concept)
        
        for other_concept in self.concepts:
            if other_concept['title'] == concept['title']:
                continue
            
            other_embedding = self.title_embeddings[other_concept['title']]
            other_complexity = self.infer_concept_complexity(other_concept)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                current_embedding.reshape(1, -1),
                other_embedding.reshape(1, -1)
            )[0][0]
            
            # If similar and simpler, it's a prerequisite
            if similarity > similarity_threshold and other_complexity < current_complexity:
                prereqs.append({
                    'title': other_concept['title'],
                    'similarity': float(similarity),
                    'complexity_gap': current_complexity - other_complexity
                })
        
        # Sort by similarity
        prereqs.sort(key=lambda x: x['similarity'], reverse=True)
        return prereqs[:5]  # Top 5 most similar prerequisites
    
    def find_concept_overlap(self, concept):
        """
        Use NLP to find concepts that share main mathematical terms.
        
        If concept A's main terms appear in concept B, A might be a prerequisite.
        """
        if 'main_concepts' not in concept:
            return []
        
        prereqs = []
        current_concepts = set(concept['main_concepts'])
        current_complexity = self.infer_concept_complexity(concept)
        
        for other_concept in self.concepts:
            if other_concept['title'] == concept['title']:
                continue
            
            if 'main_concepts' not in other_concept:
                continue
            
            other_concepts = set(other_concept['main_concepts'])
            other_complexity = self.infer_concept_complexity(other_concept)
            
            # Check for concept overlap
            overlap = current_concepts & other_concepts
            
            if overlap and other_complexity < current_complexity:
                # Other concept is simpler and shares terms
                prereqs.append({
                    'title': other_concept['title'],
                    'overlap': list(overlap),
                    'overlap_score': len(overlap) / len(current_concepts)
                })
        
        prereqs.sort(key=lambda x: x['overlap_score'], reverse=True)
        return prereqs[:3]
    
    def find_linguistic_prerequisites(self, concept):
        """
        Use linguistic analysis to infer prerequisites.
        
        Examples:
        - "Adding fractions" → prerequisite: concepts with "fraction"
        - "Advanced algebra" → prerequisite: "algebra"
        """
        prereqs = []
        title = concept['title']
        title_lower = title.lower()
        
        # Pattern 1: "Verb + Noun" → "Noun" is prerequisite
        # "Adding fractions" → "fractions"
        if self.nlp and 'pos_tags' in concept:
            doc = self.nlp(title)
            for i, token in enumerate(doc):
                if token.pos_ == 'VERB' and i + 1 < len(doc):
                    noun_phrase = doc[i+1:].text
                    prereqs.append(noun_phrase)
        
        # Pattern 2: "Advanced X" → "X" is prerequisite
        if 'advanced' in title_lower:
            basic_version = title_lower.replace('advanced', '').strip()
            prereqs.append(basic_version)
        
        # Pattern 3: "X and Y" → "X" is prerequisite for the whole
        if ' and ' in title_lower:
            parts = title_lower.split(' and ')
            if len(parts) == 2:
                prereqs.append(parts[0].strip())
        
        return prereqs
    
    def build_graph(self):
        """Build the knowledge graph with NLP-inferred relationships."""
        print("\n" + "="*80)
        print("STEP 3: Building Knowledge Graph with NLP")
        print("="*80)
        
        # Add all concepts as nodes
        print("\nAdding nodes...")
        for concept in self.concepts:
            complexity = self.infer_concept_complexity(concept)
            
            self.graph.add_node(
                concept['title'],
                type='concept',
                category=concept['category'],
                complexity=complexity,
                main_concepts=concept.get('main_concepts', [])
            )
        
        print(f"✓ Added {self.graph.number_of_nodes()} nodes")
        
        # Strategy 1: Semantic similarity prerequisites
        print("\n1️⃣ Adding edges from semantic similarity...")
        semantic_edges = 0
        for concept in tqdm(self.concepts):
            prereqs = self.find_semantic_prerequisites(concept, similarity_threshold=0.7)
            for prereq in prereqs:
                if prereq['title'] in self.graph.nodes():
                    self.graph.add_edge(
                        prereq['title'],
                        concept['title'],
                        relation='prerequisite',
                        method='semantic_similarity',
                        confidence=prereq['similarity']
                    )
                    semantic_edges += 1
        print(f"✓ Added {semantic_edges} edges from semantic similarity")
        
        # Strategy 2: Concept overlap
        print("\n2️⃣ Adding edges from concept overlap...")
        overlap_edges = 0
        for concept in tqdm(self.concepts):
            prereqs = self.find_concept_overlap(concept)
            for prereq in prereqs:
                if prereq['title'] in self.graph.nodes():
                    if not self.graph.has_edge(prereq['title'], concept['title']):
                        self.graph.add_edge(
                            prereq['title'],
                            concept['title'],
                            relation='prerequisite',
                            method='concept_overlap',
                            confidence=prereq['overlap_score']
                        )
                        overlap_edges += 1
        print(f"✓ Added {overlap_edges} edges from concept overlap")
        
        # Strategy 3: Linguistic patterns
        print("\n3️⃣ Adding edges from linguistic patterns...")
        linguistic_edges = 0
        for concept in tqdm(self.concepts):
            prereq_patterns = self.find_linguistic_prerequisites(concept)
            
            for pattern in prereq_patterns:
                # Find matching concepts
                matches = [c for c in self.concepts 
                          if pattern.lower() in c['title'].lower()]
                
                for match in matches:
                    if match['title'] != concept['title'] and match['title'] in self.graph.nodes():
                        if not self.graph.has_edge(match['title'], concept['title']):
                            self.graph.add_edge(
                                match['title'],
                                concept['title'],
                                relation='prerequisite',
                                method='linguistic_pattern',
                                confidence=0.8
                            )
                            linguistic_edges += 1
        print(f"✓ Added {linguistic_edges} edges from linguistic patterns")
        
        # Summary
        print("\n" + "="*80)
        print("Graph Statistics:")
        print("="*80)
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        print(f"  - Semantic similarity: {semantic_edges}")
        print(f"  - Concept overlap: {overlap_edges}")
        print(f"  - Linguistic patterns: {linguistic_edges}")
        print(f"Graph density: {nx.density(self.graph):.6f}")
        
        # Check connectivity
        if self.graph.number_of_edges() > 0:
            wcc = list(nx.weakly_connected_components(self.graph))
            print(f"Connected components: {len(wcc)}")
            if wcc:
                largest = max(wcc, key=len)
                print(f"Largest component: {len(largest)} nodes ({len(largest)/self.graph.number_of_nodes()*100:.1f}%)")
    
    def save_graph(self, output_path):
        """Save the knowledge graph."""
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"\n✓ Saved graph to {output_path}")
    
    def save_embeddings(self, output_path):
        """Save embeddings."""
        with open(output_path, 'wb') as f:
            pickle.dump(self.title_embeddings, f)
        print(f"✓ Saved embeddings to {output_path}")
    
    def visualize_sample(self, num_nodes=50):
        """Visualize a sample of the graph."""
        try:
            import matplotlib.pyplot as plt
            
            # Get most connected nodes
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:num_nodes]
            top_node_ids = [node for node, _ in top_nodes]
            
            subgraph = self.graph.subgraph(top_node_ids)
            
            # Create layout
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
            
            # Draw
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Color by complexity
            colors = [self.graph.nodes[node]['complexity'] for node in subgraph.nodes()]
            
            # Color edges by method
            edge_colors = []
            for u, v, data in subgraph.edges(data=True):
                method = data.get('method', 'unknown')
                if method == 'semantic_similarity':
                    edge_colors.append('blue')
                elif method == 'concept_overlap':
                    edge_colors.append('green')
                elif method == 'linguistic_pattern':
                    edge_colors.append('orange')
                else:
                    edge_colors.append('gray')
            
            # Draw nodes
            nodes = nx.draw_networkx_nodes(subgraph, pos, node_color=colors, 
                                  cmap='RdYlGn_r', node_size=200, alpha=0.8, ax=ax)
            
            # Draw edges with colors
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3, arrows=True, 
                                  arrowsize=10, edge_color=edge_colors, ax=ax)
            
            # Add labels for high-degree nodes
            labels = {}
            for node, degree in top_nodes[:20]:
                if node in subgraph.nodes():
                    labels[node] = node[:20]
            
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
            
            ax.set_title("Khan Academy Knowledge Graph (NLP-Enhanced)", fontsize=14)
            
            # Add colorbar
            plt.colorbar(nodes, ax=ax, label='Complexity')
            
            # Add legend for edge colors
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, label='Semantic Similarity'),
                Line2D([0], [0], color='green', lw=2, label='Concept Overlap'),
                Line2D([0], [0], color='orange', lw=2, label='Linguistic Pattern')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            ax.axis('off')
            plt.tight_layout()
            plt.savefig('khan_kg_nlp_visualization.png', dpi=150, bbox_inches='tight')
            print("✓ Saved visualization to khan_kg_nlp_visualization.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")


def main():
    """Main function."""
    print("="*80)
    print("BUILDING KNOWLEDGE GRAPH WITH REAL NLP")
    print("="*80)
    
    # Initialize builder
    builder = NLPKnowledgeGraphBuilder(KHAN_DIR)
    
    # Extract titles
    builder.extract_all_titles()
    
    if not builder.concepts:
        print("\n❌ No concepts extracted. Check Khan Academy path:")
        print(f"   Looking for: {KHAN_DIR}")
        return
    
    # Compute NLP features
    builder.compute_nlp_features()
    
    # Build graph
    builder.build_graph()
    
    # Save
    builder.save_graph(OUTPUT_GRAPH)
    builder.save_embeddings(OUTPUT_EMBEDDINGS)
    
    # Visualize
    print("\n" + "="*80)
    print("STEP 4: Creating Visualization")
    print("="*80)
    builder.visualize_sample()
    
    # Show sample relationships
    print("\n" + "="*80)
    print("Sample NLP-Inferred Relationships:")
    print("="*80)
    
    for u, v, data in list(builder.graph.edges(data=True))[:15]:
        method = data.get('method', 'unknown')
        confidence = data.get('confidence', 0)
        print(f"  {u} → {v}")
        print(f"    Method: {method}, Confidence: {confidence:.3f}")
    
    print("\n" + "="*80)
    print("✓ NLP-ENHANCED KNOWLEDGE GRAPH COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()