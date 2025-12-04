# 📚 Adaptive Learning System

An intelligent, AI-powered adaptive learning platform that personalizes educational content based on student performance and knowledge state. The system uses knowledge graphs, deep learning models, and OpenAI GPT to generate dynamic explanations, examples, and questions tailored to each learner.

## ✨ Features

### 🎯 Core Capabilities
- **Adaptive Content Generation**: Automatically generates explanations, examples, and multiple-choice questions
- **Knowledge Graph Integration**: Uses a comprehensive knowledge graph with 441+ mathematical concepts
- **Real-time Mastery Tracking**: Tracks student performance and updates mastery levels dynamically
- **Difficulty Adaptation**: Adjusts content difficulty (Easy, Medium, Hard) based on student knowledge state
- **Anti-Repetition System**: Ensures diverse, unique questions to prevent repetition
- **Modern UI**: Clean, intuitive Streamlit interface with beautiful visualizations

### 📊 Learning Analytics
- **Mastery Visualization**: Real-time mastery tracking for each topic
- **Progress Monitoring**: Visual progress bars and statistics
- **Performance Metrics**: Accuracy tracking and learning analytics
- **Concept Mapping**: Displays concept names instead of IDs for better understanding

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← User Interface
└────────┬────────┘
         │
┌────────▼────────┐
│  FastAPI       │  ← Backend API Server
│  Backend       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────────┐
│  KG   │ │   OpenAI    │
│ Graph │ │   GPT-3.5   │
└───────┘ └─────────────┘
```

### Components

#### 1. **Frontend Layer** (`streamlit/app_streamlit.py`)
- **Streamlit-based User Interface**: Modern, responsive web interface
- **Content Display**: Renders explanations, examples, and MCQs with formatted mathematical notation
- **Interactive Question Answering**: Real-time feedback with immediate explanations
- **Mastery Visualization**: Plotly charts (pie chart and bar chart) for mastery tracking
- **Progress Monitoring**: Real-time progress bars and statistics dashboard
- **Session Management**: Client-side state management for student sessions

#### 2. **Backend API Layer** (`streamlit/backend_api.py`)
- **FastAPI RESTful Server**: Handles all API requests from the frontend
- **Student Session Management**: Tracks individual student sessions and history
- **Knowledge State Tracking**: Maintains and updates mastery levels per concept
- **Content Generation Orchestration**: Coordinates between content generators
- **Knowledge Graph Access**: Provides endpoints for topic and concept data
- **Answer Submission**: Processes student answers and updates knowledge state

#### 3. **Adaptive Content Generator** (`adaptive_content_generator.py`)
- **Core Content Generation Engine**: Main orchestrator for educational content
- **OpenAI GPT-3.5 Turbo Integration**: Generates high-quality explanations, examples, and questions
- **Transformer DKT Integration**: Uses DKT model for knowledge state prediction
- **Knowledge Graph Embeddings**: Leverages concept embeddings for context-aware generation
- **Difficulty Adaptation**: Adjusts content difficulty based on student performance
- **Content Type Support**: Generates explanations, examples, and questions
- **Logging System**: Comprehensive logging of all generated content

#### 4. **MCQ Generator Module** (`streamlit/mcq_generator.py`)
- **Specialized Question Generation**: Focused on multiple-choice question creation
- **Anti-Repetition System**: Tracks recent questions to ensure diversity
- **Semantic Similarity Checking**: Uses sentence transformers to detect similar questions
- **Question Validation**: Ensures correct answer is always present in options
- **Randomized Option Placement**: Shuffles correct answer position
- **Plausible Distractors**: Generates realistic incorrect options
- **Retry Logic**: Automatically regenerates questions that are too similar
- **Question Type Variations**: Supports different question formats and contexts

#### 5. **Transformer DKT Model** (`transformer_dkt_model.py`)
- **Deep Knowledge Tracing**: Predicts student knowledge state over time
- **Transformer Architecture**: Replaces LSTM with Transformer Encoder for better pattern recognition
- **Multi-head Self-Attention**: Captures long-range dependencies in learning sequences
- **Positional Encoding**: Maintains sequence information for temporal patterns
- **Feature Processing**: Handles time, difficulty, streak, and other features
- **Model Training**: Supports training on EdNet and Assistments datasets
- **Knowledge State Prediction**: Provides probability of correct answer for each concept

#### 6. **Knowledge Graph Builder** (`knowledge_graph.py`)
- **NLP-based Graph Construction**: Uses real NLP techniques for relationship extraction
- **Sentence Embeddings**: Semantic similarity using Sentence Transformers
- **Named Entity Recognition**: Identifies mathematical concepts using spaCy
- **Dependency Parsing**: Extracts prerequisite relationships
- **TF-IDF Analysis**: Keyword extraction for concept identification
- **Cosine Similarity**: Finds related concepts based on embeddings
- **Graph Structure**: NetworkX DiGraph with 441+ mathematical concepts
- **Prerequisite Relationships**: Models learning dependencies between concepts

#### 7. **Content Difficulty Module** (`content_difficulty.py`)
- **Adaptive Difficulty Selection**: Recommends difficulty level based on knowledge state
- **DKT-based Prediction**: Uses Transformer DKT to predict student performance
- **Difficulty Levels**: Supports easy, medium, and hard difficulty adaptation
- **Knowledge State Caching**: Optimizes performance with caching
- **Batch Processing**: Handles multiple students efficiently
- **EdNet Data Integration**: Loads and processes EdNet dataset for training

#### 8. **Knowledge Graph Data** (`khan_knowledge_graph_nlp.pkl`)
- **441+ Mathematical Concepts**: Comprehensive coverage of K-12 mathematics
- **Prerequisite Relationships**: Directed edges showing concept dependencies
- **Concept Embeddings**: Vector representations for semantic similarity
- **Node Attributes**: Complexity, category, and metadata for each concept
- **Edge Attributes**: Relationship types and methods (prerequisite, related, etc.)

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key
- Required Python packages (see Installation)

### Installation

1. **Clone the repository**
   ```bash
   cd capstone
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit fastapi uvicorn openai networkx plotly matplotlib numpy pandas requests pydantic
   ```

3. **Set up OpenAI API Key**
   
   Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
   
   **Important**: Never commit your API key to the repository. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

4. **Verify Knowledge Graph**
   
   Ensure `streamlit/khan_knowledge_graph_nlp.pkl` exists in the streamlit directory.

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd streamlit
   python backend_api.py
   ```
   The backend will run on `http://localhost:8000`

2. **Start the Streamlit Frontend** (in a new terminal)
   ```bash
   cd streamlit
   streamlit run app_streamlit.py
   ```
   The application will open in your browser at `http://localhost:8501`

### 🐳 Docker Deployment (Recommended)

For easier deployment and consistency across environments, use Docker:

1. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Quick start with script**
   ```bash
   ./docker-start.sh
   ```

3. **Or use docker-compose directly**
   ```bash
   # Build and start services
   docker-compose up --build -d
   
   # View logs
   docker-compose logs -f
   
   # Stop services
   docker-compose down
   ```

4. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

For detailed Docker deployment instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md).

## 📖 Usage Guide

### For Students

1. **Enter Student ID**: Enter a unique identifier (e.g., "student_001")

2. **Select Topic**: Choose from 441+ available topics in the dropdown

3. **Choose Content Type**:
   - **Explanation**: Detailed explanations of concepts
   - **Example**: Worked examples with step-by-step solutions
   - **Multiple Choice Questions**: Practice questions with immediate feedback

4. **Set Difficulty**: Select Easy, Medium, or Hard based on your comfort level

5. **Generate Content**: Click "Generate Content" and wait for personalized content

6. **Answer Questions**: For MCQs, select your answer and get instant feedback with explanations

7. **Track Progress**: View your mastery level for the current topic in the sidebar

### Mastery Levels

- **🟢 Mastered (≥70%)**: Strong understanding of the topic
- **🟡 Learning (40-70%)**: Good progress, keep practicing
- **🔴 Beginner (<40%)**: Continue learning to improve

## 🔌 API Endpoints

The FastAPI backend provides the following endpoints:

### Content Generation
- `POST /generate-content`: Generate educational content
  ```json
  {
    "student_id": "string",
    "topic": "string",
    "content_type": "explanation|example|question",
    "difficulty": "easy|medium|hard",
    "n_items": 1
  }
  ```

### Knowledge State
- `GET /knowledge-state/{student_id}`: Get student's knowledge state
- `POST /submit-answer`: Submit an answer and update mastery
  ```json
  {
    "student_id": "string",
    "topic": "string",
    "concept_id": 1,
    "correct": true
  }
  ```

### Topics & Concepts
- `GET /topics`: Get list of available topics
- `GET /concept-mapping`: Get mapping of concept IDs to names
- `GET /knowledge-graph-data/{student_id}`: Get knowledge graph data with mastery

## 📁 Project Structure

```
capstone/
├── streamlit/                          # Streamlit application directory
│   ├── app_streamlit.py               # Main Streamlit frontend UI
│   ├── backend_api.py                 # FastAPI backend server
│   ├── mcq_generator.py               # MCQ generation with anti-repetition
│   ├── app.py                         # Alternative Streamlit app
│   └── khan_knowledge_graph_nlp.pkl   # Knowledge graph data (441+ concepts)
│
├── Core Modules/
│   ├── adaptive_content_generator.py  # Main content generation engine
│   ├── transformer_dkt_model.py       # Transformer-based DKT model
│   ├── knowledge_graph.py             # NLP-based knowledge graph builder
│   ├── content_difficulty.py          # Adaptive difficulty selection
│   └── streamlit_app.py               # Alternative Streamlit implementation
│
├── Data & Models/
│   ├── dkt_model.pt                   # Trained DKT model weights
│   ├── models/                         # Additional model files
│   ├── ednet/                          # EdNet dataset files
│   └── khan-exercises/                 # Khan Academy exercise data
│
├── Visualization & Analysis/
│   ├── khan visualisations/           # Knowledge graph visualizations
│   └── ednet_analysis_output/         # EdNet dataset analysis
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🛠️ Technologies Used

### Frontend & Backend
- **Streamlit**: Interactive web interface framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI

### AI & Machine Learning
- **OpenAI GPT-3.5 Turbo**: Content generation (explanations, examples, questions)
- **Transformer DKT**: Deep Knowledge Tracing model for knowledge state prediction
- **Sentence Transformers**: Semantic similarity for question diversity
- **PyTorch**: Deep learning framework for DKT model
- **spaCy**: NLP for named entity recognition and dependency parsing
- **scikit-learn**: TF-IDF vectorization and similarity metrics

### Data Processing & Storage
- **NetworkX**: Knowledge graph data structure and graph algorithms
- **NumPy & Pandas**: Data manipulation and numerical computations
- **Pickle**: Serialization for model and graph storage

### Visualization
- **Plotly**: Interactive charts (pie charts, bar charts) for mastery visualization
- **Matplotlib**: Static visualizations and model training curves

### Language & Environment
- **Python 3.10+**: Primary programming language
- **Environment Variables**: Secure API key management

## ⚙️ Configuration

### Timeout Settings
- Question generation: 90 seconds base + 30 seconds per additional question
- API calls: 10-15 seconds for standard requests

### Mastery Calculation
- Default mastery: 0.0 (0%)
- Update formula: `new_mastery = 0.7 * old_mastery + 0.3 * target`
- Target: 1.0 for correct answers, 0.0 for incorrect

### Content Generation
- Maximum retries: 3 attempts per question
- Anti-repetition: Tracks last 15 questions per topic
- Similarity threshold: 75% word overlap triggers regeneration

## 🎨 UI Features

- **Modern Design**: Gradient headers, card-based layout
- **Responsive Layout**: Wide layout optimized for learning
- **Color-Coded Feedback**: 
  - Green for correct answers
  - Red for incorrect answers
  - Color-coded mastery levels
- **Readable Content**: Formatted mathematical notation and explanations
- **Progress Visualization**: Real-time progress bars and statistics

## 🔧 Troubleshooting

### Backend Not Starting
- Check if port 8000 is available
- Verify all dependencies are installed
- Check that `khan_knowledge_graph_nlp.pkl` exists

### Timeout Errors
- Reduce number of questions generated at once
- Check OpenAI API status
- Verify network connection

### Concept Names Not Showing
- Restart backend server to reload concept mapping
- Clear browser cache
- Use a new Student ID for fresh session

### Mastery Not Updating
- Ensure backend server is running
- Check that answers are being submitted correctly
- Verify student session exists

## 📊 Data Model & User Interactions

### Question-Related Data Fields (KT2)

The system tracks comprehensive data for each question interaction:

1. **`timestamp`**: The exact moment when the question was given to the student
2. **`solving_id`**: Unique identifier representing the learning session for each student
3. **`question_id`**: The unique identifier of the question being answered
4. **`user_answer`**: The answer submitted by the student (typically: a, b, c, or d)
5. **`elapsed_time`**: The time spent by the student on each question (in seconds)


### Data Collection Benefits

This comprehensive tracking enables:
- **Performance Analysis**: Detailed understanding of student behavior and learning patterns
- **Adaptive Recommendations**: Data-driven personalization of content delivery
- **Learning Path Optimization**: Identification of optimal learning sequences
- **Engagement Metrics**: Measurement of student interaction with multimedia content
- **Time Management Insights**: Analysis of time spent on different question types

## 📝 Notes

- **Session Management**: Student sessions are stored in memory and reset when the backend restarts
- **Default Mastery**: New concepts start at 0% mastery
- **Question Diversity**: The system actively prevents repetitive questions using semantic similarity
- **Content Quality**: All content is generated using OpenAI GPT-3.5 Turbo for high-quality educational material
- **Knowledge Graph**: Built using NLP techniques (sentence embeddings, NER, dependency parsing)
- **DKT Model**: Transformer-based architecture for improved knowledge state prediction

## 🤝 Contributing

This is a capstone project. For improvements or bug fixes, please create an issue or submit a pull request.

## 📄 License

This project is part of an academic capstone project.

## 🙏 Acknowledgments

- OpenAI for GPT-3.5 Turbo API
- Khan Academy for knowledge graph concepts
- Streamlit and FastAPI communities

---



