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

1. **Frontend (Streamlit)**: `streamlit/app_streamlit.py`
   - Interactive user interface
   - Content display and question answering
   - Mastery visualization

2. **Backend API (FastAPI)**: `streamlit/backend_api.py`
   - RESTful API endpoints
   - Student session management
   - Knowledge state tracking

3. **Content Generators**:
   - `adaptive_content_generator.py`: Main content generation engine
   - `mcq_generator.py`: Specialized MCQ generation with anti-repetition

4. **Knowledge Graph**: `khan_knowledge_graph_nlp.pkl`
   - 441+ mathematical concepts
   - Prerequisite relationships
   - Concept embeddings

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
├── streamlit/
│   ├── app_streamlit.py          # Main Streamlit frontend
│   ├── backend_api.py            # FastAPI backend server
│   ├── mcq_generator.py          # MCQ generation module
│   ├── adaptive_content_generator.py  # Content generation engine
│   └── khan_knowledge_graph_nlp.pkl   # Knowledge graph data
├── adaptive_content_generator.py      # Core content generator
├── transformer_dkt_model.py           # DKT model implementation
├── knowledge_graph.py                 # Knowledge graph utilities
└── README.md                          # This file
```

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Backend**: FastAPI, Uvicorn
- **AI/ML**: OpenAI GPT-3.5 Turbo, Transformer DKT
- **Data Structures**: NetworkX (Knowledge Graph)
- **Visualization**: Plotly, Matplotlib
- **Language**: Python 3.10+

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

## 📝 Notes

- **Session Management**: Student sessions are stored in memory and reset when the backend restarts
- **Default Mastery**: New concepts start at 0% mastery
- **Question Diversity**: The system actively prevents repetitive questions
- **Content Quality**: All content is generated using OpenAI GPT-3.5 Turbo for high-quality educational material

## 🤝 Contributing

This is a capstone project. For improvements or bug fixes, please create an issue or submit a pull request.

## 📄 License

This project is part of an academic capstone project.

## 🙏 Acknowledgments

- OpenAI for GPT-3.5 Turbo API
- Khan Academy for knowledge graph concepts
- Streamlit and FastAPI communities

---



