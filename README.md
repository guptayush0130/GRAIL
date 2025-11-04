# AGORA - Graph-Backed Question Answering System

A sophisticated AI system that answers complex questions by building a knowledge graph from multimodal sources (text and video), then using graph-based ranking to synthesize comprehensive answers.

## Overview

AGORA transforms a user question into a synthesized, graph-backed answer through a 7-stage pipeline:

1. **Query Input** - Receive user question
2. **Entity Extraction** - Identify seed entity using LLM
3. **Graph Building** - Build knowledge graph via BFS using Text and Video sources
4. **Entity Expansion** - Extract related entities using LLM
5. **Featurization** - Embed all nodes using SBERT
6. **Ranking** - Apply GRAIL algorithm (weighted PageRank)
7. **Synthesis** - Generate final answer from top-ranked nodes using LLM

## Architecture Components

### Core Modules

- **`agora.py`** - Main entry point, orchestrates the entire pipeline
- **`entity_extractor.py`** - Extracts seed entity from user query
- **`graph_builder.py`** - BFS orchestrator that builds the knowledge graph
- **`entity_sense.py`** - Extracts new entities for graph expansion
- **`text_sense.py`** - Gets initial data and key events using Parallel API
- **`janus.py`** - Finds YouTube videos, transcribes them, and uses LLM to summarize relevant content
- **`featurizer.py`** - Generates SBERT embeddings for nodes
- **`ranker.py`** - GRAIL ranking algorithm with weighted PageRank
- **`synthesizer.py`** - Generates final answer from ranked nodes

### Configuration

- **`config.py`** - API keys (OPENAI_API_KEY, YOUTUBE_API_KEY, PARALLEL_API_KEY)

## Installation

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

## 1. Setup (2 minutes)

### Option A: Automated Setup (Recommended)
```bash
cd GRAIL
./setup.sh
```

### Option B: Manual Setup
```bash
cd GRAIL
source venv/bin/activate
pip install -r requirements.txt
```
```

### Configuration

Create or update `config.py` with your API keys:

```python
OPENAI_API_KEY = "your-openai-key"
YOUTUBE_API_KEY = "your-youtube-key"
PARALLEL_API_KEY = "your-parallel-key"
```

**⚠️ Security Note:** Never commit `config.py` to version control. Consider using environment variables instead.

## Usage

### Basic Usage

```python
from agora import run

# Ask a question
response = run(
    user_query="Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia. What things are going to affect it the most?",
    max_nodes=25,
    max_depth=2
)

# Access the results
print(response.final_answer)
print(response.ranked_nodes)
print(response.graph)
```

### Run from Command Line

```bash
python agora.py
```

This will run a test query and save the results to `agora_response.json`.

### AGORA_Response Object

The system returns an `AGORA_Response` object with:

- **`final_answer`** (str) - Multi-paragraph synthesized answer
- **`ranked_nodes`** (dict) - PageRank scores for all nodes, sorted by importance
- **`graph`** (networkx.Graph) - Full knowledge graph with all nodes and edges

## Example

**Input:**
```
"Give me the latest products, performance, competitors, opportunities, and key collaborators of Nvidia. What things are going to affect it the most?"
```

**Process:**
Look at ARCHITECTURE.md

**Output:**
- Comprehensive multi-paragraph answer
- Ranked list of influential entities
- Full knowledge graph for visualization

## Testing Individual Modules

Each module can be tested independently:

```bash
python entity_extractor.py
python text_sense.py
python janus.py
python entity_sense.py
python featurizer.py
python ranker.py
python synthesizer.py
python graph_builder.py
```

## Advanced Features

### Graph Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx

response = run("Your question here")
nx.draw(response.graph, with_labels=True)
plt.show()
```

### Export Graph

```python
import json

# Export as JSON
graph_data = nx.node_link_data(response.graph)
with open('graph.json', 'w') as f:
    json.dump(graph_data, f)
```

## Project Structure

```
GRAIL/
├── agora.py              # Main entry point
├── entity_extractor.py   # Component 1: Seed entity extraction
├── graph_builder.py      # Component 2: BFS graph builder
├── text_sense.py         # Component 3: Parallel API data ingestion
├── janus.py              # Component 4: YouTube video transcription
├── entity_sense.py       # Component 5: Entity expansion
├── featurizer.py         # Component 6: SBERT embeddings
├── ranker.py             # Component 7: GRAIL PageRank algorithm
├── synthesizer.py        # Component 8: Answer synthesis
├── config.py             # API keys (not in git)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment
```

## API Requirements

1. **OpenAI API** - For LLM calls (entity extraction, entity sense, synthesis)
   - Get key: https://platform.openai.com/api-keys
   
2. **YouTube Data API v3** - For video search
   - Get key: https://console.cloud.google.com/apis/credentials
   
3. **Parallel API** - For text data ingestion
   - Get key: https://parallel.ai

## Troubleshooting

### Issue: "Module not found"
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

### Issue: "API key error"
- Check `config.py` has all three API keys
- Verify keys are valid and have appropriate permissions

### Issue: "Video transcription failing"
- Check YouTube API quota (default: 10,000 units/day)
- Some videos may not have transcripts available
- Janus has a Whisper fallback for videos without transcripts

## License

MIT License.

## Credits

Built with:
- NetworkX for graph operations
- Sentence-BERT for embeddings
- OpenAI GPT-4 for synthesis
- Parallel AI for data ingestion
- YouTube API for video content
