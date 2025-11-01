# AGORA v2 - System Architecture

## Overview

AGORA v2 is a graph-backed question answering system that combines multimodal data sources (text and video) with advanced graph algorithms to produce comprehensive, well-researched answers.

## System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
│  "Give me the latest products, competitors, and collaborators   │
│   of Nvidia. What will affect it most?"                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 1: ENTITY EXTRACTION                     │
│                    (entity_extractor.py)                        │
│                                                                 │
│  LLM (OpenAI GPT-4o-mini) extracts seed entity: "Nvidia"        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              STAGE 2-4: GRAPH BUILDING (BFS)                   │
│                   (graph_builder.py)                           │
│                                                                │
│  Queue: [(Nvidia, depth=0)]                                    │
│                                                                │
│  For each entity in queue:                                     │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TEXT SENSE (text_sense.py)                               │  │
│  │ ├─ Query Parallel API (Ingest for Schema + Task for ans) │  │
│  │ ├─ Get: concise answer                                   │  │
│  │ └─ Get: recent key events                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                  │
│                             ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ VIDEO SENSE (janus.py)                                   │  │
│  │ ├─ Search YouTube for event videos                       │  │
│  │ ├─ Get transcripts (youtube Transcript API or Whisper)   │  │
│  │ ├─ LLM summarizes transcript based on user query         │  │
│  │ └─ Return focused summaries                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                  │
│                             ▼                                  │
│            combined_context = answer + transcripts             │
│                             │                                  │
│                             ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ENTITY SENSE (entity_sense.py)                           │  │
│  │ ├─ LLM OpenAI (GPT-4o-mini) extracts related entities    │  │
│  │ ├─ Add current entity as node to graph                   │  │
│  │ ├─ Add edges to new entities                             │  │
│  │ └─ Add new entities to queue (depth + 1)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Repeat until: len(nodes) >= 25 OR depth > 2                   │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │  KNOWLEDGE GRAPH │
                  └─────────┬────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 5: FEATURIZATION                           │
|(CAN BE REPLACED BY BETTER EMBEDDERS LIKE OPENAI AND MAYBE CACHE |
|                   COMMONLY USED ONES)                           |
│                    (featurizer.py)                              │
│                                                                 │
│  For each node:                                                 │
│    embedding = SBERT.encode(combined_context)                   │
│    node['embedding'] = embedding                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 6: RANKING (GRAIL Algorithm)                 │
│ (CAN BE REPLACED BY BETTER ALGORITHMS LIKE GNN FOR LARGER GRAPH)│
│                     (ranker.py)                                 │
│                                                                 │
│  1. For each edge:                                              │
│       weight = cosine_similarity(node_u, node_v)                │
│                                                                 │
│  2. Run weighted PageRank                                       │
│                                                                 │
│  3. Sort nodes by PageRank score                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │   RANKED NODES   │
                  │  1. Nvidia: 0.35 │
                  │  2. AMD: 0.20    │
                  │  3. TSMC: 0.15   │
                  │  ...             │
                  └─────────┬────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│             STAGE 7: SYNTHESIS                                  │
│                (synthesizer.py)                                 │
│                                                                 │
│  1. Select Top 5 ranked nodes                                   │
│  2. Extract their combined_context                              │
│  3. LLM generates comprehensive answer                          │
│     using only the Top 5 contexts                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    AGORA_RESPONSE                              │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ final_answer (str)                                      │   │
│  │ "Nvidia dominates the AI chip market with their H100... │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ranked_nodes (dict)                                     │   │
│  │ {"Nvidia": 0.35, "AMD": 0.20, "TSMC": 0.15, ...}        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ graph (networkx.Graph)                                  │   │
│  │ Full graph object with all nodes, edges, and data       │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

## Key Algorithms

### Breadth-First Search (BFS)
- Explores entities level by level
- Ensures diverse entity coverage
- Prevents getting stuck in deep chains

### GRAIL Ranking
1. **Featurization**: SBERT embeddings capture semantic meaning
2. **Edge Weighting**: Cosine similarity measures relevance
3. **PageRank**: Identifies central, influential nodes
4. **Result**: Non-obvious, important entities rise to top

### PageRank Formula
```
PR(v) = (1-d)/N + d × Σ(PR(u)/L(u) × w(u,v))

where:
- d = damping factor (0.85)
- N = number of nodes
- u = neighbors of v
- L(u) = out-degree of u
- w(u,v) = edge weight (cosine similarity)
```

## Design Decisions

### Why weighted PageRank?
- **Semantic**: Edge weights reflect actual relevance
- **Ease**: Easy to implement

### Why SBERT over other embeddings?
- **Speed**: Fast inference (~50ms per encoding)
- **Quality**: Good semantic understanding
- **Size**: Small model (80MB) that fits in memory

## Extension Points

The system is designed to be modular and extensible:

1. **New Data Sources**: Add modules like `twitter_sense.py`, `reddit_sense.py`
2. **Better Ranking**: Replace PageRank with Graph Attention Networks
3. **Caching**: Add Redis cache for processed entities
4. **Interactive**: Add user feedback to refine graph expansion

