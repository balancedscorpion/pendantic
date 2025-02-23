# Conversation Analysis Tools

This repository contains tools for analyzing conversations using semantic entropy and lexical density metrics. These tools help understand the complexity and diversity of language used in conversations.

## Installation

1. Install the required dependencies:
```bash
pip install spacy pandas plotly scipy scikit-learn
python -m spacy download en_core_web_sm  # For lexical density
python -m spacy download en_core_web_lg  # For semantic entropy (includes word vectors)
```

## Tools Overview

### 1. Lexical Density Analysis (`analyze_lexical_density.py`)

Measures the proportion of content words (nouns, verbs, adjectives, adverbs) to total words, indicating text complexity and information density.

```bash
# Basic usage
python analyze_lexical_density.py

# Analyze specific conversation range
python analyze_lexical_density.py --start 0 --end 100

# Analyze from a JSON file
python analyze_lexical_density.py --test-file conversations.json
```

#### Output
- Timeline plot showing lexical density over time
- Average density statistics per speaker
- Raw data for further analysis

### 2. Semantic Entropy Analysis (`analyze_semantic_entropy.py`)

Measures the semantic diversity and unpredictability of conversation content using word embeddings.

```bash
# Basic usage
python analyze_semantic_entropy.py

# Analyze specific conversation range
python analyze_semantic_entropy.py --start 0 --end 100

# Analyze from a JSON file
python analyze_semantic_entropy.py --test-file conversations.json
```

#### Output
- Two plots:
  1. Local Semantic Entropy: Diversity within each message
  2. Global Semantic Entropy: Deviation from typical language usage
- Running averages shown as dashed lines
- Statistics per speaker
- Raw data for further analysis

## Input Data Format

Both tools accept conversations in JSON format:

```json
[
    {
        "timestamp": "2024-01-01T10:00:00Z",
        "speaker": "user",
        "transcript": "Message content here"
    },
    {
        "timestamp": "2024-01-01T10:01:00Z",
        "speaker": "assistant",
        "transcript": "Response content here"
    }
]
```

## Metrics Explained

### Lexical Density
- **What it measures**: Ratio of content words to total words (%)
- **Higher values**: More complex, information-dense text
- **Lower values**: More functional, potentially more conversational text

### Semantic Entropy
1. **Local Entropy**
   - Measures semantic diversity within a single message
   - Higher values: More diverse/varied semantic content
   - Lower values: More semantically focused/consistent content

2. **Global Entropy**
   - Measures how much the text deviates from typical language patterns
   - Higher values: More unique/specialized semantic content
   - Lower values: More conventional/standard language use

## Example Usage in Code

```python
# Lexical Density Analysis
from src.lexical_density_agent import LexicalDensityAgent

agent = LexicalDensityAgent()
results = agent.analyze_conversation(messages)
print(f"Average Lexical Density: {results['average_density']}")

# Semantic Entropy Analysis
from src.semantic_entropy_agent import SemanticEntropyAgent

agent = SemanticEntropyAgent()
results = agent.analyze_conversation(messages)
print(f"Local Entropy Stats: {results['average_local']}")
print(f"Global Entropy Stats: {results['average_global']}")
```

## Visualization Tips

1. **Interactive Plots**
   - All plots are interactive (powered by Plotly)
   - Hover over points to see message content
   - Click and drag to zoom
   - Double-click to reset view

2. **Reading the Plots**
   - Scatter points: Individual message metrics
   - Dashed black line: Running average
   - Colors: Different speakers
   - X-axis: Time or message sequence
   - Y-axis: Metric value

## Common Issues and Solutions

1. **Missing Word Vectors**
   ```
   Error: Can't find model 'en_core_web_lg'
   Solution: Run python -m spacy download en_core_web_lg
   ```

2. **Low Entropy Values**
   - Check message length (need at least 2 content words)
   - Verify content isn't just stop words/punctuation

3. **Performance**
   - Semantic entropy is more computationally intensive
   - For large datasets, consider using conversation ranges

## Contributing

When adding features or fixes:
1. Maintain the same metric calculation methodology
2. Keep visualization consistent with existing plots
3. Add appropriate documentation
4. Include test cases with example conversations 