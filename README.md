# Constraint-Based POS Tagging Engine (CSP-Driven)

## Overview  
This solution deploys a deterministic, constraint-driven POS tagging workflow engineered through a CSP (Constraint Satisfaction Problem) paradigm. The system orchestrates domain generation, lexicon construction, exhaustive search, constraint enforcement, and analytics visualization. It is optimized for reproducible experimentation, linguistic rule-based modeling, and algorithmic benchmarking.

## Key Features  
- Deterministic token-domain generation with controlled ambiguity for evaluation consistency  
- Lexicon builder translating domain structures into clean lookup maps  
- CSP engine supporting pairwise constraint checks, backtracking-style pruning, and search analytics  
- Automated statistics across tested combinations, violations, and pruning efficiency  
- Visualization module highlighting constraint behavior, tag distributions, and search-space metrics  
- Fully operational CLI pipeline for streamlined experimentation  

## Dataset Requirement  
No external dataset is required. Domains are synthesized dynamically from user-provided sentences, with predefined cases available for consistent testing scenarios.

## Workflow Architecture  
1. Initialize global visualization and environment configurations.  
2. Accept user input and generate POS domains using deterministic rule patterns.  
3. Convert domain structures into a lexicon for CSP evaluation.  
4. Expand full tag-combination search space and execute constraint validation.  
5. Record statistics, compute pruning efficiency, and identify valid tag sequences.  
6. Emit tagged outputs, domain summaries, and performance metrics.  
7. Render visual insights summarizing constraint dynamics and tag distribution.  

## Installation  

### 1. Clone the repository  
```bash
git clone <your-repo-url>
cd <project-folder>
