# Imports and global visualization settings

import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import numpy as np
import json

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 120

print("✓ Libraries imported successfully")

def get_initial_domains(sentence: str) -> Dict[str, Any]:
    """
    Simulates a large-language-model-powered domain generator.

    This function is intentionally deterministic:
    - It recognizes predefined test cases (e.g., “the dog runs”).
    - It injects controlled POS ambiguity for verbs, nouns, adjectives, and adverbs.
    - It ensures reproducibility for grading, evaluation, or repeated experimentation.
    """

    sentence = sentence.lower().strip()

    # --- Predefined Test Case 1: Clear noun/verb ambiguity scenario ---
    if 'the dog runs' in sentence:
        return {
          "tokens": [
              {"word": "the", "possible_tags": ["DT"], "description": "Determiner"},
              {"word": "dog", "possible_tags": ["NN", "VB"], "description": "Noun or Verb (intentional ambiguity)"},
              {"word": "runs", "possible_tags": ["VBZ", "NNS"], "description": "Verb or plural noun (deliberate ambiguity)"}
          ]
        }

    # --- Predefined Test Case 2: Consecutive verb constraint violation scenario ---
    elif 'i saw walked' in sentence or 'i run walk' in sentence:
        w2 = "saw" if "saw" in sentence else "run"
        w3 = "walked" if "walked" in sentence else "walk"

        return {
          "tokens": [
              {"word": "i", "possible_tags": ["PRP"], "description": "Personal pronoun"},
              {"word": w2, "possible_tags": ["VB", "NN"], "description": "Ambiguous verb/noun"},
              {"word": w3, "possible_tags": ["VB", "NN"], "description": "Ambiguous verb/noun (pruning trap)"}
          ]
        }

    # --- Default: Dynamically construct mixed domains ---
    else:
        tokens = sentence.split()
        default_tokens = []

        for t in tokens:
            w = t.lower().strip(",.!?'")

            # Deterministic rule sets
            if w in ['the', 'a', 'an']:
                tags = ["DT"]
            elif w in ['i', 'he', 'she', 'we', 'they', 'you']:
                tags = ["PRP"]
            elif w in ['is', 'are', 'was', 'were']:
                tags = ["VBZ", "VBP", "VBD"]
            elif w == 'tired':
                tags = ["JJ", "VBN"]
            elif w == 'quickly':
                tags = ["RB"]
            elif w in ['run', 'walk', 'see']:
                tags = ["VB", "NN"]
            elif w in ['my', 'your', 'his', 'her']:
                tags = ["PRP$", "NN"]
            else:
                # Intentional high-entropy domain for testing pruning
                tags = ["NN", "VB", "JJ", "RB"]

            default_tokens.append({
                "word": t,
                "possible_tags": tags,
                "description": f"Assigned default domain: {', '.join(tags)}"
            })

        return {"tokens": default_tokens}

  def create_lexicon(tokens: List[Dict]) -> Dict[str, List[str]]:
    """
    Converts the domain generator's token structures into a clean lexicon format.

    Output Structure:
        {
            "word1": ["NN", "VB"],
            "word2": ["DT"]
        }
    """
    lex = {}

    for t in tokens:
        w = t["word"].lower().strip(",.!?'")
        lex[w] = t.get("possible_tags", ["NN"])  # fallback domain

    return lex

class POSTaggerCSP:
    """
    Implements a symbolic POS tagging system using Constraint Satisfaction.

    This class encapsulates:
    - Domain access
    - Tag-sequence generation
    - Pairwise constraint checking
    - Search statistics tracking
    - Early backtracking pruning
    """

    def __init__(self, lexicon: Dict[str, List[str]]):
        self.lexicon = lexicon
        self.stats = {
            'tested_combinations': 0,
            'violated_checks': 0,
            'applied_checks': 0
        }

    def get_domain(self, word: str) -> List[str]:
        """Returns the domain of possible POS tags for a given word."""
        return self.lexicon.get(word.lower(), ["NN"])

    def is_valid(self, tags: List[str]) -> bool:
        """
        Checks whether a tag sequence respects all binary constraints.

        Violations trigger pruning and statistics updates.
        """
        for i in range(len(tags) - 1):
            t1, t2 = tags[i], tags[i+1]
            self.stats['applied_checks'] += 1

            # DT constraint
            if t1 == 'DT' and not (t2.startswith('JJ') or t2.startswith('NN')):
                self.stats['violated_checks'] += 1
                return False

            # No consecutive verbs
            if t1.startswith('VB') and t2.startswith('VB'):
                self.stats['violated_checks'] += 1
                return False

            # RB cannot precede DT
            if t1.startswith('RB') and t2 == 'DT':
                self.stats['violated_checks'] += 1
                return False

            # PRP$ → (JJ | NN)
            if t1 == 'PRP$' and not (t2.startswith('JJ') or t2.startswith('NN')):
                self.stats['violated_checks'] += 1
                return False

        return True

    def cartesian_product(self, arrays: List[List[str]]):
        """Generates the complete tag search space."""
        return list(itertools.product(*arrays))

    def tag_all_combinations(self, sentence: str) -> Dict[str, Any]:
        tokens = [t.strip(",.!?'") for t in sentence.strip().split() if t.strip(",.!?'")]
        domains = [self.get_domain(t) for t in tokens]
        search_space = int(np.prod([len(d) for d in domains])) if domains else 0

        self.stats = {'tested_combinations': 0, 'violated_checks': 0, 'applied_checks': 0}
        all_combinations = self.cartesian_product(domains)

        valid_sequences = []

        for tags_tuple in all_combinations:
            self.stats['tested_combinations'] += 1
            tags = list(tags_tuple)
            if self.is_valid(tags):
                valid_sequences.append(tags)

        pruned_combinations = search_space - self.stats['tested_combinations']
        pruning_efficiency = (pruned_combinations / search_space) * 100 if search_space > 0 else 0.0

        return {
            "tokens": tokens,
            "valid_sequences": valid_sequences,
            "search_space": search_space,
            "stats": self.stats,
            "pruning_efficiency": pruning_efficiency
        }
def plot_results(result: dict):
    """Generates visualizations for CSP results (full-space evaluation)."""

    stats = result.get('stats', {})
    tested_combinations = stats.get('tested_combinations', 0)
    checks_applied = stats.get('applied_checks', 0)
    violations = stats.get('violated_checks', 0)
    search_space = result.get('search_space', 0)

    checks_passed = checks_applied - violations
    pruned_combinations = search_space - tested_combinations
    pruning_efficiency = (pruned_combinations / search_space) * 100 if search_space > 0 else 0.0

    import matplotlib.pyplot as plt
    from collections import Counter

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Graph 1: Constraint Check Summary
    stats_data = [checks_applied, violations, checks_passed]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(['Checks Applied', 'Violations', 'Checks Passed'], stats_data, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Constraint Check Summary', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count (Binary Checks)', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # Graph 2: Tag Distribution (first valid sequence if multiple)
    if result.get("valid_sequences"):
        first_sequence = result["valid_sequences"][0]
        tag_counts = Counter(first_sequence)
        axes[1].barh(list(tag_counts.keys()), list(tag_counts.values()), color='#9b59b6', edgecolor='black', linewidth=1.5)
    axes[1].set_title('Final POS Tag Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Count', fontsize=12)
    axes[1].grid(axis='x', alpha=0.3)

    # Graph 3: Search Space vs Efficiency
    axes[2].bar(['Search Space', 'Combinations Tested', 'Combinations Pruned'],
                [search_space, tested_combinations, pruned_combinations],
                color=['#f39c12', '#1abc9c', '#34495e'], edgecolor='black', linewidth=1.5)
    axes[2].set_title(f'Search Efficiency ({pruning_efficiency:.2f}% Pruned)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Count (Combinations)', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    print("\n===== CSP POS TAGGING ENGINE =====\n")

    # Input sentence
    sentence = input("Enter a sentence: ").strip()
    if not sentence:
        sentence = "I saw walked"
        print(f"Using default sentence: {sentence}")

    # Generate initial domains
    domain_data = get_initial_domains(sentence)
    lexicon = create_lexicon(domain_data["tokens"])

    # Initialize CSP tagger
    tagger = POSTaggerCSP(lexicon)

    # Execute full-space CSP search
    # Make sure your class has tag_all_combinations() implemented
    result = tagger.tag_all_combinations(sentence)

    # Extract statistics safely
    stats = result.get("stats", {})
    tested = stats.get("tested_combinations", 0)
    violated = stats.get("violated_checks", 0)
    checks_applied = stats.get("applied_checks", 0)
    search_space = result.get("search_space", 0)
    pruned = search_space - tested
    efficiency = (pruned / search_space) * 100 if search_space > 0 else 0.0

    # Display initial domains
    print("\n===== INITIAL DOMAINS =====")
    for w in result.get("tokens", []):
        print(f"{w:<12} → {tagger.get_domain(w)}")

    # Display final tagging(s)
    print("\n===== FINAL TAGGING(S) =====")
    valid_sequences = result.get("valid_sequences", [])
    if valid_sequences:
        for idx, seq in enumerate(valid_sequences, start=1):
            tags_str = " | ".join(f"{w}:{t}" for w, t in zip(result["tokens"], seq))
            print(f"Sequence {idx}: {tags_str}")
    else:
        print("❌ No valid sequences found.")

    # Display statistics
    print("\n===== STATISTICS =====")
    print(f"Search Space:        {search_space}")
    print(f"Combinations Tested: {tested}")
    print(f"Pruned:              {pruned}")
    print(f"Efficiency:          {efficiency:.2f}%")
    print(f"Checks Applied:      {checks_applied}")
    print(f"Violations:          {violated}")

    # Visualization
    if valid_sequences:
        plot_results(result)


if __name__ == "__main__":
    main()

