from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import brown

# Download the Brown corpus
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')

def tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and converting to lowercase."""
    return text.lower().split()

def load_brown_corpus() -> List[str]:
    """Load sentences from the Brown corpus."""
    return [' '.join(sent) for sent in brown.sents()]

def build_1gram_model(corpus: List[str]) -> Dict[str, int]:
    """Build a 1-gram (unigram) model from a corpus."""
    tokens = []
    for text in corpus:
        tokens.extend(tokenize(text))
    return Counter(tokens)

def build_2gram_model(corpus: List[str]) -> Dict[Tuple[str], int]:
    """Build a 2-gram (bigram) model from a corpus."""
    bigrams = []
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))
    return Counter(bigrams)

def build_3gram_model(corpus: List[str]) -> Dict[Tuple[str, str, str], int]:
    """Build a 3-gram (trigram) model from a corpus."""
    trigrams = []
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens) - 2):
            trigrams.append((tokens[i], tokens[i + 1], tokens[i + 2]))
    return Counter(trigrams)

def calculate_probabilities(ngram_counts: Dict, total_count: int = None) -> Dict:
    """Convert n-gram counts to probabilities."""
    if total_count is None:
        total_count = sum(ngram_counts.values())
    return {ngram: count / total_count for ngram, count in ngram_counts.items()}

# Example usage
if __name__ == "__main__":
    # Load Brown corpus
    print("Loading Brown corpus...")
    corpus = load_brown_corpus()
    print(f"Loaded {len(corpus)} sentences from Brown corpus")

    # Build models
    print("\nBuilding n-gram models...")
    unigrams = build_1gram_model(corpus)
    bigrams = build_2gram_model(corpus)
    trigrams = build_3gram_model(corpus)

    # Display results
    print("\nTop 10 1-grams (unigrams):")
    for word, count in unigrams.most_common(10):
        print(f"  {word}: {count}")

    print("\nTop 10 2-grams (bigrams):")
    for bigram, count in bigrams.most_common(10):
        print(f"  {bigram}: {count}")

    print("\nTop 10 3-grams (trigrams):")
    for trigram, count in trigrams.most_common(10):
        print(f"  {trigram}: {count}")

    # Calculate probabilities
    unigram_probs = calculate_probabilities(unigrams)
    print("\nSample 1-gram probabilities:")
    for word, prob in list(unigram_probs.items())[:5]:
        print(f"  {word}: {prob:.6f}")
