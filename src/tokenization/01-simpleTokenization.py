import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, gutenberg
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('gutenberg')


def tokenize_and_count(text, top_n=300, min_word_length=2, remove_stopwords=True):
    """
    Tokenize text and count word frequencies.

    Args:
        text (str): Input text to analyze
        top_n (int): Number of top words to return
        min_word_length (int): Minimum word length to include
        remove_stopwords (bool): Whether to remove stopwords

    Returns:
        list: List of tuples (word, count) for top N words
    """
    # Process text
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha() and len(word) > min_word_length]

    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

    # Count frequencies
    word_freq = Counter(words)
    return word_freq.most_common(top_n)


def plot_word_frequencies(top_words, title, figsize=(15, 8), show_values=True):
    """
    Create a bar chart of word frequencies.

    Args:
        top_words (list): List of tuples (word, count)
        title (str): Chart title
        figsize (tuple): Figure size
        show_values (bool): Whether to show count labels on bars
    """
    plt.figure(figsize=figsize)
    words_list, counts_list = zip(*top_words)

    plt.bar(words_list, counts_list, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    if show_values:
        for i, (word, count) in enumerate(top_words):
            plt.text(i, count + max(counts_list) * 0.01, str(count),
                    ha='center', va='bottom', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.show()


def load_text_from_file(filepath):
    """Load text from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_shakespeare_corpus():
    """Load all Shakespeare texts from gutenberg corpus."""
    shakespeare_files = [file for file in gutenberg.fileids() if 'shakespeare' in file.lower()]
    all_text = ""
    for file in shakespeare_files:
        all_text += gutenberg.raw(file)
    return all_text


def load_gutenberg_corpus():
    """Load all texts from the data/Gutenberg/txt folder."""
    import os
    import glob

    gutenberg_folder = 'data/Gutenberg/txt'
    all_text = ""
    file_count = 0

    print(f"Loading texts from {gutenberg_folder}...")

    # Get all .txt files
    txt_files = glob.glob(os.path.join(gutenberg_folder, '*.txt'))
    total_files = len(txt_files)

    print(f"Found {total_files} text files to process...\n")

    for filepath in txt_files:
        try:
            filename = os.path.basename(filepath)
            print(f"[{file_count + 1}/{total_files}] Processing: {filename}")

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                all_text += f.read() + " "
                file_count += 1
        except Exception as e:
            print(f"  ERROR reading {filepath}: {e}")

    print(f"\nSuccessfully loaded {file_count} files from Gutenberg corpus")
    return all_text


def print_top_words(top_words, n=50):
    """
    Print the top N most frequent words.

    Args:
        top_words (list): List of tuples (word, count)
        n (int): Number of words to print
    """
    print(f"\nTop {n} Most Frequent Words:")
    print("-" * 40)
    for i, (word, count) in enumerate(top_words[:n], 1):
        print(f"{i:3d}. {word:20s} {count:6d}")


def plot_zipf_law(top_words, title, n_words=1000):
    """
    Plot Zipf's law: log(frequency) vs log(rank).

    Args:
        top_words (list): List of tuples (word, count)
        title (str): Chart title
        n_words (int): Number of words to include in the plot
    """
    # Extract ranks and frequencies
    ranks = np.arange(1, min(len(top_words), n_words) + 1)
    frequencies = np.array([count for word, count in top_words[:n_words]])

    # Create log-log plot
    plt.figure(figsize=(12, 8))
    plt.loglog(ranks, frequencies, 'b.', alpha=0.6, markersize=8, label='Actual data')

    # Fit Zipf's law: frequency = k / rank
    # In log space: log(frequency) = log(k) - log(rank)
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    coefficients = np.polyfit(log_ranks, log_freqs, 1)
    fitted_freqs = np.exp(coefficients[1]) * ranks ** coefficients[0]

    plt.loglog(ranks, fitted_freqs, 'r-', linewidth=2,
               label=f'Zipf fit: slope={coefficients[0]:.2f}')

    plt.xlabel('Rank (log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    plt.title(f"Zipf's Law - {title}", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)

    # Add annotation
    plt.text(0.05, 0.05, f"Zipf's Law: frequency ∝ 1/rank^{abs(coefficients[0]):.2f}",
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


# Example usage: Analyze Infinite Jest
print("Analyzing Infinite Jest...")
infinite_jest_text = load_text_from_file('data/infinite-jest.txt')
infinite_jest_top_words = tokenize_and_count(infinite_jest_text, top_n=1000)
print_top_words(infinite_jest_top_words, n=50)
plot_word_frequencies(infinite_jest_top_words[:300], 'Top 300 Most Frequent Words in Infinite Jest')
plot_word_frequencies(infinite_jest_top_words[:50], 'Top 50 Most Frequent Words in Infinite Jest')
plot_zipf_law(infinite_jest_top_words, 'Infinite Jest', n_words=1000)

# Example usage: Analyze Shakespeare
print("\nAnalyzing Shakespeare corpus...")
shakespeare_text = load_shakespeare_corpus()
shakespeare_top_words = tokenize_and_count(shakespeare_text, top_n=1000)
print_top_words(shakespeare_top_words, n=50)
plot_word_frequencies(shakespeare_top_words[:300], 'Top 300 Most Frequent Words in Shakespeare Corpus')
plot_word_frequencies(shakespeare_top_words[:50], 'Top 50 Most Frequent Words in Shakespeare Corpus')
plot_zipf_law(shakespeare_top_words, 'Shakespeare Corpus', n_words=1000)

# Example usage: Analyze I Promessi Sposi
print("\nAnalyzing I Promessi Sposi...")
promessi_sposi_text = load_text_from_file('data/promessi-sposi.txt')
promessi_sposi_top_words = tokenize_and_count(promessi_sposi_text, top_n=1000)
print_top_words(promessi_sposi_top_words, n=50)
plot_word_frequencies(promessi_sposi_top_words[:300], 'Top 300 Most Frequent Words in I Promessi Sposi')
plot_word_frequencies(promessi_sposi_top_words[:50], 'Top 50 Most Frequent Words in I Promessi Sposi')
plot_zipf_law(promessi_sposi_top_words, 'I Promessi Sposi', n_words=1000)

# Example usage: Analyze Gutenberg Corpus
print("\n" + "="*60)
print("ANALYZING GUTENBERG CORPUS")
print("="*60)
gutenberg_text = load_gutenberg_corpus()
print("\nTokenizing Gutenberg corpus (this may take a few minutes)...")
gutenberg_top_words = tokenize_and_count(gutenberg_text, top_n=1000,remove_stopwords=False)
print_top_words(gutenberg_top_words, n=1000)
plot_word_frequencies(gutenberg_top_words[:300], 'Top 300 Most Frequent Words in Gutenberg Corpus')
plot_word_frequencies(gutenberg_top_words[:50], 'Top 50 Most Frequent Words in Gutenberg Corpus')
plot_zipf_law(gutenberg_top_words, 'Gutenberg Corpus', n_words=1000)
