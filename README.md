# Adelmo - NLP Educational Project

An educational project for learning Natural Language Processing (NLP) fundamentals through hands-on text analysis and statistical language modeling.

## About the Name

The project is named after **Adelmo da Otranto**, a character from Umberto Eco's masterpiece *Il nome della rosa* (*The Name of the Rose*). In the novel, Adelmo is a young monk renowned as a gifted illuminator who adorned manuscripts in the abbey's library with beautiful illustrations. His mysterious death—falling from the east tower during a violent snowstorm—sets in motion the central mystery that the protagonist William of Baskerville must solve.

The choice of this name reflects the project's connection to textual analysis and the literary tradition, bridging medieval manuscript illumination with modern computational approaches to understanding language. Just as Adelmo brought beauty to written words through his art, this project aims to reveal the hidden patterns and structures within texts through statistical analysis.

## Overview

This project contains Python scripts demonstrating core NLP concepts including tokenization, word frequency analysis, n-gram models, and Zipf's law. The scripts analyze various text corpora to explore patterns in natural language.

## Project Structure

```
adelmo/
├── src/
│   ├── tokenization/
│   │   └── 01-simpleTokenization.py
│   └── ngrams/
│       └── 02-ngrams.py
├── data/
├── environment.yml
└── README.md
```

## Scripts

### `src/tokenization/zipfs-law.py`

A comprehensive script for text tokenization and frequency analysis with visualization capabilities, demonstrating Zipf's Law in natural language.

**Features:**
- **Text Tokenization**: Uses NLTK's word tokenizer to break text into individual tokens
- **Word Frequency Analysis**: Counts word occurrences with configurable filtering options
- **Stopword Removal**: Filters common English stopwords for more meaningful analysis
- **Visualization**: Creates bar charts showing the most frequent words
- **Zipf's Law Analysis**: Plots log-log graphs to demonstrate Zipf's law in natural language
- **Configurable Corpus Selection**: Select which corpus to analyze via `conf.txt` configuration file
- **Multiple Corpus Support**: Analyzes various texts including:
  - Gutenberg corpus (hundreds of texts)
  - I Promessi Sposi by Alessandro Manzoni
  - Infinite Jest by David Foster Wallace
  - Shakespeare corpus (from NLTK's Gutenberg collection)

**Configuration:**
Edit `src/tokenization/conf.txt` to select which corpus to process:
```
corpus=gutenberg
```

**Key Functions:**
- `tokenize_and_count()`: Tokenizes text and counts word frequencies
- `plot_word_frequencies()`: Visualizes word frequency distributions
- `plot_zipf_law()`: Demonstrates Zipf's law with fitted curves
- `load_gutenberg_corpus()`: Loads multiple text files from local directory
- `process_corpus()`: Processes the selected corpus based on configuration

**For detailed analysis results and methodology**, see [src/tokenization/zipfs-law.MD](src/tokenization/zipfs-law.MD)

### `src/ngrams/02-ngrams.py`

Implements statistical language models using n-grams on the Brown corpus.

**Features:**
- **1-gram Model (Unigrams)**: Builds frequency distribution of individual words
- **2-gram Model (Bigrams)**: Captures word pair statistics
- **3-gram Model (Trigrams)**: Models three-word sequences
- **Probability Calculation**: Converts n-gram counts to probability distributions
- **Brown Corpus Integration**: Uses NLTK's Brown corpus for training

**Key Functions:**
- `build_1gram_model()`: Creates unigram frequency model
- `build_2gram_model()`: Creates bigram frequency model
- `build_3gram_model()`: Creates trigram frequency model
- `calculate_probabilities()`: Converts counts to probabilities
- `load_brown_corpus()`: Loads sentences from the Brown corpus

## Environment Setup

This project uses Conda for dependency management. The environment includes Python 3.12 and various data science libraries.

### Creating the Conda Environment

To create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

### Activating the Environment

```bash
conda activate adelmo
```

### Updating the Environment

If dependencies are added or modified in `environment.yml`, update your existing environment:

```bash
conda env update -f environment.yml --prune
```

The `--prune` flag removes dependencies that are no longer in the environment file.

### Exporting the Environment

To export your current environment to update the `environment.yml` file:

```bash
conda env export > environment.yml
```

Or for a more minimal export (without build strings):

```bash
conda env export --no-builds > environment.yml
```

## Dependencies

The project includes the following main dependencies:

- **Python 3.12**: Core programming language
- **NLTK 3.9.1**: Natural Language Toolkit for text processing
- **NumPy 2.2.2**: Numerical computing
- **Pandas 2.2.3**: Data manipulation and analysis
- **Matplotlib 3.10.0**: Data visualization
- **Seaborn 0.13.2**: Statistical data visualization
- **Plotly 6.0.0**: Interactive visualizations
- **Jupyter Notebook 7.3.2**: Interactive development environment

## Getting Started

1. Clone the repository
2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate adelmo
   ```
3. Ensure you have the required data files in the `data/` directory
4. Run the scripts:
   ```bash
   python src/tokenization/zipfs-law.py
   python src/ngrams/02-ngrams.py
   ```

## NLTK Data Requirements

The scripts automatically download required NLTK data packages:
- `punkt`: Tokenizer models
- `stopwords`: Common stopwords for multiple languages
- `gutenberg`: Gutenberg corpus including Shakespeare texts
- `brown`: Brown corpus for n-gram modeling

**Note**: IDEs may show "Unresolved reference" warnings for NLTK imports (e.g., `word_tokenize`, `stopwords`, `gutenberg`). These warnings can be safely ignored as the NLTK data packages are downloaded at runtime when the scripts execute.

## Educational Goals

This project demonstrates:
- Text preprocessing and tokenization techniques
- Statistical analysis of natural language
- Zipf's law and its application to word frequency distributions
- N-gram language models and probability estimation
- Data visualization for linguistic analysis

## License

Educational use only.
