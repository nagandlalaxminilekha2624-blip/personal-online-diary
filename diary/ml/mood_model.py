"""Multinomial Naive Bayes mood classifier, implemented in pure Python.

No external ML libraries are needed, so the project only depends on Django.

How it works
------------
1. Text is lower-cased and split into words; very common words are dropped.
   A negation such as "not" is glued to the next word ("not happy" becomes
   "not_happy") so negated feelings are not mistaken for positive ones.
2. For every mood we count how often each word appears in its training texts.
3. To classify new text we compute, for each mood,
       log P(mood) + sum( log P(word | mood) )
   using Laplace smoothing so unseen words never give a zero probability,
   then turn the scores into probabilities that sum to 1.
"""
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple, Optional

from .training_data import SEED_DATA

TOKEN_RE = re.compile(r"[a-z']+")
NEGATIONS = {"not", "no", "never", "cannot", "can't", "cant", "don't", "dont",
             "didn't", "didnt", "isn't", "isnt", "wasn't", "wasnt", "won't", "wont"}
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "am", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "myself", "we",
    "our", "you", "your", "he", "she", "they", "them", "his", "her", "their", "do",
    "did", "does", "have", "has", "had", "will", "would", "can", "could", "just",
    "very", "really", "too", "than", "then", "about", "into", "out", "up", "down",
    "all", "some", "any", "what", "when", "while", "also", "again", "one", "got",
    "get", "went", "go", "today", "day", "feel", "felt", "feeling",
}


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    negate_next = False
    for word in TOKEN_RE.findall(text.lower()):
        word = word.strip("'")
        if len(word) < 2:
            continue
        if word in NEGATIONS:
            negate_next = True
            continue
        if word in STOPWORDS:
            continue
        tokens.append(f"not_{word}" if negate_next else word)
        negate_next = False
    return tokens


class MoodPrediction(NamedTuple):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class NaiveBayesMoodClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_docs: Counter = Counter()
        self.word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.total_words: Counter = Counter()
        self.vocab: set = set()

    def fit(self, texts: List[str], labels: List[str]) -> "NaiveBayesMoodClassifier":
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.class_docs[label] += 1
            self.word_counts[label].update(tokens)
            self.total_words[label] += len(tokens)
            self.vocab.update(tokens)
        return self

    def predict(self, text: str) -> Optional[MoodPrediction]:
        """Return the most likely mood, or None if no word is recognised."""
        tokens = [t for t in tokenize(text) if t in self.vocab]
        if not tokens or not self.class_docs:
            return None

        total_docs = sum(self.class_docs.values())
        vocab_size = len(self.vocab)
        log_scores: Dict[str, float] = {}
        for label, doc_count in self.class_docs.items():
            score = math.log(doc_count / total_docs)
            denom = self.total_words[label] + self.alpha * vocab_size
            for token in tokens:
                score += math.log((self.word_counts[label][token] + self.alpha) / denom)
            log_scores[label] = score

        # Convert log scores to probabilities (log-sum-exp for stability).
        top = max(log_scores.values())
        exps = {label: math.exp(s - top) for label, s in log_scores.items()}
        norm = sum(exps.values())
        probs = {label: v / norm for label, v in exps.items()}
        best = max(probs, key=probs.get)
        return MoodPrediction(best, probs[best], probs)


def build_classifier(extra_texts=None, extra_labels=None) -> NaiveBayesMoodClassifier:
    """Train on the seed data plus any extra (user-labelled) examples."""
    texts, labels = [], []
    for label, sentences in SEED_DATA.items():
        texts.extend(sentences)
        labels.extend([label] * len(sentences))
    if extra_texts:
        texts.extend(extra_texts)
        labels.extend(extra_labels)
    return NaiveBayesMoodClassifier().fit(texts, labels)
