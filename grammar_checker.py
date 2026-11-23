import math
import random
import re
from collections import Counter
import nltk
from nltk.corpus import brown, wordnet as wn
from nltk.wsd import lesk

# Confusion sets learned purely from the Brown Corpus using Supervised Naive Bayes
CONFUSION_SETS = {
    "Article (a / an)": {"a", "an"},
    "Present Be (am / is / are)": {"am", "is", "are"},
    "Past Be (was / were)": {"was", "were"},
    "Present Have (has / have)": {"has", "have"},
    "Present Do (do / does)": {"do", "does"},
    "Comparison (then / than)": {"then", "than"},
    "Preposition/Degree (to / too)": {"to", "too"},
    "Existential/Possessive (their / there)": {"their", "there"},
}


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement


def extract_features(tagged_sent: list[tuple[str, str]], idx: int, task_name: str) -> dict:
    """
    Unit 2 & 3: Feature Engineering.
    Extracts contextual, POS-tag, and morphological features around index `idx`.
    """
    prev_word, prev_pos = tagged_sent[idx - 1] if idx > 0 else ("<START>", "<START>")
    prev2_word, prev2_pos = tagged_sent[idx - 2] if idx > 1 else ("<START>", "<START>")
    next_word, next_pos = tagged_sent[idx + 1] if idx < len(tagged_sent) - 1 else ("<END>", "<END>")
    next2_word, next2_pos = tagged_sent[idx + 2] if idx < len(tagged_sent) - 2 else ("<END>", "<END>")

    lw_prev = prev_word.lower()
    lw_next = next_word.lower()

    features = {
        "prev_word": lw_prev,
        "prev_pos": prev_pos,
        "prev2_pos": prev2_pos,
        "next_word": lw_next,
        "next_pos": next_pos,
        "next2_pos": next2_pos,
        "prev_bigram_pos": f"{prev2_pos}_{prev_pos}",
        "surrounding_pos": f"{prev_pos}_{next_pos}",
    }

    # For article classification, include prefix features so Naive Bayes learns
    # vowel/consonant and phonetic patterns (e.g., 'uni', 'hou') directly from data
    if task_name == "Article (a / an)":
        features["next_first1"] = lw_next[:1]
        features["next_first2"] = lw_next[:2]
        features["next_first3"] = lw_next[:3]

    return features


class StatisticalGrammarModel:
    """
    Pure ML & Statistical NLP Grammar AutoCorrector:
    - Unit 1: Brown Corpus + Regex Tokenization + WordNet
    - Unit 2: N-Gram Backoff POS Tagger + Supervised Naive Bayes Classifiers + Evaluation
    - Unit 3: Feature Engineering + Cosine Similarity
    - Unit 4: WordNet Synsets + Lesk Word Sense Disambiguation
    """

    def __init__(self):
        nltk.download("brown", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        # Load tagged sentences from Brown Corpus (Unit 1)
        tagged_sents = list(brown.tagged_sents())

        # Train N-Gram Backoff POS Tagger (Unit 2: Automatic & N-Gram Tagging)
        tagger_train = tagged_sents[:25000]
        tagger_test = tagged_sents[25000:27000]

        default_tagger = nltk.DefaultTagger("NN")
        unigram_tagger = nltk.UnigramTagger(tagger_train, backoff=default_tagger)
        self.pos_tagger = nltk.BigramTagger(tagger_train, backoff=unigram_tagger)
        self.tagger_accuracy = self.pos_tagger.accuracy(tagger_test)

        # Train Supervised Naive Bayes Classifiers for each confusion set (Unit 2)
        self.classifiers = {}
        self.eval_metrics = []
        self.word_to_task = {}

        for task_name, word_set in CONFUSION_SETS.items():
            for w in word_set:
                self.word_to_task[w] = task_name

        # Collect training examples per confusion set from Brown Corpus
        datasets = {task: [] for task in CONFUSION_SETS}
        max_samples_per_task = 2500

        for sent in tagged_sents:
            for i, (word, _) in enumerate(sent):
                lw = word.lower()
                task = self.word_to_task.get(lw)
                if task and len(datasets[task]) < max_samples_per_task:
                    feats = extract_features(sent, i, task)
                    datasets[task].append((feats, lw))

        rng = random.Random(42)
        for task_name, samples in datasets.items():
            rng.shuffle(samples)
            split_idx = int(len(samples) * 0.85)
            train_set = samples[:split_idx]
            test_set = samples[split_idx:]

            clf = nltk.NaiveBayesClassifier.train(train_set)
            acc = nltk.classify.accuracy(clf, test_set) if test_set else 0.0

            self.classifiers[task_name] = clf
            self.eval_metrics.append({
                "Classifier Task": task_name,
                "Training Samples": len(train_set),
                "Test Samples": len(test_set),
                "Naive Bayes Accuracy": f"{acc * 100:.2f}%",
            })

    def compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """Unit 3: Computes Cosine Similarity between two texts using term frequency vectors."""
        v1 = Counter(re.findall(r"[A-Za-z']+", text1.lower()))
        v2 = Counter(re.findall(r"[A-Za-z']+", text2.lower()))
        common = set(v1.keys()) & set(v2.keys())
        dot_product = sum(v1[w] * v2[w] for w in common)
        mag1 = math.sqrt(sum(c * c for c in v1.values()))
        mag2 = math.sqrt(sum(c * c for c in v2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def analyze_semantics_wsd(self, text: str) -> list[dict]:
        """Unit 4: Word Sense Disambiguation (Lesk) & WordNet Synsets/Hypernyms."""
        tokens = re.findall(r"[A-Za-z]+", text)
        tagged = self.pos_tagger.tag(tokens)
        results = []
        seen = set()

        for word, pos in tagged:
            lw = word.lower()
            if lw in seen or len(lw) < 4:
                continue
            # Focus on content words (nouns, verbs, adjectives)
            if not pos.startswith(("NN", "VB", "JJ")):
                continue
            seen.add(lw)

            wn_pos = wn.NOUN if pos.startswith("NN") else (wn.VERB if pos.startswith("VB") else wn.ADJ)
            synset = lesk(tokens, lw, pos=wn_pos)
            if synset:
                synonyms = sorted({lemma.name().replace("_", " ") for lemma in synset.lemmas()})[:4]
                hypernyms = [h.lemmas()[0].name().replace("_", " ") for h in synset.hypernyms()][:3]
                results.append({
                    "Word": word,
                    "POS Tag": pos,
                    "Disambiguated Synset": synset.name(),
                    "Definition": synset.definition(),
                    "Synonyms": ", ".join(synonyms) if synonyms else "None",
                    "Hypernyms": ", ".join(hypernyms) if hypernyms else "None",
                })
            if len(results) >= 6:
                break

        return results

    def correct_text(self, text: str, confidence_threshold: float = 0.72) -> tuple[str, list[dict], list[tuple[str, str]]]:
        """
        Corrects grammar using N-Gram POS Tagging + Naive Bayes Posterior Probabilities.
        No hardcoded grammar rules are used.
        """
        if not text.strip():
            return text, [], []

        # Unit 1: Regex Tokenization (separates words from punctuation/spaces)
        tokens = re.findall(r"[A-Za-z']+|\s+|[^A-Za-z'\s]+", text)
        word_indices = [i for i, t in enumerate(tokens) if re.fullmatch(r"[A-Za-z']+", t)]
        words_only = [tokens[i] for i in word_indices]

        # Unit 2: N-Gram Backoff POS Tagging
        tagged_words = self.pos_tagger.tag(words_only)
        corrections = []

        for idx, (word, _) in enumerate(tagged_words):
            lw = word.lower()
            task = self.word_to_task.get(lw)
            if not task:
                continue

            clf = self.classifiers[task]
            feats = extract_features(tagged_words, idx, task)
            prob_dist = clf.prob_classify(feats)
            best_label = prob_dist.max()
            best_prob = prob_dist.prob(best_label)
            orig_prob = prob_dist.prob(lw)

            # Apply correction only if Naive Bayes predicts a different word with high confidence
            if best_label != lw and best_prob >= confidence_threshold and best_prob > orig_prob:
                fixed_word = _match_case(word, best_label)
                token_pos = word_indices[idx]
                tokens[token_pos] = fixed_word

                # Update tagged_words so subsequent tokens see the corrected context
                new_tag = self.pos_tagger.tag([fixed_word])[0][1]
                tagged_words[idx] = (fixed_word, new_tag)

                corrections.append({
                    "Original": word,
                    "Corrected": fixed_word,
                    "Classifier (ML Model)": task,
                    "Confidence": f"{best_prob * 100:.1f}%",
                })

        corrected_text = "".join(tokens)
        return corrected_text, corrections, tagged_words
