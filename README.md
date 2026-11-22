# NLP Grammar AutoCorrector

An interactive web-based Grammar AutoCorrector built in Python using Natural Language Processing: Regex Tokenization, NLTK Part-of-Speech (POS) Tagging, and Contextual Syntactic Rules.

## Features
- **Subject-Verb Agreement:** Uses NLTK POS tagging (`NN` vs `NNS`) and pronoun rules.
- **Article Agreement:** Phonetic `a` / `an` rules with English exceptions (e.g., *an hour*, *a university*).
- **Contextual Disambiguation:** Fixes commonly confused words (`then`/`than`, `to`/`too`, `your`/`you're`, `their`/`there`).
- **Live Test Suite:** Built-in dynamic test runner that generates 10 unique test cases on every run.# grammar-auto-corrector
