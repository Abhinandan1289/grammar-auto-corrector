import re
import nltk

def init_nlp_resources():
    """Downloads required NLTK POS tagger resources quietly."""
    for resource in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

# Phonetic exceptions for a / an article rules
AN_EXCEPTIONS = ("hour", "honest", "honor", "honour", "heir", "nlp", "mba", "html", "sql")
A_EXCEPTIONS = ("university", "user", "uniform", "unique", "unit", "european", "one", "useful", "usual", "union")

# Common base verbs for 3rd-person singular agreement (e.g., "He go" -> "He goes")
THIRD_PERSON_VERBS = {
    "go": "goes", "do": "does", "have": "has", "like": "likes", "play": "plays",
    "work": "works", "study": "studies", "eat": "eats", "run": "runs", "need": "needs",
    "want": "wants", "make": "makes", "take": "takes", "know": "knows", "think": "thinks",
    "write": "writes", "read": "reads", "walk": "walks", "talk": "talks", "learn": "learns",
    "code": "codes", "build": "builds", "love": "loves", "live": "lives", "speak": "speaks"
}

# Irregular past -> base form after modals/auxiliaries (e.g., "did not went" -> "did not go")
PAST_TO_BASE = {
    "went": "go", "ate": "eat", "saw": "see", "did": "do", "came": "come",
    "took": "take", "made": "make", "wrote": "write", "spoke": "speak",
    "bought": "buy", "brought": "bring", "thought": "think", "knew": "know",
    "ran": "run", "gave": "give", "told": "tell", "felt": "feel", "kept": "keep"
}

def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement

def _needs_an(next_word: str) -> bool:
    lw = next_word.lower()
    if lw.startswith(AN_EXCEPTIONS):
        return True
    if lw.startswith(A_EXCEPTIONS):
        return False
    return lw[0] in "aeiou"

def grammar_check_text(text: str) -> tuple[str, list[dict]]:
    """
    Analyzes input text and fixes grammar, agreement, context,
    capitalization, and punctuation errors.
    Returns (corrected_text, list_of_corrections).
    """
    if not text.strip():
        return text, []

    corrected = text
    corrections = []

    def record(original: str, fixed: str, category: str, reason: str):
        if original != fixed:
            corrections.append({
                "Original": original,
                "Corrected": fixed,
                "Category": category,
                "Reason": reason
            })

    # 1. Remove consecutive duplicate words (e.g., "the the" -> "the")
    def fix_duplicates(match):
        full = match.group(0)
        first = match.group(1)
        record(full, first, "Duplicate Word", "Removed repeated consecutive word.")
        return first

    corrected = re.sub(r"\b([A-Za-z]+)\s+\1\b", fix_duplicates, corrected, flags=re.IGNORECASE)

    # 2. Modal / Auxiliary + "of" (e.g., "should of" -> "should have")
    def fix_modal_of(match):
        modal = match.group(1)
        fixed = f"{modal} have"
        record(match.group(0), fixed, "Auxiliary Verb", f"Use '{modal} have' instead of '{modal} of'.")
        return fixed

    corrected = re.sub(r"\b(should|could|would|must|might)\s+of\b", fix_modal_of, corrected, flags=re.IGNORECASE)

    # 3. Auxiliary / Modal (+ optional "not") + Past Tense Verb (e.g., "did not went" -> "did not go")
    past_pattern = "|".join(PAST_TO_BASE.keys())
    def fix_aux_past(match):
        aux, verb = match.group(1), match.group(2)
        base = _match_case(verb, PAST_TO_BASE[verb.lower()])
        fixed = f"{aux} {base}"
        record(match.group(0), fixed, "Verb Tense", f"Use base verb '{base}' after '{aux}'.")
        return fixed

    corrected = re.sub(
        rf"\b((?:did|didn't|does|doesn't|can|cannot|can't|will|won't|should|shouldn't|could|would)(?:\s+not)?)\s+({past_pattern})\b",
        fix_aux_past,
        corrected,
        flags=re.IGNORECASE
    )

    # 4. Subject-Verb Agreement (Pronouns)
    pronoun_rules = [
        (r"\b(I)\s+(is|are)\b", "am", "Use 'am' with pronoun 'I'."),
        (r"\b(I)\s+(were)\b", "was", "Use 'was' with pronoun 'I'."),
        (r"\b(I)\s+(has)\b", "have", "Use 'have' with pronoun 'I'."),
        (r"\b(I)\s+(doesn't)\b", "don't", "Use 'don't' with pronoun 'I'."),
        (r"\b(he|she|it)\s+(are)\b", "is", "Use singular verb 'is' with singular pronoun."),
        (r"\b(he|she|it)\s+(were)\b", "was", "Use singular verb 'was' with singular pronoun."),
        (r"\b(he|she|it)\s+(have)\b", "has", "Use singular verb 'has' with singular pronoun."),
        (r"\b(he|she|it)\s+(don't)\b", "doesn't", "Use 'doesn't' with he/she/it."),
        (r"\b(we|they|you)\s+(is)\b", "are", "Use plural verb 'are' with we/they/you."),
        (r"\b(we|they|you)\s+(was)\b", "were", "Use plural verb 'were' with we/they/you."),
        (r"\b(we|they|you)\s+(has)\b", "have", "Use plural verb 'have' with we/they/you."),
        (r"\b(we|they|you)\s+(doesn't)\b", "don't", "Use 'don't' with we/they/you."),
    ]
    for pattern, replacement, reason in pronoun_rules:
        def _sub_pronoun(m, rep=replacement, rsn=reason):
            subj, verb = m.group(1), m.group(2)
            new_verb = _match_case(verb, rep)
            fixed = f"{subj} {new_verb}"
            record(m.group(0), fixed, "Subject-Verb Agreement", rsn)
            return fixed
        corrected = re.sub(pattern, _sub_pronoun, corrected, flags=re.IGNORECASE)

    # 3rd-person singular base verbs (e.g., "She go" -> "She goes")
    base_verb_pattern = "|".join(THIRD_PERSON_VERBS.keys())
    def fix_third_person(match):
        subj, verb = match.group(1), match.group(2)
        new_verb = _match_case(verb, THIRD_PERSON_VERBS[verb.lower()])
        fixed = f"{subj} {new_verb}"
        record(match.group(0), fixed, "Subject-Verb Agreement", f"Singular subject '{subj}' takes '{new_verb}'.")
        return fixed

    corrected = re.sub(
        rf"\b(he|she|it)\s+({base_verb_pattern})\b",
        fix_third_person,
        corrected,
        flags=re.IGNORECASE
    )

    # 5. Subject-Verb Agreement for Nouns using NLTK POS Tagging (e.g., "The students is" -> "The students are")
    tokens = re.findall(r"[A-Za-z']+|\s+|[^A-Za-z'\s]+", corrected)
    word_indices = [i for i, t in enumerate(tokens) if re.fullmatch(r"[A-Za-z']+", t)]
    words_only = [tokens[i] for i in word_indices]

    if words_only:
        try:
            pos_tags = nltk.pos_tag(words_only)
            for idx in range(len(pos_tags) - 1):
                w1, tag1 = pos_tags[idx]
                w2, _ = pos_tags[idx + 1]
                lw2 = w2.lower()

                t_idx1, t_idx2 = word_indices[idx], word_indices[idx + 1]
                between = "".join(tokens[t_idx1 + 1:t_idx2])
                if between.strip() != "":
                    continue

                if tag1 == "NNS" and lw2 in ("is", "was", "has"):
                    rep_map = {"is": "are", "was": "were", "has": "have"}
                    new_w2 = _match_case(w2, rep_map[lw2])
                    record(f"{w1} {w2}", f"{w1} {new_w2}", "Subject-Verb Agreement", f"Plural noun '{w1}' requires '{new_w2}'.")
                    tokens[t_idx2] = new_w2
                elif tag1 == "NN" and lw2 in ("are", "were"):
                    rep_map = {"are": "is", "were": "was"}
                    new_w2 = _match_case(w2, rep_map[lw2])
                    record(f"{w1} {w2}", f"{w1} {new_w2}", "Subject-Verb Agreement", f"Singular noun '{w1}' requires '{new_w2}'.")
                    tokens[t_idx2] = new_w2
            corrected = "".join(tokens)
        except LookupError:
            pass

    # 6. Indefinite Articles ("a" vs "an")
    def fix_article(match):
        art, next_word = match.group(1), match.group(2)
        should_be_an = _needs_an(next_word)
        expected = _match_case(art, "an" if should_be_an else "a")
        if expected.lower() != art.lower():
            fixed = f"{expected} {next_word}"
            record(match.group(0), fixed, "Article Usage", f"Use '{expected}' before '{next_word}'.")
            return fixed
        return match.group(0)

    corrected = re.sub(r"\b(a|an)\s+([A-Za-z]+)\b", fix_article, corrected, flags=re.IGNORECASE)

    # 7. Contextual Homophones (then/than, to/too, your/you're, their/there)
    context_rules = [
        (
            r"\b(better|more|less|bigger|smaller|faster|slower|older|younger|higher|lower|smarter|easier|harder|longer|shorter)\s+then\b",
            lambda m: f"{m.group(1)} than",
            "Contextual Word",
            "Use 'than' for comparisons instead of 'then'."
        ),
        (
            r"\bto\s+(much|many|late|fast|slow|hard|easy|cold|hot|tired|big|small|loud|expensive)\b",
            lambda m: f"too {m.group(1)}",
            "Contextual Word",
            "Use 'too' to mean 'excessively'."
        ),
        (
            r"\b(your)\s+(welcome|right|wrong|going|doing|learning|working|ready|happy|late)\b",
            lambda m: _match_case(m.group(1), "you're") + " " + m.group(2),
            "Contextual Word",
            "Use 'you're' (you are) instead of possessive 'your'."
        ),
        (
            r"\b(their)\s+(is|are|was|were)\b",
            lambda m: f"{_match_case(m.group(1), 'there')} {m.group(2)}",
            "Contextual Word",
            "Use 'there' instead of possessive 'their' before a verb."
        ),
    ]
    for pattern, repl_fn, cat, reason in context_rules:
        def _ctx_sub(m, fn=repl_fn, c=cat, r=reason):
            fixed = fn(m)
            record(m.group(0), fixed, c, r)
            return fixed
        corrected = re.sub(pattern, _ctx_sub, corrected, flags=re.IGNORECASE)

    # 8. Standalone pronoun "i" capitalization
    def fix_pronoun_i(match):
        orig = match.group(0)
        fixed = "I" + orig[1:]
        if orig != fixed:
            record(orig, fixed, "Capitalization", "Always capitalize the pronoun 'I'.")
        return fixed

    corrected = re.sub(r"\bi(?:'m|'ve|'ll|'d)?\b", fix_pronoun_i, corrected)

    # 9. Punctuation spacing & extra whitespace
    spaced = re.sub(r"\s+([,.?!])", r"\1", corrected)
    spaced = re.sub(r"[ \t]{2,}", " ", spaced).strip()
    if spaced != corrected.strip():
        record(corrected.strip(), spaced, "Punctuation & Spacing", "Removed extra spaces before punctuation.")
    corrected = spaced

    # 10. Capitalize first letter of each sentence
    def cap_sentence(match):
        prefix = match.group(1)
        letter = match.group(2)
        if letter.islower():
            record(letter, letter.upper(), "Capitalization", "Capitalize the first letter of a sentence.")
            return prefix + letter.upper()
        return match.group(0)

    corrected = re.sub(r"(^|[.!?]\s+)([a-z])", cap_sentence, corrected)

    return corrected, corrections