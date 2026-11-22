import random
import streamlit as st
from grammar_checker import init_nlp_resources, grammar_check_text

st.set_page_config(page_title="NLP Grammar AutoCorrector", page_icon="✍️", layout="centered")

@st.cache_resource
def setup_nlp():
    init_nlp_resources()
    return True

setup_nlp()

def generate_unique_test_cases(count: int = 10) -> list[dict]:
    """Dynamically builds 10 unique grammar test cases on every click."""
    names = ["She", "He"]
    plural_subjects = ["The students", "The teachers", "The engineers", "The developers"]
    singular_subjects = ["The student", "The teacher", "The engineer", "The developer"]
    an_words = ["apple", "orange", "umbrella", "hour", "honest person", "NLP model"]
    a_words = ["university", "book", "project", "user", "unique idea", "computer"]
    comparatives = ["better", "faster", "easier", "smarter", "higher", "more"]
    intensifiers = ["much", "late", "fast", "hard", "cold", "tired"]
    past_pairs = [("went", "go"), ("ate", "eat"), ("saw", "see"), ("wrote", "write"), ("made", "make")]
    verb_pairs = [("go", "goes"), ("work", "works"), ("study", "studies"), ("play", "plays"), ("code", "codes")]

    generators = [
        lambda: (
            "Article Usage (a → an)",
            f"I bought a {w} yesterday.",
            f"I bought an {w} yesterday."
        ) if (w := random.choice(an_words)) else None,

        lambda: (
            "Article Usage (an → a)",
            f"This is an {w} for college.",
            f"This is a {w} for college."
        ) if (w := random.choice(a_words)) else None,

        lambda: (
            "Subject-Verb Agreement (Pronoun)",
            f"{s} {v[0]} to the lab daily.",
            f"{s} {v[1]} to the lab daily."
        ) if (s := random.choice(names)) and (v := random.choice(verb_pairs)) else None,

        lambda: (
            "Subject-Verb Agreement (Plural Noun)",
            f"{s} is working on the project.",
            f"{s} are working on the project."
        ) if (s := random.choice(plural_subjects)) else None,

        lambda: (
            "Subject-Verb Agreement (Singular Noun)",
            f"{s} are reading the report.",
            f"{s} is reading the report."
        ) if (s := random.choice(singular_subjects)) else None,

        lambda: (
            "Auxiliary Verb (of → have)",
            f"You {m} of finished the assignment.",
            f"You {m} have finished the assignment."
        ) if (m := random.choice(["should", "could", "would"])) else None,

        lambda: (
            "Verb Tense (did + past)",
            f"He did not {p[0]} to class.",
            f"He did not {p[1]} to class."
        ) if (p := random.choice(past_pairs)) else None,

        lambda: (
            "Contextual Word (then → than)",
            f"Python is {c} then manual checking.",
            f"Python is {c} than manual checking."
        ) if (c := random.choice(comparatives)) else None,

        lambda: (
            "Contextual Word (to → too)",
            f"This task is to {adj} for today.",
            f"This task is too {adj} for today."
        ) if (adj := random.choice(intensifiers)) else None,

        lambda: (
            "Contextual Word (their → there)",
            f"Their {v} a new update available.",
            f"There {v} a new update available."
        ) if (v := random.choice(["is", "was"])) else None,

        lambda: (
            "Contextual Word (your → you're)",
            f"I think your {adj} about this.",
            f"I think you're {adj} about this."
        ) if (adj := random.choice(["right", "wrong", "ready", "happy"])) else None,

        lambda: (
            "Duplicate Word",
            f"This is {w} {w} best NLP project.",
            f"This is {w} best NLP project."
        ) if (w := random.choice(["the", "my", "our"])) else None,

        lambda: (
            "Capitalization & Spacing",
            f"i am learning {t} ?",
            f"I am learning {t}?"
        ) if (t := random.choice(["Python", "grammar", "NLP", "AI"])) else None,
    ]

    random.shuffle(generators)
    cases = []
    for idx, gen in enumerate(generators[:count], start=1):
        category, inp, exp = gen()
        cases.append({
            "Test #": idx,
            "Grammar Rule": category,
            "Input": inp,
            "Expected": exp
        })
    return cases

# --- WEB INTERFACE ---
st.title("✍️ NLP Grammar AutoCorrector")
st.markdown("Built using **Regex Tokenization**, **NLTK POS Tagging**, and **Contextual Grammar Rules**.")

tab1, tab2 = st.tabs(["🔍 Grammar AutoCorrector", "🧪 Live Dynamic Test Suite (10 Cases)"])

# TAB 1: Interactive Grammar Checker
with tab1:
    default_text = "i is learning NLP at an university. He go to class and their is to much homework ?"
    user_input = st.text_area("Enter a sentence or paragraph to check:", value=default_text, height=130)

    if st.button("Check Grammar", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing POS tags and grammar rules..."):
                corrected_text, corrections = grammar_check_text(user_input)

            st.subheader("✅ Corrected Text")
            st.success(corrected_text)

            st.subheader("🔍 Grammar Corrections Breakdown")
            if corrections:
                st.table(corrections)
            else:
                st.info("No grammatical mistakes found!")

# TAB 2: Dynamic 10 Unique Test Cases
with tab2:
    st.markdown("Click the button below to generate **10 brand-new, unique grammar test cases** and run them live.")

    if st.button("Generate & Run 10 Unique Test Cases", type="primary"):
        with st.spinner("Generating unique grammar test cases..."):
            cases = generate_unique_test_cases(count=10)
            results = []
            passed_count = 0

            for case in cases:
                actual_output, _ = grammar_check_text(case["Input"])
                is_pass = actual_output == case["Expected"]
                if is_pass:
                    passed_count += 1

                results.append({
                    "Test #": case["Test #"],
                    "Grammar Rule": case["Grammar Rule"],
                    "Input Sentence": case["Input"],
                    "Expected Output": case["Expected"],
                    "Actual Output": actual_output,
                    "Status": "✅ PASS" if is_pass else "❌ FAIL"
                })

        if passed_count == len(cases):
            st.success(f"All {passed_count} / {len(cases)} unique test cases passed!")
        else:
            st.error(f"{passed_count} / {len(cases)} test cases passed.")

        st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()
st.caption("NLP Project | Powered by Python, NLTK POS Tagging & Streamlit")