import random
import streamlit as st
from grammar_checker import StatisticalGrammarModel

st.set_page_config(page_title="ML NLP Grammar AutoCorrector", page_icon="🧠", layout="centered")


@st.cache_resource
def load_model():
    return StatisticalGrammarModel()


with st.spinner("Training N-Gram POS Tagger & Naive Bayes Classifiers on NLTK Brown Corpus..."):
    model = load_model()


def generate_unique_test_cases(count: int = 10) -> list[dict]:
    """Dynamically generates 10 unique randomized test cases for the ML classifiers."""
    an_words = ["apple", "orange", "hour", "idea", "honest man", "umbrella"]
    a_words = ["university", "book", "computer", "project", "user", "house"]
    plural_nouns = ["The students", "The teachers", "The engineers", "The players"]
    singular_nouns = ["The student", "The teacher", "The engineer", "The player"]
    comparatives = ["better", "faster", "more", "higher", "larger", "older"]
    adjectives = ["late", "hard", "cold", "much", "tired", "fast"]

    generators = [
        lambda: (
            "Naive Bayes: Article (a → an)",
            f"I found a {w} yesterday.",
            f"I found an {w} yesterday.",
        ) if (w := random.choice(an_words)) else None,

        lambda: (
            "Naive Bayes: Article (an → a)",
            f"This is an {w} for class.",
            f"This is a {w} for class.",
        ) if (w := random.choice(a_words)) else None,

        lambda: (
            "Naive Bayes: Present Be (I is → I am)",
            f"I is reading a {w} now.",
            f"I am reading a {w} now.",
        ) if (w := random.choice(["book", "report", "paper", "story"])) else None,

        lambda: (
            "Naive Bayes: Present Be (Plural NNS)",
            f"{s} is working on the task.",
            f"{s} are working on the task.",
        ) if (s := random.choice(plural_nouns)) else None,

        lambda: (
            "Naive Bayes: Present Be (Singular NN)",
            f"{s} are reading the book.",
            f"{s} is reading the book.",
        ) if (s := random.choice(singular_nouns)) else None,

        lambda: (
            "Naive Bayes: Past Be (Plural)",
            f"They was ready for the {w}.",
            f"They were ready for the {w}.",
        ) if (w := random.choice(["exam", "game", "meeting", "trip"])) else None,

        lambda: (
            "Naive Bayes: Present Have (Singular)",
            f"He have a new {w}.",
            f"He has a new {w}.",
        ) if (w := random.choice(["car", "book", "job", "plan"])) else None,

        lambda: (
            "Naive Bayes: Present Do (Singular)",
            f"She do not like the {w}.",
            f"She does not like the {w}.",
        ) if (w := random.choice(["rain", "noise", "cold", "delay"])) else None,

        lambda: (
            "Naive Bayes: Comparison (then → than)",
            f"This method is {c} then the old one.",
            f"This method is {c} than the old one.",
        ) if (c := random.choice(comparatives)) else None,

        lambda: (
            "Naive Bayes: Degree (to → too)",
            f"The work is to {adj} for today.",
            f"The work is too {adj} for today.",
        ) if (adj := random.choice(adjectives)) else None,

        lambda: (
            "Naive Bayes: Existential (their → there)",
            f"Their {v} a big problem here.",
            f"There {v} a big problem here.",
        ) if (v := random.choice(["is", "was"])) else None,
    ]

    random.shuffle(generators)
    cases = []
    for idx, gen in enumerate(generators[:count], start=1):
        category, inp, exp = gen()
        cases.append({
            "Test #": idx,
            "ML Classifier": category,
            "Input": inp,
            "Expected": exp,
        })
    return cases


# --- WEB INTERFACE ---
st.title("🧠 ML & Statistical NLP Grammar AutoCorrector")
st.markdown(
    "Built without hardcoded grammar rules — powered by **Brown Corpus Statistics**, "
    "**N-Gram Backoff POS Tagging**, **Supervised Naive Bayes Classifiers**, "
    "**Cosine Similarity**, and **WordNet WSD**."
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Grammar AutoCorrector",
    "🧪 Live Dynamic Test Suite (10 Cases)",
    "📊 Model Evaluation ",
])

# TAB 1: Interactive ML Grammar Checker
with tab1:
    default_text = (
        "I is studying at an university for a hour. "
        "The students is working because their is to much work, and this plan is better then nothing."
    )
    user_input = st.text_area("Enter a sentence or paragraph to check:", value=default_text, height=130)

    if st.button("Check Grammar", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            corrected_text, corrections, tagged_words = model.correct_text(user_input)
            cos_sim = model.compute_cosine_similarity(user_input, corrected_text)

            st.subheader("✅ Corrected Text")
            st.success(corrected_text)

            col1, col2 = st.columns(2)
            col1.metric("Corrections Made", len(corrections))
            col2.metric("Cosine Similarity (Unit 3)", f"{cos_sim:.4f}")

            st.subheader("🤖 Naive Bayes Predictions Breakdown")
            if corrections:
                st.table(corrections)
            else:
                st.info("No grammatical errors predicted above the confidence threshold.")

            with st.expander("🏷️ N-Gram Backoff POS Tags (Unit 2)"):
                st.write(" ".join([f"`{w}/{t}`" for w, t in tagged_words]))

            with st.expander("📖 WordNet Semantic Analysis & Lesk WSD (Unit 4)"):
                wsd_results = model.analyze_semantics_wsd(corrected_text)
                if wsd_results:
                    st.dataframe(wsd_results, use_container_width=True, hide_index=True)
                else:
                    st.info("No content words available for WordNet disambiguation.")

# TAB 2: Dynamic 10 Random Test Cases
with tab2:
    st.markdown("Click below to generate **10 random, unique test cases** and evaluate the trained Naive Bayes models live.")

    if st.button("Generate & Run 10 Unique Test Cases", type="primary"):
        cases = generate_unique_test_cases(count=10)
        results = []
        passed_count = 0

        for case in cases:
            actual_output, _, _ = model.correct_text(case["Input"])
            is_pass = actual_output == case["Expected"]
            if is_pass:
                passed_count += 1

            results.append({
                "Test #": case["Test #"],
                "ML Classifier": case["ML Classifier"],
                "Input Sentence": case["Input"],
                "Expected Output": case["Expected"],
                "Actual Output": actual_output,
                "Status": "✅ PASS" if is_pass else "❌ FAIL",
            })

        if passed_count == len(cases):
            st.success(f"All {passed_count} / {len(cases)} unique test cases passed!")
        else:
            st.warning(f"{passed_count} / {len(cases)} test cases passed.")

        st.dataframe(results, use_container_width=True, hide_index=True)

# TAB 3: Real Evaluation Metrics on Held-Out Brown Corpus Test Split
with tab3:
    st.subheader("N-Gram Backoff POS Tagger Evaluation")
    st.metric("Bigram → Unigram → Default Tagger Accuracy", f"{model.tagger_accuracy * 100:.2f}%")

    st.subheader("Supervised Naive Bayes Classifiers (85% Train / 15% Test Split)")
    st.dataframe(model.eval_metrics, use_container_width=True, hide_index=True)

st.divider()
st.caption("NLP Project | Brown Corpus • N-Gram POS Tagging • Naive Bayes • Cosine Similarity • WordNet WSD")
