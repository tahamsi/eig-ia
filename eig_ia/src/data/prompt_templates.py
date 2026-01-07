ART_QUESTION_TEMPLATE = "Did {hypothesis} happen?"

ART_SCORE_PROMPT = (
    "Observation: {observation}\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Hypothesis: "
)

ART_PRIOR_PROMPT = (
    "Observation: {observation}\n"
    "Hypothesis: "
)

AMBIGQA_QGEN_PROMPT = (
    "Ambiguous question: {question}\n"
    "Possible intents:\n{rewrites}\n"
    "Ask a clarifying question to disambiguate the intent."
)

AMBIGQA_SCORE_PROMPT = (
    "Ambiguous question: {question}\n"
    "Clarifying question: {clarifying_question}\n"
    "User answer: {answer}\n"
    "Intended meaning: "
)

AMBIGQA_PRIOR_PROMPT = (
    "Ambiguous question: {question}\n"
    "Intended meaning: "
)
