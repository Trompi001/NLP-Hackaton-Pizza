from typing import Any
from pathlib import Path
import re
import lmstudio as lms


MODEL_NAME = "qwen3-14b"  

SCHEMA_FILES = {
    "super": "knowledge-graphs/schema_superheroes.txt",
    "recipe": "knowledge-graphs/schema_recipes.txt",
}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = lms.llm(MODEL_NAME)
    return _model


def detect_schema_file(question_obj: dict[str, Any]) -> str:
    qid = question_obj.get("id", "").lower()

    if "super" in qid:
        return SCHEMA_FILES["super"]

    if "recipe" in qid or "recipes" in qid:
        return SCHEMA_FILES["recipe"]

    # Fallback über Fragetext
    text = question_obj.get("question", "").lower()
    if any(w in text for w in ["recipe", "ingredient", "cook", "cuisine"]):
        return SCHEMA_FILES["recipe"]

    return SCHEMA_FILES["super"]


def extract_sparql(text: str) -> str:
    # Entfernt Markdown-Codeblöcke
    match = re.search(r"```(?:sparql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)

    # Entfernt mögliche Erklärungen vor PREFIX/SELECT
    start_candidates = [
        text.find("PREFIX"),
        text.find("SELECT"),
        text.find("ASK"),
        text.find("CONSTRUCT"),
    ]
    starts = [x for x in start_candidates if x != -1]
    if starts:
        text = text[min(starts):]

    return text.strip()


def generate_sparql(question_obj: dict[str, Any]) -> str:
    """
    Generiert eine SPARQL-Query mit LM Studio + Qwen3-14B.
    Erwartet lokale Schema-Dateien für superheroes/recipes.
    """

    schema_path = Path(detect_schema_file(question_obj))
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema-Datei nicht gefunden: {schema_path.resolve()}")

    schema_text = schema_path.read_text(encoding="utf-8")

    question = (
        question_obj.get("question")
        or question_obj.get("text")
        or question_obj.get("query")
        or str(question_obj)
    )

    prompt = f"""
You are an expert SPARQL generator.

Generate exactly one valid SPARQL query for the given RDF graph.

Rules:
- Use ONLY the provided schema.
- Do NOT invent classes or properties.
- Use the listed prefixes exactly.
- Return ONLY the SPARQL query.
- Do not explain anything.
- Do not use Markdown.
- Prefer schema:name for human-readable names.
- If sorting is useful, add ORDER BY.
- If the question asks for a count, use COUNT.
- If the question asks yes/no, use ASK.

Schema:
{schema_text}

Question:
{question}
""".strip()

    model = get_model()
    response = model.respond(prompt)

    sparql = extract_sparql(str(response))

    return sparql