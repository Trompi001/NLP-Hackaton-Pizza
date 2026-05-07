from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import lmstudio as lms
from rdflib import Graph


MODEL_NAME = "qwen3-14b"

TEST_SET_FILE = "dev-test-set/test_set_public.json"
OUTPUT_FILE = "submission.json"

SCHEMA_FILES = {
    "super": "knowledge-graphs/schema_superheroes.txt",
    "recipe": "knowledge-graphs/schema_recipes.txt",
}

GRAPH_FILES = {
    "superhero_universe": "knowledge-graphs/superhero_universe.ttl",
    "recipes_100": "knowledge-graphs/recipes_100.ttl",
}

_model = None
GRAPH_CACHE: dict[str, Graph] = {}


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

    text = (
        question_obj.get("question")
        or question_obj.get("question_de")
        or question_obj.get("text")
        or ""
    ).lower()

    if any(w in text for w in ["recipe", "ingredient", "cook", "cuisine", "rezept", "zutat", "küche"]):
        return SCHEMA_FILES["recipe"]

    return SCHEMA_FILES["super"]


def extract_sparql(text: str) -> str:
    match = re.search(r"```(?:sparql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)

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
    schema_path = Path(detect_schema_file(question_obj))
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema-Datei nicht gefunden: {schema_path.resolve()}")

    schema_text = schema_path.read_text(encoding="utf-8")

    question = (
        question_obj.get("question_de")
        or question_obj.get("question")
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

    response = get_model().respond(prompt)
    return extract_sparql(str(response))


def load_graph(graph_name: str) -> Graph:
    if graph_name not in GRAPH_CACHE:
        path = GRAPH_FILES[graph_name]
        g = Graph()
        g.parse(path, format="turtle")
        GRAPH_CACHE[graph_name] = g

    return GRAPH_CACHE[graph_name]


def run_query(graph: Graph, query: str) -> list[dict[str, Any]]:
    results = graph.query(query)

    if results.type == "ASK":
        return [{"answer": bool(results)}]

    var_names = [str(v) for v in results.vars]
    rows: list[dict[str, Any]] = []

    for row in results:
        row_dict: dict[str, Any] = {}
        for var_name, cell in zip(var_names, row):
            value = cell.toPython() if hasattr(cell, "toPython") else str(cell)
            if not isinstance(value, (str, int, float, bool)):
                value = str(value)
            row_dict[var_name] = value
        rows.append(row_dict)

    rows.sort(key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    return rows


def solve_question(question_obj: dict[str, Any]) -> dict[str, Any]:
    graph = load_graph(question_obj["graph"])
    sparql = generate_sparql(question_obj).strip()

    execution_success = False
    predicted_result: list[dict[str, Any]] = []

    if sparql:
        try:
            predicted_result = run_query(graph, sparql)
            execution_success = True
        except Exception as e:
            print(f"Query failed for {question_obj['id']}: {e}")
            print(sparql)

    return {
        "id": question_obj["id"],
        "graph": question_obj["graph"],
        "difficulty": question_obj.get("difficulty", ""),
        "question_de": question_obj["question_de"],
        "generated_sparql": sparql,
        "execution_success": execution_success,
        "predicted_result": predicted_result,
    }


def main() -> None:
    test_set = json.loads(Path(TEST_SET_FILE).read_text(encoding="utf-8"))

    submissions = []
    for question_obj in test_set:
        print(f"Solving {question_obj['id']}...")
        submissions.append(solve_question(question_obj))

    out_path = Path(OUTPUT_FILE)
    out_path.write_text(
        json.dumps(submissions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Created {out_path.resolve()}")


if __name__ == "__main__":
    main()