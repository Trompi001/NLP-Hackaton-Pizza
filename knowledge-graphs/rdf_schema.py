from rdflib import Graph, RDF, Literal
from collections import defaultdict

TTL_PATH = "hackathon-20260507/knowledge-graphs/superhero_universe.ttl"
OUTPUT_PATH = "knowledge-graphs/schema_superheroes.txt"

g = Graph()
g.parse(TTL_PATH, format="turtle")

def shorten(x):
    try:
        return g.namespace_manager.normalizeUri(x)
    except Exception:
        return str(x)

classes = set()
properties = defaultdict(lambda: {"subjects": set(), "objects": set()})

for s, p, o in g:
    if p == RDF.type:
        classes.add(o)
    else:
        properties[p]["subjects"].add(s)
        properties[p]["objects"].add(o)

def get_types(node):
    return set(g.objects(node, RDF.type))

lines = []
lines.append("SCHEMA SUMMARY\n")

# Classes
lines.append("Classes:")
for c in sorted(classes, key=str):
    lines.append(f"- {shorten(c)}")

# Properties
lines.append("\nProperties:")
for p, vals in sorted(properties.items(), key=lambda x: str(x[0])):
    subj_types = set()
    obj_types = set()
    obj_literals = False

    for s in vals["subjects"]:
        subj_types |= get_types(s)

    for o in vals["objects"]:
        if isinstance(o, Literal):
            obj_literals = True
        else:
            obj_types |= get_types(o)

    subj = ", ".join(shorten(t) for t in sorted(subj_types, key=str)) or "unknown"
    obj = "Literal" if obj_literals else ", ".join(shorten(t) for t in sorted(obj_types, key=str)) or "unknown"

    lines.append(f"- {shorten(p)}: {subj} -> {obj}")

# 👉 Prompt-Hinweis direkt hinzufügen (sehr hilfreich!)
lines.append("\nInstructions for LLM:")
lines.append("Use ONLY the provided schema.")
lines.append("Do NOT invent properties or classes.")
lines.append("Use schema:name for readable labels when available.")

# Write to file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Schema written to {OUTPUT_PATH}")