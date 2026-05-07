from rdflib import Graph

g = Graph()
g.parse("hackathon-20260507/knowledge-graphs/superhero_universe.ttl", format="turtle")

query = """
PREFIX ex: <http://example.org/>
PREFIX schema: <https://schema.org/>

SELECT ?name ?teamName
WHERE {
  ?r a schema:Person ;
     schema:name ?name ;
     ex:memberOf ?team .

  ?team schema:name ?teamName .
}
ORDER BY ?teamName
"""

for row in g.query(query):
    print(row.name, row.teamName)