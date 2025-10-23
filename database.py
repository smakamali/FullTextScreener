from neo4j import GraphDatabase

uri = "bolt://localhost:7687"  # or use container IP if needed
user = "neo4j"
password = "neo4j_rag_poc"

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print(result.single()["test"])
driver.close()
