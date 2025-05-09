# For check the connection to docker image 

from neo4j import GraphDatabase

try:
    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "neo4j_rag_poc")) #replace your_password
    with driver.session() as session:
        result = session.run("RETURN 1")
        print(result.single()[0])
    driver.close()
    print("Neo4j connection successful!")

except Exception as e:
    print(f"Error connecting to Neo4j: {e}")