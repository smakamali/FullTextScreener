from neo4j import GraphDatabase, basic_auth

# testing
# Connection details
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "neo4j_rag_poc"

def clear_database(session):
    try:
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared successfully!")
    except Exception as e:
        print(f"Error while clearing database: {e}")


def list_running_databases(session):
    print("Available Neo4j Databases:\n")
    result = session.run("SHOW DATABASES")
    for record in result:
        name = record.get("name")
        status = record.get("currentStatus") or record.get("status")
        role = record.get("role")
        print(f"Name: {name}, Status: {status}, Role: {role}")

def show_labels(session):
    print("\nNode Labels:")
    result = session.run("CALL db.labels()")
    for record in result:
        print(f" - {record['label']}")

def show_relationship_types(session):
    print("\nRelationship Types:")
    result = session.run("CALL db.relationshipTypes()")
    for record in result:
        print(f" - {record['relationshipType']}")
        
def count_all_nodes(session):
    result = session.run("MATCH (n) RETURN count(n) AS total_nodes")
    total_nodes = result.single()["total_nodes"]
    print(f"Total number of nodes in the database: {total_nodes}")

        
def sample_nodes(session, limit=5):
    print(f"\nSample Nodes (limit {limit}):")
    result = session.run(f"MATCH (n) RETURN n LIMIT {limit}")
    for record in result:
        print(record["n"])

def show_indexes(session):
    print("\nIndexes:")
    result = session.run("SHOW INDEXES")
    for record in result:
        print(f" - Name: {record['name']}, Type: {record['type']}, EntityType: {record['entityType']}")

def show_constraints(session):
    print("\nConstraints:")
    result = session.run("SHOW CONSTRAINTS")
    for record in result:
        print(f" - Name: {record['name']}, Type: {record['type']}, EntityType: {record['entityType']}")

def explore_neo4j(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
    try:
        with driver.session() as session:
            count_all_nodes(session)
            list_running_databases(session)
            show_labels(session)
            show_relationship_types(session)
            # sample_nodes(session)
            show_indexes(session)
            show_constraints(session)
            clear_database(session)
    except Exception as e:
        print(f"Error while exploring Neo4j: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    explore_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
