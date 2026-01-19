"""
Neo4j database connection and utilities.

Provides thread-safe connection management for Neo4j graph database.
"""

import threading
from typing import Optional

from neo4j import GraphDatabase

from lib.config import config

_driver = None
_driver_lock = threading.Lock()


def get_neo4j_driver():
    """Get or create the Neo4j driver (singleton, thread-safe)."""
    global _driver

    if _driver is not None:
        return _driver

    with _driver_lock:
        if _driver is not None:
            return _driver

        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        return _driver


def close_driver():
    """Close the Neo4j driver."""
    global _driver
    with _driver_lock:
        if _driver is not None:
            _driver.close()
            _driver = None


def check_health() -> dict:
    """Check Neo4j connection health."""
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        return {"status": "healthy", "uri": config.NEO4J_URI}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def get_node_counts() -> dict:
    """Get counts of different node types."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
        """)
        return {record["label"]: record["count"] for record in result}


def get_relationship_counts() -> dict:
    """Get counts of different relationship types."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(*) as count
            ORDER BY count DESC
        """)
        return {record["type"]: record["count"] for record in result}
