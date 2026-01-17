// Polymath v4 Neo4j Schema
// Run: cat schema/005_neo4j.cypher | docker exec -i polymath-neo4j cypher-shell -u neo4j -p polymathic2026

// Constraints (uniqueness)
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.passage_id IS UNIQUE;
CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT repo_url IF NOT EXISTS FOR (r:Repository) REQUIRE r.url IS UNIQUE;
CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE;

// Indexes for performance
CREATE INDEX doc_title IF NOT EXISTS FOR (d:Document) ON (d.title);
CREATE INDEX doc_doi IF NOT EXISTS FOR (d:Document) ON (d.doi);
CREATE INDEX concept_type IF NOT EXISTS FOR (c:Concept) ON (c.type);
CREATE INDEX passage_section IF NOT EXISTS FOR (p:Passage) ON (p.section);

// Full-text index for search
CREATE FULLTEXT INDEX passage_text IF NOT EXISTS FOR (p:Passage) ON EACH [p.text];
CREATE FULLTEXT INDEX concept_search IF NOT EXISTS FOR (c:Concept) ON EACH [c.name];

// Graph structure:
//
// (:Document)-[:HAS_PASSAGE]->(:Passage)-[:MENTIONS]->(:Concept)
// (:Document)-[:CITES]->(:Document)
// (:Document)-[:CITES_CODE]->(:Repository)
// (:Repository)-[:CONTAINS]->(:CodeChunk)
// (:Skill)-[:DERIVED_FROM]->(:Passage)
// (:Skill)-[:IMPLEMENTS]->(:Concept)
// (:Concept)-[:RELATED_TO]->(:Concept)
