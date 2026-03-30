"""
Cypher query templates for GraphRAG.

Each template returns structured data from the DES Knowledge Graph
that can be injected into the chatbot's context alongside vector search results.
"""

# ── Researcher queries ─────────────────────────────────────────────────

RESEARCHERS_BY_THEME = """
MATCH (r:Researcher)-[:AUTHORED]->(p:Publication)-[:TAGGED_WITH]->(t:Theme)
WHERE toLower(t.label) CONTAINS toLower($keyword)
MATCH (r)-[:AFFILIATED_WITH]->(i:Institution {country: 'Sweden'})
WITH r.name AS name, r.h_index AS h_index, collect(DISTINCT i.label)[0] AS institution,
     count(DISTINCT p) AS publications
ORDER BY publications DESC
LIMIT 5
RETURN name, h_index, institution, publications
"""

RESEARCHERS_BY_INSTITUTION = """
MATCH (r:Researcher)-[:AFFILIATED_WITH]->(i:Institution)
WHERE toLower(i.label) CONTAINS toLower($keyword) OR toLower(i.id) CONTAINS toLower($keyword)
WITH r, i
MATCH (r)-[:AUTHORED]->(p:Publication)
WITH r.name AS name, r.h_index AS h_index, i.label AS institution,
     count(p) AS publications
ORDER BY publications DESC
LIMIT 10
RETURN name, h_index, institution, publications
"""

TOP_RESEARCHERS_BY_DOMAIN = """
MATCH (r:Researcher)-[:AUTHORED]->(p:Publication)-[:TAGGED_WITH]->(t:Theme)
WHERE t.domain = $domain
MATCH (r)-[:AFFILIATED_WITH]->(i:Institution {country: 'Sweden'})
WITH r.name AS name, r.h_index AS h_index, collect(DISTINCT i.label)[0] AS institution,
     count(DISTINCT p) AS publications, collect(DISTINCT t.label)[0..3] AS themes
ORDER BY publications DESC
LIMIT 5
RETURN name, h_index, institution, publications, themes
"""

# ── Publication queries ────────────────────────────────────────────────

PUBLICATIONS_BY_THEME = """
MATCH (p:Publication)-[:TAGGED_WITH]->(t:Theme)
WHERE toLower(t.label) CONTAINS toLower($keyword)
OPTIONAL MATCH (p)<-[:AUTHORED]-(r:Researcher)-[:AFFILIATED_WITH]->(i:Institution {country: 'Sweden'})
WITH p, collect(DISTINCT r.name)[0..3] AS authors, collect(DISTINCT i.label)[0] AS institution
ORDER BY p.year DESC
LIMIT 5
RETURN p.title AS title, p.year AS year, p.doi AS doi, authors, institution
"""

PUBLICATIONS_BY_INSTITUTION = """
MATCH (r:Researcher)-[:AFFILIATED_WITH]->(i:Institution)
WHERE toLower(i.label) CONTAINS toLower($keyword)
MATCH (r)-[:AUTHORED]->(p:Publication)
WITH p, collect(DISTINCT r.name)[0..3] AS authors, i.label AS institution
ORDER BY p.year DESC
LIMIT 5
RETURN p.title AS title, p.year AS year, p.doi AS doi, authors, institution
"""

# ── Institution queries ────────────────────────────────────────────────

INSTITUTIONS_BY_DOMAIN = """
MATCH (r:Researcher)-[:AFFILIATED_WITH]->(i:Institution {country: 'Sweden'})
MATCH (r)-[:AUTHORED]->(p:Publication)-[:TAGGED_WITH]->(t:Theme)
WHERE t.domain = $domain
WITH i.label AS institution, i.city AS city,
     count(DISTINCT r) AS researchers, count(DISTINCT p) AS publications
ORDER BY publications DESC
LIMIT 5
RETURN institution, city, researchers, publications
"""

# ── Theme/domain queries ───────────────────────────────────────────────

THEMES_IN_DOMAIN = """
MATCH (t:Theme)
WHERE t.domain = $domain
MATCH (p:Publication)-[:TAGGED_WITH]->(t)
WITH t.label AS theme, t.id AS id, count(p) AS publications
ORDER BY publications DESC
RETURN theme, id, publications
"""

# ── Keyword mapping for query classification ───────────────────────────

DOMAIN_KEYWORDS = {
    "earth_obs": ["fjärranalys", "remote sensing", "sentinel-2", "sentinel", "marktäcke",
                  "land cover", "sar", "radar", "vegetation", "skog", "forest",
                  "jordbruk", "agriculture", "nmd", "lidar", "drone", "uav"],
    "wireless": ["6g", "satcom", "mimo", "antenna", "terahertz", "gnss", "navigation",
                 "kommunikation", "satellite communication"],
    "space_physics": ["rymdväder", "space weather", "magnetosfär", "magnetosphere",
                      "solvind", "solar wind", "jonosfär", "ionosphere"],
    "climate": ["klimat", "climate", "arktis", "arctic", "atmosfär", "atmosphere"],
}

THEME_KEYWORDS = {
    "sar_radar": ["sar", "radar", "syntetisk apertur", "synthetic aperture", "insar"],
    "forest": ["skog", "forest", "trädh", "tree height", "avverkning", "harvest"],
    "land_cover": ["marktäcke", "land cover", "lulc", "klassificering", "classification"],
    "agriculture": ["jordbruk", "agriculture", "gröda", "crop", "åker"],
    "ml_ai": ["maskininlärning", "machine learning", "ai", "neural", "deep learning"],
    "sentinel": ["sentinel", "copernicus", "s2", "s1"],
    "water": ["vatten", "water", "hydrologi", "hydrology", "sjö", "lake"],
    "ocean_marine": ["hav", "ocean", "marin", "marine", "fartyg", "vessel", "ship"],
}
