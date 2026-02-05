1. refactor the frontend. Using high performance python lib PyDeck.


2. Refine Extraction and implement hiearchy cluster labels (this is only one step within the semantic view):

Update your worker.py to create a semantic_representation string (Title + Headers + Abstract) and embed that, not the whole text.

Implement Hierarchy:

Run HDBSCAN on your embeddings.

Run c-TF-IDF on the clusters to get keywords.

Use Ollama (local LLM) to label the top 2000 cluster centroids.



3. Handling both static and dynamic web content fetching, with high performance expectation: maybe using headless when dynamic, and trauflutra

4. Implement the "Multi-Layout" Architecture:

View A: The Semantic Landscape (The "What")

Input: The raw BGE-small embedding (384 dims).

Technique: Standard UMAP.

Result: Dots cluster by topic. Physics is here, Poetry is there.

User Question: "What is the content landscape?"

View B: The Epistemic Landscape (The "How")

Input: A custom vector constructed only from your metadata scores.

Vector = [quality_score, complexity_score, subjectivity_score, consensus_score]

Technique: PCA or direct mapping.

X-Axis: Consensus vs. Dissent.

Y-Axis: Simple vs. Complex.

Color: Quality.

Result: A scatter plot where "Mainstream Science" is top-right and "Fringe Theories" are bottom-left.

User Question: "Where are the high-quality, complex contrarian takes?"

View C: The Chronological Landscape (The "When")

Input: Time + Semantic Topic.

Technique:

X-Axis: Publication Date.

Y-Axis: 1D UMAP (Topic).

Result: A "Streamgraph" flow.

User Question: "How did the discussion on AI safety evolve over time?"

Steps(should be critically evaluated before continuing): expand the Mapper Module and the Database Schema.

Step 1: Database Schema Update (what's the current schema?)

Currently, your documents table likely has x and y columns. Change this to a JSON structure or separate table to support multiple layouts.

Table: document_layouts | doc_id | layout_type | x | y | z | | :--- | :--- | :--- | :--- | :--- | | doc_123 | semantic | 12.5 | -4.2 | 0.5 | | doc_123 | epistemic | 0.8 | 0.9 | 0.1 | | doc_123 | timeline | 2024 | 12.5 | 0.0 |

Step 2: Pipeline Modification (Mapper Module)

In your map script, instead of running UMAP once, you run a loop over your defined "Modes."

Step 3: The "Google Earth" Transition (Frontend)

This is where the magic happens. You use Deck.gl's transition capabilities.