**Research & Design Prompt: MCP RAG Ingestion Server for Code and Documentation**

**Objective:** Design the core architecture and implementation strategy for a Minimum Viable Product (MCP) server in Python. This server will act as an ingestion pipeline for Retrieval Augmented Generation (RAG), focused on processing technical documentation (URLs, sitemaps) and code repositories (Python, C#) to populate a local vector store (Milvus initially). The goal is to enable highly accurate and contextually relevant RAG prompts for tools like VSCode, Windsurf, and other LLM applications.

**Target Audience:** An expert software engineer proficient in modern Python, object-oriented design, DRY principles, asynchronous programming, and experienced with the specified tech stack components.

**Core Philosophy:** The design must prioritize modularity, testability, maintainability, and adherence to OOP/DRY principles from the outset, even for an MCP. Asynchronous programming should be leveraged where I/O-bound operations are dominant (crawling, embedding, vector store interactions).

**Research & Design Areas:**

1.  **Overall Architecture & Server Design:**
    *   Propose a high-level architecture for the MCP server. Consider whether a simple Flask/FastAPI app is sufficient or if a more structured approach (e.g., based on task queues like Celery for background processing of ingestion) is necessary given the potentially long-running crawling/processing tasks.
    *   How will the different components (Crawler, Processor, Storage, API) interact? Design the primary classes and their responsibilities.
    *   How will asynchronous operations (using `asyncio`) be integrated throughout the pipeline, from crawling (`Crawl4AI` likely supports this) to embedding (Ollama) and vector store interactions (Milvus client)?
    *   Define the API endpoints or command-line interface needed for triggering ingestion (by URL, sitemap, or local path) and potentially managing sources.

2.  **Ingestion Layer (Crawling & Saving):**
    *   **Crawl4AI Integration:** How to effectively use `Crawl4AI` to handle single URLs and sitemaps? Research its capabilities for extracting main content, handling different website structures, and its asynchronous features.
    *   **HTML to Markdown Conversion:** Research robust Python libraries for converting HTML content obtained from `Crawl4AI` into clean Markdown. Consider edge cases like tables, code blocks, images, and complex layouts. How to ensure the conversion preserves as much semantic structure as possible relevant for technical documentation?
    *   **Local Storage:** Design the structure for saving the raw Markdown files locally *before* splitting/chunking. What naming convention is robust enough to link back to the source URL or original filename? How to handle updates to source content in the local storage layer?

3.  **Processing Layer (Splitting, Chunking, Embedding):**
    *   This is the most critical part for effective RAG. Design intelligent splitting and chunking strategies for three distinct content types:
        *   **Technical Documentation (Markdown):**
            *   Research advanced Markdown splitting techniques. How can we use headings (H1, H2, etc.), code blocks, lists, and paragraphs to create logically coherent chunks?
            *   Investigate Langchain's document transformers (e.g., `RecursiveCharacterTextSplitter`) but be prepared to design custom logic if needed to better respect Markdown structure.
            *   How to capture metadata like the source URL, original filename, and potentially section titles/hierarchy for each chunk?
        *   **Python Scripts:**
            *   Research parsing techniques for Python code, focusing on Abstract Syntax Trees (AST) or similar language-aware methods.
            *   How to split code into logical units like classes, functions, methods, and significant code blocks?
            *   Investigate Langchain's code splitters (`RecursiveCharacterTextSplitter.from_language("python")`) and evaluate their effectiveness. Propose enhancements if necessary.
            *   How to capture crucial metadata: original filename, *exact line numbers* for the start and end of the code chunk, and potentially the containing class/function name?
        *   **C# Scripts:**
            *   Research parsing techniques for C# code (e.g., using Python libraries that wrap .NET parsing capabilities or other dedicated parsers).
            *   Apply similar principles as Python splitting to identify logical units (classes, methods, properties).
            *   Investigate existing C# code splitters in Python ecosystems (if any) or propose a strategy for implementing one, potentially leveraging tools like Roslyn via interop if feasible or simpler regex/syntax-aware methods if not.
            *   Define the metadata to capture: original filename, *exact line numbers*, containing class/method name.
    *   **Embedding with Ollama:** How to integrate Ollama for generating embeddings for each chunk? Research selecting appropriate local models for embedding technical text and code (e.g., models fine-tuned for code). Design the asynchronous call pattern to the Ollama API.

4.  **Storage Layer (Vector Store & Metadata Management):**
    *   **Milvus Integration & Schema Design:**
        *   Design the Milvus collection schema(s). How will Markdown chunks, Python chunks, and C# chunks be stored? Propose a schema that allows:
            *   Storing the vector embedding.
            *   Storing the raw text of the chunk.
            *   Storing *all* identified useful metadata: source URL, original filename, language (markdown, python, c#), line numbers (for code), section titles (for markdown), etc.
        *   Crucially, how will the schema enable searching Python and C# scripts *separately* while still allowing unified search if needed? (e.g., using a `language` field in the schema and filtering during queries, or separate collections). Justify the chosen approach.
    *   **Vector Store Abstraction:** Design an abstract interface or base class for the vector store interaction layer. This `VectorStoreClient` interface should define methods like `upsert`, `search`, `delete`, etc., without assuming the underlying implementation (Milvus). Implement the `MilvusClient` concrete class conforming to this interface. This is key for easily swapping/adding FAISS, ChromaDB, Supabase, etc., later following the Dependency Inversion Principle.
    *   **Metadata Storage Strategy:** Beyond the vector store, is there a need for a separate metadata index or database to support queries based *only* on metadata (e.g., find all documents from a specific URL, or all Python files from a project) without requiring a vector search? Analyze the pros and cons.

5.  **Management Layer (Upsert Logic):**
    *   Design the logic for the `upsert` operation triggered by URL, filename, or topic.
    *   How will the system determine if content is new or updated? (e.g., tracking source URLs/filenames and checking modification dates or content hashes).
    *   When content is updated, how will the system handle replacing old chunks in the vector store with new ones? This needs careful coordination with the chosen vector store (Milvus) capabilities (e.g., deleting based on metadata filters, inserting new vectors).

6.  **Observability (Langfuse Integration):**
    *   Research the `langfuse` Python library and its integration points.
    *   Propose how to integrate `langfuse` to trace the ingestion pipeline: crawling, processing (chunking, embedding), and vector store upserts.
    *   How can `langfuse` be used to monitor the performance and success/failure of different stages?

7.  **PydanticAI Model Validation:**
    *   Identify key data structures and inputs/outputs that should be validated using Pydantic models (e.g., input URL/filename, chunk schema, metadata).
    *   Integrate Pydantic validation into the API endpoints and internal processing functions to ensure data integrity.

**Expected Outcome of Research & Design:**

The result of this research should be a comprehensive document or set of design proposals covering:

1.  A clear, documented architectural diagram.
2.  Detailed design of the primary Python classes, their responsibilities, and how they interact (UML-like diagrams or detailed descriptions).
3.  Specific strategies and identified libraries/techniques for Markdown, Python, and C# intelligent chunking, including how metadata like line numbers will be captured.
4.  The proposed Milvus collection schema(s) and justification for separating code types.
5.  The design of the `VectorStoreClient` abstraction layer.
6.  The design of the `upsert` logic and how content updates will be handled in the vector store.
7.  A plan for integrating `Crawl4AI`, `Ollama`, `Milvus`, `Langfuse`, and `PydanticAI` cleanly within the OOP structure.
8.  Identification of potential challenges and proposed solutions (e.g., error handling for failed crawls/embeddings, scaling considerations even for MCP, dependency management).
9.  A prioritized list of implementation steps for the MCP.

This prompt encourages deep thinking about the technical challenges and architectural choices necessary to build a robust, maintainable, and effective RAG ingestion pipeline using the specified modern Python stack.