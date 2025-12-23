"""
Embedding Generation Module - Stage 4
Generates vector embeddings using BGE-M3 via Ollama

Storage Backends (Auto-detected):
- pgvector: Used when PostgreSQL has pgvector extension (Ubuntu/Production)
- LanceDB: Used as fallback when pgvector unavailable (Windows/Development)
"""

import asyncio
import asyncpg
import httpx
import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# BGE-M3 Configuration - Read from environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "bge-m3"))
EMBEDDING_DIMENSIONS = 1024  # BGE-M3 produces 1024-dimensional vectors
MAX_TOKENS = 8192  # BGE-M3 supports up to 8192 tokens
# For Korean text, roughly 1 token = 1-2 chars; use conservative limit
MAX_CHARS_FOR_EMBEDDING = 6000  # Safe limit for Korean text to avoid context overflow

# LanceDB storage path
LANCEDB_PATH = Path("./data-lake/embeddings")


class OllamaEmbedding:
    """Generate embeddings using Ollama's BGE-M3 model"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = EMBEDDING_MODEL):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def check_model_available(self) -> bool:
        """Check if the embedding model is available in Ollama"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self.model.split(":")[0] in model_names
            return False
        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
            return False

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text using Ollama

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector, or None on error
        """
        if not text or not text.strip():
            return None

        # Truncate text if too long - use safe limit for Korean text
        # Korean characters typically use 1-2 tokens per character in BGE-M3
        if len(text) > MAX_CHARS_FOR_EMBEDDING:
            text = text[:MAX_CHARS_FOR_EMBEDDING]
            logger.debug(f"Text truncated to {MAX_CHARS_FOR_EMBEDDING} chars for embedding")

        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )

            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding")
                if embedding and len(embedding) == EMBEDDING_DIMENSIONS:
                    return embedding
                else:
                    logger.warning(f"Unexpected embedding dimensions: {len(embedding) if embedding else 0}")
                    return embedding  # Return anyway, might still work
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 50
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in parallel batches.

        Parallel requests are MUCH faster than sequential due to Ollama's
        model loading behavior (~0.08s parallel vs ~17s sequential).

        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of concurrent requests (default: 50 for speed)

        Returns:
            List of embeddings (same order as input texts)
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tasks = [self.generate_embedding(text) for text in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Minimal delay - parallel requests keep model warm
            if i + batch_size < len(texts):
                await asyncio.sleep(0.05)

        return results


# ============================================================================
# Abstract Vector Store Interface
# ============================================================================

class VectorStore(ABC):
    """Abstract interface for vector storage backends"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the vector store"""
        pass

    @abstractmethod
    async def save_embedding(
        self,
        section_id: str,
        bid_id: str,
        embedding: List[float],
        text: str
    ) -> bool:
        """Save an embedding"""
        pass

    @abstractmethod
    async def get_existing_section_ids(self) -> set:
        """Get set of section IDs that already have embeddings"""
        pass

    @abstractmethod
    async def find_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """Find similar sections"""
        pass

    @abstractmethod
    async def close(self):
        """Close connections"""
        pass


# ============================================================================
# pgvector Backend (PostgreSQL)
# ============================================================================

class PgVectorStore(VectorStore):
    """Vector storage using PostgreSQL with pgvector extension"""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize pgvector extension and create tables"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if pgvector is available
                try:
                    result = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
                    )
                    if not result:
                        logger.warning("pgvector extension not available on this server")
                        return False
                except Exception:
                    return False

                # Try to create/enable extension
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except Exception as e:
                    logger.warning(f"Cannot create pgvector extension: {e}")
                    return False

                # Create embedding table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS bid_section_embeddings (
                        embedding_id SERIAL PRIMARY KEY,
                        section_id VARCHAR(100) NOT NULL,
                        bid_id VARCHAR(50) NOT NULL,
                        embedding vector(1024),
                        model_name VARCHAR(100) NOT NULL DEFAULT 'bge-m3',
                        model_version VARCHAR(50),
                        text_snippet TEXT,
                        char_count INTEGER,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CONSTRAINT uq_section_embedding UNIQUE (section_id, model_name)
                    )
                """)

                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_bid_id
                    ON bid_section_embeddings(bid_id)
                """)

                # HNSW index for fast similarity search
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw
                    ON bid_section_embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)

                logger.info("pgvector initialized successfully")
                self._initialized = True
                return True

        except Exception as e:
            logger.error(f"Failed to initialize pgvector: {e}")
            return False

    async def save_embedding(
        self,
        section_id: str,
        bid_id: str,
        embedding: List[float],
        text: str
    ) -> bool:
        """Save embedding to PostgreSQL with pgvector"""
        try:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            text_snippet = text[:500] if text else ""
            char_count = len(text) if text else 0

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO bid_section_embeddings (
                        section_id, bid_id, embedding, model_name,
                        text_snippet, char_count
                    ) VALUES ($1, $2, $3::vector, $4, $5, $6)
                    ON CONFLICT (section_id, model_name)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        text_snippet = EXCLUDED.text_snippet,
                        char_count = EXCLUDED.char_count,
                        created_at = NOW()
                """, section_id, bid_id, embedding_str, EMBEDDING_MODEL,
                    text_snippet, char_count)

            return True
        except Exception as e:
            logger.error(f"Failed to save embedding (pgvector): {e}")
            return False

    async def get_existing_section_ids(self) -> set:
        """Get section IDs that already have embeddings"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT section_id FROM bid_section_embeddings WHERE model_name = $1",
                    EMBEDDING_MODEL
                )
                return {row['section_id'] for row in rows}
        except Exception as e:
            logger.error(f"Failed to get existing embeddings: {e}")
            return set()

    async def find_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """Find similar sections using pgvector"""
        try:
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        e.section_id,
                        e.bid_id,
                        e.text_snippet,
                        1 - (e.embedding <=> $1::vector) as similarity
                    FROM bid_section_embeddings e
                    WHERE 1 - (e.embedding <=> $1::vector) >= $2
                    ORDER BY e.embedding <=> $1::vector
                    LIMIT $3
                """, embedding_str, min_similarity, limit)

                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to find similar (pgvector): {e}")
            return []

    async def close(self):
        """No-op for pgvector (pool managed externally)"""
        pass


# ============================================================================
# LanceDB Backend (Local File-based)
# ============================================================================

class LanceDBStore(VectorStore):
    """Vector storage using LanceDB (local file-based, no extensions needed)"""

    def __init__(self, db_path: Path = LANCEDB_PATH):
        self.db_path = db_path
        self.db = None
        self.table = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize LanceDB"""
        try:
            import lancedb
            import pyarrow as pa

            # Create directory if needed
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Connect to LanceDB
            self.db = lancedb.connect(str(self.db_path))

            # Check if table exists
            if "bid_embeddings" in self.db.table_names():
                self.table = self.db.open_table("bid_embeddings")
                logger.info(f"LanceDB initialized (existing table with {self.table.count_rows()} rows)")
            else:
                # Create empty table with schema
                schema = pa.schema([
                    pa.field("section_id", pa.string()),
                    pa.field("bid_id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSIONS)),
                    pa.field("text_snippet", pa.string()),
                    pa.field("char_count", pa.int32()),
                    pa.field("model_name", pa.string()),
                    pa.field("created_at", pa.string()),
                ])
                self.table = self.db.create_table("bid_embeddings", schema=schema)
                logger.info("LanceDB initialized (new table created)")

            self._initialized = True
            return True

        except ImportError:
            logger.error("LanceDB not installed. Run: pip install lancedb pyarrow")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            return False

    async def save_embedding(
        self,
        section_id: str,
        bid_id: str,
        embedding: List[float],
        text: str
    ) -> bool:
        """Save embedding to LanceDB"""
        try:
            text_snippet = text[:500] if text else ""
            char_count = len(text) if text else 0

            # Check if exists and delete (LanceDB doesn't have upsert)
            try:
                existing = self.table.search().where(
                    f"section_id = '{section_id}'"
                ).limit(1).to_list()
                if existing:
                    # Delete existing
                    self.table.delete(f"section_id = '{section_id}'")
            except Exception:
                pass  # Table might be empty

            # Add new record
            self.table.add([{
                "section_id": section_id,
                "bid_id": bid_id,
                "vector": embedding,
                "text_snippet": text_snippet,
                "char_count": char_count,
                "model_name": EMBEDDING_MODEL,
                "created_at": datetime.now().isoformat()
            }])

            return True
        except Exception as e:
            logger.error(f"Failed to save embedding (LanceDB): {e}")
            return False

    async def get_existing_section_ids(self) -> set:
        """Get section IDs that already have embeddings"""
        try:
            if self.table.count_rows() == 0:
                return set()

            # Get all section_ids
            df = self.table.to_pandas()
            return set(df['section_id'].tolist())
        except Exception as e:
            logger.error(f"Failed to get existing embeddings: {e}")
            return set()

    async def find_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """Find similar sections using LanceDB"""
        try:
            if self.table.count_rows() == 0:
                return []

            # LanceDB uses L2 distance by default, we need cosine
            results = self.table.search(query_embedding) \
                .metric("cosine") \
                .limit(limit) \
                .to_list()

            # Convert distance to similarity and filter
            similar = []
            for r in results:
                # LanceDB returns _distance (cosine distance = 1 - similarity)
                similarity = 1 - r.get('_distance', 1)
                if similarity >= min_similarity:
                    similar.append({
                        'section_id': r['section_id'],
                        'bid_id': r['bid_id'],
                        'text_snippet': r['text_snippet'],
                        'similarity': similarity
                    })

            return similar
        except Exception as e:
            logger.error(f"Failed to find similar (LanceDB): {e}")
            return []

    async def close(self):
        """Close LanceDB connection"""
        # LanceDB doesn't require explicit close
        pass


# ============================================================================
# Auto-detection and Factory
# ============================================================================

async def create_vector_store(db_pool: asyncpg.Pool) -> Tuple[VectorStore, str]:
    """
    Create appropriate vector store based on available backends

    Returns:
        Tuple of (VectorStore instance, backend name)

    Environment Variables:
        FORCE_PGVECTOR: If set to 'true', will only use pgvector (no LanceDB fallback)
    """
    force_pgvector = os.getenv('FORCE_PGVECTOR', 'false').lower() == 'true'

    # Try pgvector first (preferred for production)
    pgvector_store = PgVectorStore(db_pool)
    if await pgvector_store.initialize():
        return pgvector_store, "pgvector"

    # If FORCE_PGVECTOR is set, fail hard instead of falling back
    if force_pgvector:
        raise RuntimeError(
            "FORCE_PGVECTOR is set but pgvector is not available. "
            "Ensure the 'vector' extension is installed: CREATE EXTENSION IF NOT EXISTS vector;"
        )

    logger.info("pgvector not available, falling back to LanceDB")

    # Fall back to LanceDB (works everywhere)
    lancedb_store = LanceDBStore()
    if await lancedb_store.initialize():
        return lancedb_store, "lancedb"

    raise RuntimeError("No vector storage backend available. Install lancedb: pip install lancedb pyarrow")


# ============================================================================
# Stage 4 Integration for ETL Pipeline
# ============================================================================

async def get_sections_without_embeddings(
    conn: asyncpg.Connection,
    existing_ids: set,
    limit: Optional[int] = None
) -> List[Dict]:
    """Get text sections that don't have embeddings yet"""
    query = """
        SELECT
            ts.section_id,
            ts.bid_id,
            ts.text,
            ts.char_count
        FROM bid_text_sections ts
        WHERE ts.text IS NOT NULL
            AND LENGTH(ts.text) > 50
        ORDER BY ts.extracted_at
    """

    if limit:
        query += f" LIMIT {limit * 2}"  # Get more since we'll filter

    rows = await conn.fetch(query)

    # Filter out existing embeddings
    sections = [dict(row) for row in rows if row['section_id'] not in existing_ids]

    if limit:
        sections = sections[:limit]

    return sections


async def generate_and_save_embeddings(
    db_pool: asyncpg.Pool,
    limit: Optional[int] = None,
    batch_size: int = 50
) -> Dict:
    """
    Generate embeddings for all text sections without embeddings
    Auto-detects storage backend (pgvector or LanceDB)

    Note: Parallel requests are ~200x faster than sequential with Ollama.
    Default batch_size=50 processes 3000 sections in ~5 minutes.

    Args:
        db_pool: Database connection pool
        limit: Optional limit on number of sections to process
        batch_size: Number of embeddings to generate concurrently (default: 50)

    Returns:
        Dictionary with processing statistics
    """
    logger.info("=" * 80)
    logger.info("STAGE 4: Embedding Generation (BGE-M3 via Ollama)")
    logger.info("=" * 80)

    # Initialize Ollama client
    ollama = OllamaEmbedding()
    vector_store = None

    try:
        # Check if model is available
        if not await ollama.check_model_available():
            logger.error(f"Model {EMBEDDING_MODEL} not available in Ollama")
            logger.error("Run: ollama pull bge-m3")
            return {"error": "Model not available", "generated": 0, "errors": 0}

        logger.info(f"Using embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dimensions)")

        # Create vector store (auto-detects backend)
        vector_store, backend_name = await create_vector_store(db_pool)
        logger.info(f"Using vector storage backend: {backend_name.upper()}")

        # Get existing embeddings
        existing_ids = await vector_store.get_existing_section_ids()
        logger.info(f"Found {len(existing_ids)} existing embeddings")

        # Get sections without embeddings
        async with db_pool.acquire() as conn:
            sections = await get_sections_without_embeddings(conn, existing_ids, limit)

        total_sections = len(sections)
        logger.info(f"Found {total_sections} sections to process")

        if total_sections == 0:
            logger.info("No sections need embedding generation")
            return {"generated": 0, "errors": 0, "total": 0, "backend": backend_name}

        generated_count = 0
        error_count = 0

        # Process in batches
        for i in range(0, total_sections, batch_size):
            batch = sections[i:i + batch_size]
            batch_texts = [s['text'] for s in batch]

            batch_num = i // batch_size + 1
            total_batches = (total_sections + batch_size - 1) // batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} sections)")

            # Generate embeddings for batch
            embeddings = await ollama.generate_embeddings_batch(batch_texts, batch_size=batch_size)

            # Save embeddings
            for section, embedding in zip(batch, embeddings):
                if embedding:
                    success = await vector_store.save_embedding(
                        section['section_id'],
                        section['bid_id'],
                        embedding,
                        section['text']
                    )
                    if success:
                        generated_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1
                    logger.warning(f"Failed to generate embedding for {section['section_id']}")

            # Progress log
            if (i + batch_size) % 50 == 0 or (i + batch_size) >= total_sections:
                logger.info(f"Progress: {min(i + batch_size, total_sections)}/{total_sections} "
                          f"(Generated: {generated_count}, Errors: {error_count})")

        logger.info("=" * 80)
        logger.info(f"Stage 4 Complete: {generated_count} embeddings generated, {error_count} errors")
        logger.info(f"Backend: {backend_name}")
        logger.info("=" * 80)

        return {
            "generated": generated_count,
            "errors": error_count,
            "total": total_sections,
            "backend": backend_name
        }

    finally:
        await ollama.close()
        if vector_store:
            await vector_store.close()


async def search_similar_content(
    db_pool: asyncpg.Pool,
    query_text: str,
    limit: int = 10
) -> List[Dict]:
    """
    Search for similar content using text query

    Args:
        db_pool: Database connection pool
        query_text: Text to search for
        limit: Maximum number of results

    Returns:
        List of similar sections with similarity scores
    """
    ollama = OllamaEmbedding()
    vector_store = None

    try:
        # Generate embedding for query
        query_embedding = await ollama.generate_embedding(query_text)

        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        # Create vector store
        vector_store, backend_name = await create_vector_store(db_pool)
        logger.info(f"Searching with backend: {backend_name}")

        # Search for similar sections
        results = await vector_store.find_similar(query_embedding, limit)

        return results

    finally:
        await ollama.close()
        if vector_store:
            await vector_store.close()


# ============================================================================
# Utility: Sync embeddings from LanceDB to pgvector (for deployment)
# ============================================================================

async def sync_lancedb_to_pgvector(db_pool: asyncpg.Pool) -> Dict:
    """
    Sync embeddings from LanceDB to pgvector
    Use this when deploying from Windows (LanceDB) to Ubuntu (pgvector)
    """
    logger.info("Syncing embeddings from LanceDB to pgvector...")

    # Check pgvector is available
    pgvector_store = PgVectorStore(db_pool)
    if not await pgvector_store.initialize():
        return {"error": "pgvector not available on target database"}

    # Load from LanceDB
    lancedb_store = LanceDBStore()
    if not await lancedb_store.initialize():
        return {"error": "LanceDB not available"}

    try:
        # Get all embeddings from LanceDB
        df = lancedb_store.table.to_pandas()
        total = len(df)
        logger.info(f"Found {total} embeddings in LanceDB")

        if total == 0:
            return {"synced": 0, "errors": 0}

        synced = 0
        errors = 0

        for _, row in df.iterrows():
            success = await pgvector_store.save_embedding(
                row['section_id'],
                row['bid_id'],
                row['vector'],
                row['text_snippet']
            )
            if success:
                synced += 1
            else:
                errors += 1

            if synced % 100 == 0:
                logger.info(f"Progress: {synced}/{total}")

        logger.info(f"Sync complete: {synced} synced, {errors} errors")
        return {"synced": synced, "errors": errors, "total": total}

    finally:
        await lancedb_store.close()
        await pgvector_store.close()


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        from dotenv import load_dotenv

        load_dotenv()

        # Build database URL
        db_host = os.getenv("POSTGRES_HOST")
        db_user = os.getenv("POSTGRES_USER")
        db_password = os.getenv("POSTGRES_PASSWORD")
        db_name = os.getenv("POSTGRES_DB", "procurement")
        db_port = os.getenv("POSTGRES_PORT", "5432")

        if not all([db_host, db_user, db_password]):
            print("Database credentials not found in .env")
            sys.exit(1)

        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)

        try:
            # Parse command line args
            limit = None
            sync_mode = False

            for arg in sys.argv[1:]:
                if arg == "--sync":
                    sync_mode = True
                else:
                    try:
                        limit = int(arg)
                        print(f"Limiting to {limit} sections")
                    except ValueError:
                        pass

            if sync_mode:
                result = await sync_lancedb_to_pgvector(pool)
            else:
                result = await generate_and_save_embeddings(pool, limit=limit)

            print(f"\nResult: {result}")

        finally:
            await pool.close()

    asyncio.run(main())
