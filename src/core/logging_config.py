import sys
from loguru import logger 
from src.core.config import settings

# Remove the Loguru's default handler (plain stderr, no format control)
logger.remove()

# Terminal handler - colored, humna-readable during development
logger.add(
    sys.stdout,
    level = settings.log_level,
    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    ),
)


# File Handler - JSON lines, rotates daily, keeps 14 days
logger.add(
    'logs/medguard_{time:YYYY-MM-DD}.log',
    rotation = "00:00",
    retention = "14 days",
    level = "DEBUG",
    serialize = True,
)


__all__ = ["logger"]



"""

User Query
     │
     ▼
Embed Query
     │
     ▼
FAISS Search
     │
     ▼
Retrieve Chunks
     │
     ▼
LLM Generates Answer

At every step, you can log what is happening.

1. Model Loading
        logger.info("Loading embedding model...")

Output
        INFO | Loading embedding model...

2. Document Loading
        logger.info(f"Loaded {len(documents)} documents")

Output

        INFO | Loaded 250 documents


Why not use print()?

Imagine your RAG API serves 10,000 users per day.

With print():

Output is mixed together.
No timestamps.
No log levels.
No easy way to save logs.
Hard to debug production issues.

With Loguru:

2026-06-27 10:14:22 INFO Loading embedding model
2026-06-27 10:14:24 INFO Created 4821 chunks
2026-06-27 10:14:26 INFO Indexed into FAISS
2026-06-27 10:15:03 INFO User query received
2026-06-27 10:15:04 INFO Retrieved top 5 chunks
2026-06-27 10:15:05 INFO Response generated

This makes it much easier to trace what happened and when.



In a RAG application, Loguru is used to log every important step in the pipeline, such as 
document loading, chunking, embedding generation, vector database indexing, retrieval, 
LLM response generation, and errors. These logs help developers debug issues, 
monitor system performance, and troubleshoot failures in production without relying on print statements.
"""