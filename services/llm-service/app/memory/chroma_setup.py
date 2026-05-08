import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class ChromaMemoryManager:
    """
    Manages interactions with ChromaDB for RAG knowledge base and long-term user memory.
    Uses Sentence Transformers for local, free embeddings.
    """
    
    def __init__(self, chroma_host: str = "chromadb", chroma_port: int = 8000, ollama_host: str = "http://ollama:11434"):
        try:
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
            # Initialize collections
            self.session_memories = self.client.get_or_create_collection(
                name="session_memories",
                embedding_function=self.embedding_function
            )
            self.psychology_knowledge = self.client.get_or_create_collection(
                name="psychology_knowledge",
                embedding_function=self.embedding_function
            )
            
            # Initialize Ollama for summarization tasks
            self.llm = Ollama(base_url=ollama_host, model="llama3")
            logger.info("ChromaMemoryManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaMemoryManager: {e}")
            raise

    async def add_session_memory(self, session_id: str, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Adds a summarized session into the user's long-term vector memory."""
        try:
            doc_metadata = {"user_id": user_id, "session_id": session_id}
            if metadata:
                doc_metadata.update(metadata)
                
            self.session_memories.add(
                ids=[session_id],
                documents=[content],
                metadatas=[doc_metadata]
            )
            logger.info(f"Added session memory for user {user_id}, session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding session memory: {e}")
            return False

    async def retrieve_similar_sessions(self, user_id: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieves past session memories for a specific user based on semantic similarity."""
        try:
            results = self.session_memories.query(
                query_texts=[query],
                n_results=n_results,
                where={"user_id": user_id}
            )
            
            memories = []
            if results["documents"] and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
                for doc, meta in zip(docs, metas):
                    memories.append({"content": doc, "metadata": meta})
            
            return memories
        except Exception as e:
            logger.error(f"Error retrieving similar sessions for user {user_id}: {e}")
            return []

    async def add_knowledge_chunk(self, doc_id: str, theory: str, content: str) -> bool:
        """Adds a chunk of psychological knowledge (e.g., CBT textbook snippet) to the RAG base."""
        try:
            self.psychology_knowledge.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[{"theory": theory}]
            )
            logger.info(f"Added knowledge chunk {doc_id} for theory {theory}")
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge chunk {doc_id}: {e}")
            return False

    async def retrieve_therapy_technique(self, query: str, theory_filter: Optional[str] = None, n_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieves relevant psychological techniques based on the user's current situation."""
        try:
            where_clause = {"theory": theory_filter} if theory_filter else None
            results = self.psychology_knowledge.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            techniques = []
            if results["documents"] and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
                for doc, meta in zip(docs, metas):
                    techniques.append({"content": doc, "metadata": meta})
                    
            return techniques
        except Exception as e:
            logger.error(f"Error retrieving therapy technique: {e}")
            return []

    async def summarize_old_memories(self, user_id: str, limit: int = 10) -> Optional[str]:
        """Uses Ollama to summarize older memories into a dense profile if there are too many."""
        try:
            # Fetch recent memories to summarize
            results = self.session_memories.get(
                where={"user_id": user_id},
                limit=limit
            )
            
            if not results["documents"] or len(results["documents"]) == 0:
                return None
                
            memories_text = "\n".join(results["documents"])
            
            prompt = PromptTemplate.from_template(
                "Summarize the following therapy session memories into a concise psychological profile focusing on core triggers, progress, and recurring themes. Keep it clinical but empathetic.\n\nMemories:\n{memories}"
            )
            
            chain = prompt | self.llm
            summary = chain.invoke({"memories": memories_text})
            
            logger.info(f"Successfully summarized {len(results['documents'])} memories for user {user_id}")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing memories for user {user_id}: {e}")
            return None
