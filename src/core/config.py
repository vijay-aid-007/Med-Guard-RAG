from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    
    #_____LLM________________________________________________
    ollama_base_url: str = Field(default= "http://localhost:11434")
    llm_model: str = Field(default="openai/gpt-oss-120b")
    llm_temperature: float = Field(default=0.3)
    llm_max_tokens: int = Field(default=512)
    llm_provider:    str = Field(default="groq") 
    groq_api_key:    str = Field(default="")  

    #_____Embeddings_________________________________________
    embedding_model: str = Field(
        default = "NeuML/pubmedbert-base-embeddings" #NeuML/pubmedbert-base-embeddings
    )
    embedding_dim: int = Field(default=768)

    #_____Reranker____________________________________________
    reranker_model: str = Field(
        default = "ncbi/MedCPT-Cross-Encoder" #"cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    #_____Retrieval___________________________________________
    retrieval_top_k: int = Field(default=20)
    rerank_top_k : int = Field(default=15)
    chunk_size:  int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    #_____Guardrail Threshold_________________________________
    faithfulness_threshold: float = Field(default=0.3)
    frustration_block_threshold: float = Field(default=3.0)
    repetition_handoff_count: int = Field(default=2)

    #_____Paths_______________________________________________
    faiss_index_path: str = Field(
        default = "./data/faiss_index/medguard.index"
    )
    faiss_metadata_path: str = Field(
        default = "./data/faiss_index/metadata.pkl"
    )

    #_____Datastores__________________________________________
    postgres_url: str = Field(
        default = "postgresql://medguard:medguard@localhost:5432/medguard"
    )
    redis_url: str = Field(
        default = "redis://localhost:6379/0"
    )

    #_____App__________________________________________________
    app_name: str = Field(default="MedGuard RAG")
    log_level: str = Field(default="INFO")
    session_ttl_seconds: int = Field(default=3600)

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8"
    )


settings = Settings()









"""
pydantic → A library that checks your data. If you say “this must be a number,” it makes sure you don’t accidentally pass a word.

pydantic_settings → An add‑on for pydantic that’s made for app configuration. Its special trick: it can pull values straight from a .env file (where you keep secrets and settings).

BaseSettings → The parent class you inherit from. It’s what gives your Settings class the power to read .env automatically.

SettingsConfigDict → A helper dictionary that tells BaseSettings where your .env file is and how to read it.

Field → Used inside your Settings class to define each variable. You can give it extra info, like a default value if .env doesn’t provide one.

In short: pydantic checks types, pydantic_settings reads .env, BaseSettings enables it, SettingsConfigDict guides it, and Field customizes each variable.
"""