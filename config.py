"""Configuration for the Query API."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database configuration
    db_name: str = "embeddings"
    db_user: str = "burrow"
    db_password: str = "capstone"
    db_host: str = "burrow-serverless-wilson.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com"
    db_port: int = 5432

    # Vector store configuration
    table_name: str = "burrow_table"  # PGVectorStore adds "data_" prefix
    embed_dim: int = 1024  # Amazon Titan default dimension

    # AWS Bedrock configuration
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.titan-embed-text-v2:0"

    # API configuration
    api_title: str = "RAGline Query API"
    api_version: str = "1.0.0"
    api_description: str = "Vector database query API for document retrieval"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


# Global settings instance
settings = Settings()
