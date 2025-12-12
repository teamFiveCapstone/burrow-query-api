from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    db_name: str = "burrowdb"
    db_user: str = "burrow_admin"
    db_password: str = "dummy"
    db_host: str = "localhost"
    db_port: int = 5432

    table_name: str = "burrow_table_hybrid2"
    embed_dim: int = 1024

    aws_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.titan-embed-text-v2:0"

    api_title: str = "Burrow Query API"
    api_version: str = "1.0.0"
    api_description: str = "Vector database query API for document retrieval"

    api_token: str = ""

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
