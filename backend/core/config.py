from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "CRONOS"
    API_V1_STR: str = "/api/v1"
    
    # SQLite Configuration
    SQLITE_URL: str = "sqlite:///./cronos.db"
    
    # SQLite Configuration
    SQLITE_URL: str = "sqlite:///./cronos.db"

    # Telemetry Engine gRPC config
    TELEMETRY_GRPC_URL: str = "http://localhost:50051"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
