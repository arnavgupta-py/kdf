import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, Text
from backend.db.sqlite import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), index=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Inferred Preferences (Stored as JSON text for SQLite compatibility)
    # e.g., {"toll_aversion": 0.8, "variance_tolerance": 0.2, "highway_preference": 0.6}
    preferences = Column(Text, default="{}")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
