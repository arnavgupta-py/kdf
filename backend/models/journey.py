import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.db.sqlite import Base

class Journey(Base):
    """
    Records travel history per user to serve as a training signal for the Personalised Learning Module.
    """
    __tablename__ = "journeys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    
    origin_lat = Column(Float, nullable=False)
    origin_lon = Column(Float, nullable=False)
    dest_lat = Column(Float, nullable=False)
    dest_lon = Column(Float, nullable=False)
    
    # Store chosen route as a serialized list or GeoJSON string
    chosen_route = Column(Text)
    
    scheduled_departure_time = Column(DateTime, nullable=False)
    actual_departure_time = Column(DateTime)
    predicted_arrival_time = Column(DateTime)
    actual_arrival_time = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref="journeys")
