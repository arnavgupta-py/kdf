from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

class JourneyBase(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    scheduled_departure_time: datetime
    chosen_route: str # JSON/GeoJSON representation

class JourneyCreate(JourneyBase):
    user_id: str

class JourneyUpdate(BaseModel):
    actual_departure_time: Optional[datetime] = None
    predicted_arrival_time: Optional[datetime] = None
    actual_arrival_time: Optional[datetime] = None

class JourneyInDBBase(JourneyBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    user_id: str
    actual_departure_time: Optional[datetime] = None
    predicted_arrival_time: Optional[datetime] = None
    actual_arrival_time: Optional[datetime] = None
    created_at: datetime

# Response model returned by API
class Journey(JourneyInDBBase):
    pass
