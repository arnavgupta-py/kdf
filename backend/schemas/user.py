from pydantic import BaseModel, EmailStr, ConfigDict, Field
from typing import Optional, Dict
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None
    preferences: Optional[Dict[str, float]] = None

class UserInDBBase(UserBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    created_at: datetime
    updated_at: datetime

# Response model returned by API
class User(UserInDBBase):
    preferences: Dict[str, float] = Field(default_factory=dict)
