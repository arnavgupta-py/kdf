from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any
from backend.schemas.user import User, UserCreate
from backend.services.user_service import get_user_by_email, create_user, parse_user_preferences, update_user_preferences
from backend.db.sqlite import get_db
from pydantic import BaseModel
from typing import Dict

router = APIRouter()

@router.post("/", response_model=User)
def register_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """
    Create new user.
    """
    user = get_user_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )
    user = create_user(db, user_in=user_in)
    
    # Transform SQLAlchemy object to output format required by Pydantic
    user_dict = {
        **user.__dict__,
        "preferences": parse_user_preferences(user)
    }
    return user_dict

class PreferencesUpdate(BaseModel):
    toll_aversion: bool
    eco_routing: bool
    high_confidence: bool

@router.put("/me/preferences", response_model=Dict[str, Any])
def update_my_preferences(
    *,
    db: Session = Depends(get_db),
    prefs_in: PreferencesUpdate,
) -> Any:
    """
    Update preferences for the currently authenticated user.
    (Mocked to guest@cronos.com for MVP)
    """
    email = "guest@cronos.com"
    user = get_user_by_email(db, email=email)
    
    # Auto-create guest user if they don't exist yet
    if not user:
        user_in = UserCreate(email=email, password="password123", full_name="Guest User")
        user = create_user(db, user_in=user_in)
        
    preferences = {
        "toll_aversion": 1.0 if prefs_in.toll_aversion else 0.0,
        "eco_routing": 1.0 if prefs_in.eco_routing else 0.0,
        "high_confidence": 1.0 if prefs_in.high_confidence else 0.0,
    }
    
    user = update_user_preferences(db, user, preferences)
    
    return {"status": "success", "preferences": parse_user_preferences(user)}
