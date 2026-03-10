from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any
from backend.schemas.user import User, UserCreate
from backend.services.user_service import get_user_by_email, create_user, parse_user_preferences
from backend.db.sqlite import get_db

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
