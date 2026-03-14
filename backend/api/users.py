from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any
from backend.schemas.user import User, UserCreate
from backend.services.user_service import get_user_by_email, create_user, parse_user_preferences, update_user_preferences, verify_password
from backend.db.sqlite import get_db
from pydantic import BaseModel, field_validator
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

    user_dict = {
        **user.__dict__,
        "preferences": parse_user_preferences(user)
    }
    return user_dict


class LoginIn(BaseModel):
    email: str
    password: str


@router.post("/login")
def login(
    *,
    db: Session = Depends(get_db),
    login_in: LoginIn,
) -> Any:
    """
    Authenticate user and return profile.
    """
    user = get_user_by_email(db, email=login_in.email)
    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    user_dict = {
        **user.__dict__,
        "preferences": parse_user_preferences(user)
    }
    return user_dict


class PreferencesUpdate(BaseModel):
    """
    Accepts boolean toggles from Alpine.js.
    Alpine sends Python-style True/False or JS true/false; we normalise.
    """
    toll_aversion: bool
    eco_routing: bool
    high_confidence: bool

    @field_validator("toll_aversion", "eco_routing", "high_confidence", mode="before")
    @classmethod
    def coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)


@router.put("/me/preferences", response_model=Dict[str, Any])
def update_my_preferences(
    *,
    db: Session = Depends(get_db),
    prefs_in: PreferencesUpdate,
) -> Any:
    """
    Update preferences for the currently authenticated user.
    Mapped to guest@cronos.com for MVP; auto-creates user if missing.
    """
    email = "guest@cronos.com"
    user = get_user_by_email(db, email=email)

    if not user:
        user = create_user(
            db,
            UserCreate(email=email, password="cronos-demo", full_name="Demo User"),
        )

    # Convert booleans to float weights used by the optimiser
    preferences = {
        "toll_aversion":      1.0 if prefs_in.toll_aversion  else 0.0,
        "eco_routing":        1.0 if prefs_in.eco_routing     else 0.0,
        "high_confidence":    1.0 if prefs_in.high_confidence else 0.0,
        # Derive variance_tolerance from high_confidence:
        # high confidence = low tolerance for variance
        "variance_tolerance": 0.2 if prefs_in.high_confidence else 0.7,
        # Highway preference is inverse of toll-aversion
        "highway_preference": 0.3 if prefs_in.toll_aversion  else 0.7,
    }

    user = update_user_preferences(db, user, preferences)
    return {"status": "ok", "preferences": parse_user_preferences(user)}

