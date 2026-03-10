from sqlalchemy.orm import Session
from backend.models.user import User
from backend.schemas.user import UserCreate, UserUpdate
from loguru import logger
import json

def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user_in: UserCreate) -> User:
    # MVP hash emulation
    fake_hashed_password = user_in.password + "notreallyhashed"
    db_obj = User(
        email=user_in.email,
        hashed_password=fake_hashed_password,
        full_name=user_in.full_name,
        is_active=user_in.is_active,
        is_superuser=user_in.is_superuser
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    logger.info(f"Created new user with email: {db_obj.email}")
    return db_obj

def parse_user_preferences(user: User) -> dict:
    try:
        return json.loads(user.preferences) if user.preferences else {}
    except AttributeError:
        # Before ORM has populated
        return {}

def update_user_preferences(db: Session, user: User, preferences: dict) -> User:
    user.preferences = json.dumps(preferences)
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"Updated preferences for user: {user.email}")
    return user
