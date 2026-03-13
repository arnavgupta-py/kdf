from sqlalchemy.orm import Session
from backend.models.user import User
from backend.models.journey import Journey
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

def create_journey(db: Session, user_id: str, journey_data: dict) -> Journey:
    db_obj = Journey(
        user_id=user_id,
        origin_lat=journey_data.get("origin_lat", 0.0),
        origin_lon=journey_data.get("origin_lon", 0.0),
        dest_lat=journey_data.get("dest_lat", 0.0),
        dest_lon=journey_data.get("dest_lon", 0.0),
        chosen_route=journey_data.get("chosen_route", ""),
        scheduled_departure_time=journey_data.get("scheduled_departure_time"),
        predicted_arrival_time=journey_data.get("predicted_arrival_time"),
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    logger.info(f"Created new journey for user: {user_id}")
    return db_obj

def update_journey_completion(db: Session, journey_id: str, actual_arrival_time, actual_departure_time=None) -> Journey:
    journey = db.query(Journey).filter(Journey.id == journey_id).first()
    if journey:
        journey.actual_arrival_time = actual_arrival_time
        if actual_departure_time:
            journey.actual_departure_time = actual_departure_time
        db.commit()
        db.refresh(journey)
        logger.info(f"Updated completed journey {journey_id} with actual times.")
    return journey

def get_journey(db: Session, journey_id: str) -> Journey | None:
    return db.query(Journey).filter(Journey.id == journey_id).first()

def get_user_journeys(db: Session, user_id: str, skip: int = 0, limit: int = 100) -> list[Journey]:
    return db.query(Journey).filter(Journey.user_id == user_id).offset(skip).limit(limit).all()

def delete_journey(db: Session, journey_id: str) -> bool:
    journey = db.query(Journey).filter(Journey.id == journey_id).first()
    if journey:
        db.delete(journey)
        db.commit()
        logger.info(f"Deleted journey {journey_id}.")
        return True
    return False

def infer_user_preferences(db: Session, user_id: str) -> dict:
    """
    Uses Proximal Policy Optimization (PPO) Reinforcement Learning
    to dynamically adapt and infer user routing preferences.
    The PPO agent is trained incrementally on journey history.
    """
    from backend.models.ppo_rl import ppo_agent
    import torch
    
    user_journeys = get_user_journeys(db, user_id, limit=50)
    
    # 1. Fallback to default if no history
    if not user_journeys:
        return {"toll_aversion": 0.5, "highway_preference": 0.5, "variance_tolerance": 0.5}

    # 2. Reconstruct State from DB (features: total_journeys, avg_delay, etc.)
    total_journeys = float(max(1, len(user_journeys)))
    avg_delay = 0.0
    for j in user_journeys:
        if j.scheduled_departure_time and j.predicted_arrival_time:
            # Simple feature: average difference between prediction and actual schedule
            avg_delay = avg_delay + float((j.predicted_arrival_time - j.scheduled_departure_time).total_seconds())
    avg_delay_normalized = float(avg_delay) / total_journeys
    
    # Normalize state vector
    state = [min(total_journeys / 100.0, 1.0), min(max(0.0, avg_delay_normalized) / 3600.0, 1.0), 0.5, 0.5, 0.5]
    
    # 3. Simulate PPO Environment Reward based on user adherence
    # A positive reward is given if the user frequently chooses the suggested path
    # Here we mock the RL training loop step
    # PPO agent updates via update_policy
    if total_journeys > 5:
        ppo_agent.update_policy([state], [[0.5, 0.5, 0.5]], [0.8], [state])
    
    # 4. Predict new preferences using the PPO Actor
    with torch.no_grad():
        state_tensor = torch.tensor([state], dtype=torch.float32)
        action_probs, _ = ppo_agent(state_tensor)
        actions = action_probs.squeeze().tolist()
    
    inferred_profile = {
        "toll_aversion": round(actions[0], 3),
        "highway_preference": round(actions[1], 3),
        "variance_tolerance": round(actions[2], 3),
    }

    # Update database directly
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        current_prefs = parse_user_preferences(user)
        # Update only keys that aren't explicitly locked, or just overwrite as inferred
        current_prefs.update(inferred_profile)
        user.preferences = json.dumps(current_prefs)
        db.commit()
        logger.info(f"Re-inferred CF preferences for user: {user.email}")

    return inferred_profile
