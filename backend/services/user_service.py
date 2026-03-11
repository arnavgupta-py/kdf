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
    Lightweight collaborative filtering inference.
    Maps a user's recent history into a set of probabilistic preference weights.
    For instance, choosing longer but faster routes (or highways) implies
    a low tolerance for travel time variance.
    """
    user_journeys = get_user_journeys(db, user_id, limit=50)
    
    # Target User implicit baselines
    variance_scores = []
    toll_scores = []
    highway_scores = []
    
    for j in user_journeys:
        route_text = str(j.chosen_route or "").lower()
        
        if "expressway" in route_text or "highway" in route_text:
            highway_scores.append(0.8)
            variance_scores.append(0.2) # prefers fast/consistent -> low variance tolerance
        else:
            highway_scores.append(0.2)
            
        if "toll" in route_text:
            toll_scores.append(0.2) # Chose toll -> low aversion
        else:
            toll_scores.append(0.8) # Avoided toll -> high aversion

        if j.scheduled_departure_time and j.predicted_arrival_time:
            buffer = (j.predicted_arrival_time - j.scheduled_departure_time).total_seconds()
            if buffer > 3600:
                variance_scores.append(0.8) # Huge buffer -> high tolerance for sitting in variance
            else:
                variance_scores.append(0.4)

    # Calculate local means
    local_toll = sum(toll_scores) / len(toll_scores) if toll_scores else 0.5
    local_hwy = sum(highway_scores) / len(highway_scores) if highway_scores else 0.5
    local_var = sum(variance_scores) / len(variance_scores) if variance_scores else 0.5

    # Collaborative Smoothing (CF)
    all_users = db.query(User).filter(User.id != user_id).all()
    
    cf_tolls, cf_hwys, cf_vars = [], [], []
    for u in all_users:
        prefs = parse_user_preferences(u)
        if prefs:
            cf_tolls.append(prefs.get("toll_aversion", 0.5))
            cf_hwys.append(prefs.get("highway_preference", 0.5))
            cf_vars.append(prefs.get("variance_tolerance", 0.5))
            
    global_toll = sum(cf_tolls) / len(cf_tolls) if cf_tolls else 0.5
    global_hwy = sum(cf_hwys) / len(cf_hwys) if cf_hwys else 0.5
    global_var = sum(cf_vars) / len(cf_vars) if cf_vars else 0.5

    # Weight by number of journeys: more journeys = trust local, fewer = trust global
    alpha = min(len(user_journeys) / 10.0, 1.0)
    
    inferred_profile = {
        "toll_aversion": round(alpha * local_toll + (1 - alpha) * global_toll, 3),
        "highway_preference": round(alpha * local_hwy + (1 - alpha) * global_hwy, 3),
        "variance_tolerance": round(alpha * local_var + (1 - alpha) * global_var, 3),
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
