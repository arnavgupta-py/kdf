import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db.sqlite import Base
from backend.models.user import User
from backend.models.journey import Journey
from backend.schemas.user import UserCreate
from backend.services.user_service import (
    create_user, get_user_by_email, create_journey, update_journey_completion,
    get_journey, get_user_journeys, delete_journey, infer_user_preferences, update_user_preferences
)


@pytest.fixture
def db_session():
    # Setup an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    session = SessionLocal()
    yield session
    
    # Teardown
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user(db_session):
    user_in = UserCreate(
        email="test_user@example.com",
        password="testpassword",
        full_name="Test User"
    )
    return create_user(db_session, user_in)

def test_journey_crud(db_session, test_user):
    # 1. Create a journey
    journey_data = {
        "origin_lat": 19.0760,
        "origin_lon": 72.8777,
        "dest_lat": 19.0822,
        "dest_lon": 72.8812,
        "chosen_route": "Route via Expressway",
        "scheduled_departure_time": datetime(2026, 3, 11, 8, 0, 0),
        "predicted_arrival_time": datetime(2026, 3, 11, 8, 45, 0)
    }
    journey = create_journey(db_session, test_user.id, journey_data)
    
    assert journey.id is not None
    assert journey.user_id == test_user.id
    assert journey.chosen_route == "Route via Expressway"
    
    # 2. Read - single journey
    fetched_journey = get_journey(db_session, journey.id)
    assert fetched_journey is not None
    assert fetched_journey.id == journey.id

    # 3. Read - user journeys
    user_journeys = get_user_journeys(db_session, test_user.id)
    assert len(user_journeys) == 1
    assert user_journeys[0].id == journey.id

    # 4. Update journey completion with actual arrival and departure times
    actual_departure = datetime(2026, 3, 11, 8, 5, 0)
    actual_arrival = datetime(2026, 3, 11, 8, 55, 0)
    updated_journey = update_journey_completion(
        db_session, 
        journey.id, 
        actual_arrival_time=actual_arrival,
        actual_departure_time=actual_departure
    )
    
    assert updated_journey.actual_arrival_time == actual_arrival
    assert updated_journey.actual_departure_time == actual_departure

    # 5. Delete the journey
    assert delete_journey(db_session, journey.id) is True
    
    # Verify deletion
    assert get_journey(db_session, journey.id) is None
    assert len(get_user_journeys(db_session, test_user.id)) == 0

def test_infer_user_preferences(db_session, test_user):
    # Base user - 0 journeys
    prefs = infer_user_preferences(db_session, test_user.id)
    assert "toll_aversion" in prefs
    assert prefs["toll_aversion"] == 0.5 # Default fallback

    # Create another user to serve as CF weight
    user2_in = UserCreate(email="cf@example.com", password="pwd", full_name="CF")
    user2 = create_user(db_session, user2_in)
    update_user_preferences(db_session, user2, {"toll_aversion": 0.2, "variance_tolerance": 0.9, "highway_preference": 0.8})

    # Since test_user has 0 journeys, profile should just take global CF averge
    prefs2 = infer_user_preferences(db_session, test_user.id)
    assert prefs2["toll_aversion"] == 0.2
    assert prefs2["variance_tolerance"] == 0.9

    # Add 5 journeys for test_user that highly prefer highways/tolls and tolerate variance
    for _ in range(5):
        journey_data = {
            "chosen_route": "highway toll",
            "scheduled_departure_time": datetime(2026, 3, 11, 8, 0, 0),
            "predicted_arrival_time": datetime(2026, 3, 11, 9, 30, 0) # Big buffer (>3600s) -> variance_tolerance local metric gets 0.8
        }
        create_journey(db_session, test_user.id, journey_data)

    # test_user local baseline: 
    # highway = 0.8 (because "highway" in text)
    # toll_aversion = 0.2 (because "toll" in text, low aversion)
    # variance_tolerance = 0.8 (buffer is 1.5 hr > 3600s)
    
    # alpha = length of journeys / 10 = 5 / 10 = 0.5
    # global (other user) averages:
    # toll_aversion = 0.2
    # variance_tolerance = 0.9
    # highway_preference = 0.8
    # 
    # Weighted calculation for var_tol = (0.5 * 0.5) + (0.5 * 0.9) = 0.25 + 0.45 = 0.7
    prefs3 = infer_user_preferences(db_session, test_user.id)
    assert prefs3["toll_aversion"] == 0.2
    assert prefs3["variance_tolerance"] == 0.7
    assert prefs3["highway_preference"] == 0.8

