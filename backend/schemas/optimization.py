from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional

_IST = timezone(timedelta(hours=5, minutes=30))

def _fmt_ist(unix_ts: float) -> tuple[str, str, str]:
    """Return (time_str, date_str, countdown_str) for a Unix timestamp in IST."""
    import time as _time
    d = datetime.fromtimestamp(unix_ts, tz=_IST)
    now = datetime.now(_IST)

    # Time: e.g. "07:31 am"
    h, m = d.hour, d.minute
    ampm = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    time_str = f"{h12:02d}:{m:02d} {ampm}"

    # Date: Today / Tomorrow / "Sat, Mar 14"
    if d.date() == now.date():
        date_str = "Today"
    elif d.date() == (now + timedelta(days=1)).date():
        date_str = "Tomorrow"
    else:
        date_str = d.strftime("%a, %b %-d")

    # Countdown
    diff = int(unix_ts - _time.time())
    if diff <= 0:
        countdown_str = "Now"
    elif diff < 3600:
        countdown_str = f"in {diff // 60}m"
    else:
        countdown_str = f"in {diff // 3600}h {(diff % 3600) // 60}m"

    return time_str, date_str, countdown_str


class DepartureOption(BaseModel):
    departure_time: float = Field(..., description="Unix timestamp for departure")
    departure_time_ist: str = Field("", description="IST-formatted time, e.g. '07:31 am'")
    departure_date_ist: str = Field("", description="IST-formatted date, e.g. 'Today'")
    countdown_ist: str = Field("", description="Human countdown, e.g. 'in 2h 5m'")
    expected_travel_time: float = Field(..., description="Expected travel time in minutes")
    travel_time_variance: float = Field(..., description="Variance of travel time in minutes^2")
    arrival_probability: float = Field(..., description="Probability of arriving before the deadline (0.0 to 1.0)")
    route_id: Optional[str] = Field(None, description="Recommended route ID or summary")

    def model_post_init(self, __context):
        t, d, c = _fmt_ist(self.departure_time)
        self.departure_time_ist = self.departure_time_ist or t
        self.departure_date_ist = self.departure_date_ist or d
        self.countdown_ist = self.countdown_ist or c


class OptimizationRequest(BaseModel):
    origin: str = Field(..., description="Origin node ID or address")
    destination: str = Field(..., description="Destination node ID or address")
    deadline: float = Field(..., description="Unix timestamp for the hard arrival deadline")
    planning_horizon_hours: int = Field(24, description="Planning horizon in hours")
    user_id: Optional[str] = Field(None, description="Optional User ID to load personal routing behaviour weights")

class OptimizationResponse(BaseModel):
    options: List[DepartureOption] = Field(..., description="Pareto frontier of departure candidates")

class ParkingAlternative(BaseModel):
    location_id: str = Field(..., description="Node ID or descriptor of the parking spot")
    occupancy_probability: float = Field(..., description="Probability of this spot being full (0.0 to 1.0)")
    walking_distance_meters: float = Field(..., description="Distance to final destination in meters")
    cost: float = Field(..., description="Cost metric (e.g., USD or abstract units)")

class ParkingRequest(BaseModel):
    destination: str = Field(..., description="Destination node ID where parking is needed")
    arrival_time: float = Field(..., description="Estimated arrival time as Unix epoch")
    zone_type: str = Field("commercial", description="Land-use zone type (e.g., commercial, residential, transit)")

class ParkingResponse(BaseModel):
    primary_occupancy_probability: float = Field(..., description="Occupancy probability of the requested destination")
    alternatives: List[ParkingAlternative] = Field(..., description="Alternative suggestions if primary is likely full")
