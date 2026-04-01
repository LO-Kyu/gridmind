"""
GridMind-RL OpenEnv Pydantic models.
These types mirror the Go structs exactly for full schema compliance.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class BatchJob(BaseModel):
    id: int
    deadline_slot: int
    duration: int
    power_draw: float
    scheduled: bool
    scheduled_at: int
    completed: bool
    missed_deadline: bool


class ObservationModel(BaseModel):
    """Full observation returned on each step / GET /state."""
    indoor_temperature: float = Field(..., description="Current building indoor temperature (°C)")
    thermal_storage_level: float = Field(..., ge=0.0, le=1.0, description="Thermal storage fill level (0–1)")
    process_demand: float = Field(..., ge=0.0, description="Current process power demand (kW)")
    current_price: float = Field(..., gt=0.0, description="Real-time electricity price ($/kWh)")
    grid_stress_signal: float = Field(..., ge=0.0, le=1.0, description="Utility demand-response urgency (0–1)")
    carbon_intensity: float = Field(..., ge=0.0, description="Grid carbon intensity (gCO2/kWh)")
    hour_of_day: int = Field(..., ge=0, le=23, description="Current hour of day (0–23)")
    batch_queue: List[int] = Field(default_factory=list, description="Deadline slots of pending batch jobs")
    cumulative_cost: float = Field(..., ge=0.0, description="Running energy cost this episode ($)")
    step: int = Field(..., ge=0, description="Current timestep (0–287)")
    building_id: int = Field(default=0, description="Building index in federation")


class ActionModel(BaseModel):
    """Agent action for a single timestep."""
    hvac_power_level: float = Field(..., ge=0.0, le=1.0, description="HVAC fraction of max power (0–1)")
    thermal_charge_rate: float = Field(..., ge=-1.0, le=1.0, description="Storage charge (+) or discharge (-) rate")
    batch_job_slot: int = Field(..., ge=0, le=4, description="Time slot offset for next batch job (0=now, 1–4=defer)")
    load_shed_fraction: float = Field(..., ge=0.0, le=0.5, description="Fraction of non-critical load to shed (0–0.5)")
    building_id: int = Field(default=0, description="Building index this action targets")

    @field_validator("hvac_power_level")
    @classmethod
    def clamp_hvac(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("thermal_charge_rate")
    @classmethod
    def clamp_charge(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))

    @field_validator("load_shed_fraction")
    @classmethod
    def clamp_shed(cls, v: float) -> float:
        return max(0.0, min(0.5, v))


class RewardComponents(BaseModel):
    """Individual reward signal components."""
    cost_savings: float = Field(..., description="Negative reward for energy cost")
    temp_constraint: float = Field(..., description="Positive if temperature within bounds")
    grid_response: float = Field(..., description="Bonus for shedding during high grid stress")
    deadline_penalty: float = Field(..., description="Negative for missed batch deadlines")
    efficiency_bonus: float = Field(..., description="Storage arbitrage bonus")
    stability_penalty: float = Field(..., description="Penalty for rapid HVAC oscillation")
    carbon_reward: float = Field(..., description="Low-carbon operation bonus")
    total: float = Field(..., description="Weighted sum of all components")


class StepInfo(BaseModel):
    """Auxiliary information returned at each step."""
    reward_components: RewardComponents
    energy_used_kwh: float
    carbon_emitted_gco2: float
    price_signal: float
    grid_stress: float
    batch_completed: List[int] = Field(default_factory=list)
    batch_missed: List[int] = Field(default_factory=list)
    episode: int
    step: int


class StepResponse(BaseModel):
    """Full response from POST /step."""
    observation: ObservationModel
    reward: float
    done: bool
    info: StepInfo


class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    task_id: int = Field(1, ge=1, le=3, description="Task to run (1=easy, 2=medium, 3=hard)")
    difficulty: Optional[str] = Field(None, description="Override difficulty: easy/medium/hard")
    num_buildings: int = Field(1, ge=1, le=3, description="Number of buildings in federation")


class ResetResponse(BaseModel):
    """Response from POST /reset."""
    observations: List[ObservationModel]
    episode: int
    task_id: int
    seed: int


class BuildingStatePublic(BaseModel):
    """Full building state including history for dashboard rendering."""
    # ObservationModel fields (flattened)
    indoor_temperature: float
    thermal_storage_level: float
    process_demand: float
    current_price: float
    grid_stress_signal: float
    carbon_intensity: float
    hour_of_day: int
    batch_queue: List[int] = Field(default_factory=list)
    cumulative_cost: float
    step: int
    building_id: int
    # Extended state
    outdoor_temperature: float
    setpoint_temperature: float
    baseline_cost: float
    cumulative_carbon: float
    jobs: List[BatchJob] = Field(default_factory=list)
    # History arrays
    temp_history: List[float] = Field(default_factory=list)
    cost_history: List[float] = Field(default_factory=list)
    hvac_history: List[float] = Field(default_factory=list)
    load_shed_history: List[float] = Field(default_factory=list)
    reward_history: List[RewardComponents] = Field(default_factory=list)


class StateResponse(BaseModel):
    """Full environment state from GET /state."""
    buildings: List[BuildingStatePublic]
    price_curve_episode: List[float]
    carbon_curve_episode: List[float]
    episode: int
    step: int
    task_id: int
    done: bool
    seed: int


class TaskConfig(BaseModel):
    """Task configuration."""
    id: int
    name: str
    description: str
    difficulty: str
    weights: Dict[str, float]


class EpisodeGrade(BaseModel):
    """Graded episode result."""
    task_id: int
    score: float = Field(..., ge=0.0, le=1.0)
    sub_scores: Dict[str, float]
    exploit_detected: bool
    penalty_applied: float
    details: Dict[str, Any]


# ── Action space schema (for LLM prompting) ────────────────────────────────
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "hvac_power_level": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Fraction of max HVAC power (0=off, 1=full power)"
        },
        "thermal_charge_rate": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Charge (+) or discharge (-) thermal storage at this fraction of max rate"
        },
        "batch_job_slot": {
            "type": "integer",
            "minimum": 0,
            "maximum": 4,
            "description": "Schedule next batch job: 0=run now, 1-4=defer by N 15-min intervals"
        },
        "load_shed_fraction": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 0.5,
            "description": "Fraction of non-critical load to shed during this step (0=no shedding)"
        },
        "building_id": {
            "type": "integer",
            "minimum": 0,
            "description": "Which building to apply this action to (0 for single-building mode)"
        }
    },
    "required": ["hvac_power_level", "thermal_charge_rate", "batch_job_slot", "load_shed_fraction"]
}

# ── Observation space schema ───────────────────────────────────────────────
OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "indoor_temperature": {"type": "number", "description": "Indoor temperature °C"},
        "thermal_storage_level": {"type": "number", "minimum": 0, "maximum": 1},
        "process_demand": {"type": "number", "description": "Process power demand kW"},
        "current_price": {"type": "number", "description": "Electricity price $/kWh"},
        "grid_stress_signal": {"type": "number", "minimum": 0, "maximum": 1},
        "carbon_intensity": {"type": "number", "description": "Grid carbon intensity gCO2/kWh"},
        "hour_of_day": {"type": "integer", "minimum": 0, "maximum": 23},
        "batch_queue": {"type": "array", "items": {"type": "integer"}},
        "cumulative_cost": {"type": "number"},
        "step": {"type": "integer"},
        "building_id": {"type": "integer"}
    }
}
