// Package env defines all data models for the GridMind-RL environment.
package env

// BatchJob represents a pending industrial/commercial batch process job.
type BatchJob struct {
	ID           int `json:"id"`           // Unique job identifier
	DeadlineSlot int `json:"deadline_slot"` // Latest timestep by which the job must run
	Duration     int `json:"duration"`      // Steps the job takes to complete
	PowerDraw    float64 `json:"power_draw"` // kW drawn when job is running
	Scheduled    bool    `json:"scheduled"`  // Whether a time slot has been assigned
	ScheduledAt  int     `json:"scheduled_at"` // Timestep at which job was scheduled (-1 if not yet)
	Completed    bool    `json:"completed"`  // Whether the job finished execution
	MissedDeadline bool  `json:"missed_deadline"` // True if deadline was exceeded
}

// BuildingState holds the full simulation state for a single building.
type BuildingState struct {
	// Core physical state
	IndoorTemperature    float64 `json:"indoor_temperature"`     // °C
	ThermalStorageLevel  float64 `json:"thermal_storage_level"`  // 0.0–1.0 normalized
	ProcessDemand        float64 `json:"process_demand"`          // kW current process need

	// Market & grid signals
	CurrentPrice         float64 `json:"current_price"`          // $/kWh
	GridStressSignal     float64 `json:"grid_stress_signal"`     // 0.0–1.0 (DR urgency)
	CarbonIntensity      float64 `json:"carbon_intensity"`       // gCO2/kWh

	// Temporal
	HourOfDay            int     `json:"hour_of_day"`            // 0–23
	Step                 int     `json:"step"`                   // 0–95 within a 96-step (24h) episode

	// Batch job queue: pending deadlines (raw slots)
	BatchQueue           []int   `json:"batch_queue"`            // deadline slots of pending jobs

	// Running cost tracker
	CumulativeCost       float64 `json:"cumulative_cost"`        // $ total this episode
	CumulativeCarbon     float64 `json:"cumulative_carbon"`      // gCO2 total this episode

	// Internal tracking (not exposed in observation)
	Jobs                 []BatchJob `json:"-"`
	OutdoorTemperature   float64    `json:"-"` // °C for weather perturbation
	PrevHVACLevel        float64    `json:"-"` // for stability penalty
	BaselineCost         float64    `json:"-"` // always-on policy running cost
	BaselineCarbon       float64    `json:"-"` // baseline policy gCO2 (for grading)
	SetpointTemperature  float64    `json:"-"` // target indoor temp (°C)
	MaxHVACPower         float64    `json:"-"` // kW
	MaxStorageCapacity   float64    `json:"-"` // kWh
	ThermalLossRate      float64    `json:"-"` // fraction lost per step
	BuildingID             int        `json:"-"` // which building in federation
	HVACEfficiency       float64    `json:"hvac_efficiency"` // 1.0 = perfect, degrades over time
	HVACDegradationRate  float64    `json:"-"` // e.g. 0.001 per step
	TempObservationNoise float64    `json:"-"` // sensor fault noise added to obs only (not physics)
	LoadShedFraction   float64    `json:"-"` // actual load shed fraction applied (for fault reward)
}

// InstructionCard carries a natural-language task objective for Task 4.
type InstructionCard struct {
	Text    string             `json:"text"`    // human-readable instruction sentence
	Targets map[string]float64 `json:"targets"` // machine-readable KPI targets
	Weights map[string]float64 `json:"weights"` // scoring weights for each target
}

// ObservationModel is the JSON-serializable observation returned on each step/state.
type ObservationModel struct {
	IndoorTemperature   float64          `json:"indoor_temperature"`
	ThermalStorageLevel float64          `json:"thermal_storage_level"`
	ProcessDemand       float64          `json:"process_demand"`
	CurrentPrice        float64          `json:"current_price"`
	GridStressSignal    float64          `json:"grid_stress_signal"`
	CarbonIntensity     float64          `json:"carbon_intensity"`
	HourOfDay           int              `json:"hour_of_day"`
	BatchQueue          []int            `json:"batch_queue"`
	CumulativeCost      float64          `json:"cumulative_cost"`
	Step                int              `json:"step"`
	BuildingID          int              `json:"building_id"`
	HVACEfficiency      float64          `json:"hvac_efficiency"`
	InstructionCard     *InstructionCard `json:"instruction_card,omitempty"` // populated for Task 4 only
	ActiveFaults        []string         `json:"active_faults,omitempty"`    // human-readable alarm strings for active faults
}

// ActionModel is the parsed agent action for a single step.
type ActionModel struct {
	HVACPowerLevel     float64 `json:"hvac_power_level"`    // 0.0–1.0
	ThermalChargeRate  float64 `json:"thermal_charge_rate"` // -1.0 to 1.0
	BatchJobSlot       int     `json:"batch_job_slot"`      // 0–4 (0=now, 1–4=defer)
	LoadShedFraction   float64 `json:"load_shed_fraction"`  // 0.0–0.5
	BuildingID         int     `json:"building_id"`         // which building to act on
}

// RewardComponents holds the individual components of the dense reward signal.
type RewardComponents struct {
	CostSavings        float64 `json:"cost_savings"`         // negative = expensive
	TempConstraint   float64 `json:"temp_constraint"`     // positive = within bounds
	GridResponse    float64 `json:"grid_response"`       // bonus for DR compliance
	DeadlinePenalty  float64 `json:"deadline_penalty"`    // negative for missed jobs
	EfficiencyBonus float64 `json:"efficiency_bonus"`    // storage arbitrage
	StabilityPenalty float64 `json:"stability_penalty"`   // HVAC oscillation penalty
	CarbonReward    float64 `json:"carbon_reward"`       // low-carbon bonus
	InstructionReward float64 `json:"instruction_reward"`  // Task 4: instruction-following score
	FaultMitigation float64 `json:"fault_mitigation"`  // Track 3: reward for proper fault response
	Total           float64 `json:"total"`
}

// StepResponse is the full HTTP body returned from POST /step.
type StepResponse struct {
	Observation ObservationModel `json:"observation"`
	Reward      float64          `json:"reward"`
	Done        bool             `json:"done"`
	Info        StepInfo         `json:"info"`
}

// StepInfo carries auxiliary information per step.
type StepInfo struct {
	RewardComponents RewardComponents `json:"reward_components"`
	EnergyUsed       float64          `json:"energy_used_kwh"`
	CarbonEmitted    float64          `json:"carbon_emitted_gco2"`
	PriceSignal      float64          `json:"price_signal"`
	GridStress       float64          `json:"grid_stress"`
	BatchCompleted   []int            `json:"batch_completed"`   // IDs completed this step
	BatchMissed      []int            `json:"batch_missed"`      // IDs that missed deadline
	Episode          int              `json:"episode"`
	Step             int              `json:"step"`
}

// ResetRequest is the JSON body for POST /reset.
type ResetRequest struct {
	Seed       *int64 `json:"seed,omitempty"`       // optional random seed
	TaskID     int    `json:"task_id"`              // 1, 2, or 3
	Difficulty string `json:"difficulty,omitempty"` // "easy", "medium", "hard" or "" (auto)
	NumBuildings int  `json:"num_buildings,omitempty"` // 1–3 for federation
}

// ResetResponse is returned from POST /reset.
type ResetResponse struct {
	Observations    []ObservationModel `json:"observations"`               // one per building
	Episode         int                `json:"episode"`
	TaskID          int                `json:"task_id"`
	Seed            int64              `json:"seed"`
	InstructionCard *InstructionCard   `json:"instruction_card,omitempty"` // populated for Task 4 only
}

// StateResponse is returned from GET /state.
type StateResponse struct {
	Buildings    []BuildingStatePublic `json:"buildings"`
	PriceCurve   []float64            `json:"price_curve_episode"`    // full episode ToU prices
	CarbonCurve  []float64            `json:"carbon_curve_episode"`   // full episode carbon intensities
	Episode      int                  `json:"episode"`
	Step         int                  `json:"step"`
	TaskID       int                  `json:"task_id"`
	Done         bool                 `json:"done"`
	Seed         int64                `json:"seed"`
}

// BuildingStatePublic is the dashboard-friendly full state per building.
type BuildingStatePublic struct {
	ObservationModel
	OutdoorTemperature  float64    `json:"outdoor_temperature"`
	SetpointTemperature float64    `json:"setpoint_temperature"`
	BaselineCost        float64    `json:"baseline_cost"`
	BaselineCarbon      float64    `json:"baseline_carbon"`
	CumulativeCarbon    float64    `json:"cumulative_carbon"`
	Jobs                []BatchJob `json:"jobs"`
	// History for chart rendering
	TempHistory         []float64  `json:"temp_history"`
	CostHistory         []float64  `json:"cost_history"`
	HVACHistory         []float64  `json:"hvac_history"`
	LoadShedHistory     []float64  `json:"load_shed_history"`
	RewardHistory       []RewardComponents `json:"reward_history"`
}

// ReplayEntry records a single timestep for episode replay export.
type ReplayEntry struct {
	Step        int              `json:"step"`
	Observation ObservationModel `json:"observation"`
	Action      ActionModel      `json:"action"`
	Reward      float64          `json:"reward"`
	Components  RewardComponents `json:"components"`
	Done        bool             `json:"done"`
}

// EpisodeGrade is the final grade returned for a completed episode.
type EpisodeGrade struct {
	TaskID          int                    `json:"task_id"`
	Score           float64                `json:"score"`           // 0.0–1.0
	SubScores       map[string]float64     `json:"sub_scores"`
	ExploitDetected bool                   `json:"exploit_detected"`
	PenaltyApplied  float64                `json:"penalty_applied"`
	Details         map[string]interface{} `json:"details"`
}

// BuildingSummary is a compact per-building view used by the coordinator.
type BuildingSummary struct {
	BuildingID          int     `json:"building_id"`
	CurrentDemandKW     float64 `json:"current_demand_kw"`
	IndoorTemperature   float64 `json:"indoor_temperature"`
	ThermalStorageLevel float64 `json:"thermal_storage_level"`
	CumulativeCost      float64 `json:"cumulative_cost"`
	GridStressSignal    float64 `json:"grid_stress_signal"`
	PriceMultiplier     float64 `json:"price_multiplier"` // set by coordinator (default 1.0)
}

// FeederState is the aggregate fleet view returned by GET /feeder.
// An LLM coordinator reads this to decide per-building price signals.
type FeederState struct {
	TotalDemandKW     float64           `json:"total_demand_kw"`
	FeederLimitKW     float64           `json:"feeder_limit_kw"`
	FeederOverload    bool              `json:"feeder_overload"`
	UtilizationPct    float64           `json:"utilization_pct"`  // TotalDemandKW / FeederLimitKW * 100
	Buildings         []BuildingSummary `json:"buildings"`
	PriceCurveHourly  []float64         `json:"price_curve_hourly"` // downsampled 24-point curve
	Step              int               `json:"step"`
	Episode           int               `json:"episode"`
}

// CoordinateRequest is the JSON body for POST /coordinate.
type CoordinateRequest struct {
	PriceMultipliers []float64 `json:"price_multipliers"` // one per building, default 1.0
}
