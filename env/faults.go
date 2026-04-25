// Package env defines fault events and injection logic for GridMind-RL.
// Faults are rare but high-impact events that force agents to adapt their strategies.
// They target the Wild Card + World Modeling judging themes.
package env

import "math/rand"

// FaultType identifies the kind of fault event.
type FaultType string

const (
	FaultChillerFailure FaultType = "chiller_failure" // HVAC efficiency drops
	FaultGridOutage     FaultType = "grid_outage"      // Price spike + max grid stress
	FaultSensorFault    FaultType = "sensor_fault"     // Observation noise on temperature
	FaultTariffSpike    FaultType = "tariff_spike"     // Flash electricity price surge
)

// FaultEvent describes a single active fault during an episode.
type FaultEvent struct {
	Type        FaultType `json:"type"`
	StartStep   int       `json:"start_step"`
	EndStep     int       `json:"end_step"` // exclusive
	Severity    float64   `json:"severity"` // 0.0–1.0
	Description string    `json:"description"`
}

// IsActive returns true if the fault is active at the given step.
func (f *FaultEvent) IsActive(step int) bool {
	return step >= f.StartStep && step < f.EndStep
}

// FaultSchedule holds all faults scheduled for an episode.
type FaultSchedule struct {
	Events []FaultEvent `json:"events"`
}

// ActiveAt returns all fault events active at the given step.
func (fs *FaultSchedule) ActiveAt(step int) []FaultEvent {
	var active []FaultEvent
	for _, e := range fs.Events {
		if e.IsActive(step) {
			active = append(active, e)
		}
	}
	return active
}

// GenerateFaultSchedule creates a randomised schedule of fault events for an episode.
// Probability and severity are scaled by difficulty level.
// Guarantees at least one fault fires in hard mode.
func GenerateFaultSchedule(rng *rand.Rand, difficulty string) *FaultSchedule {
	schedule := &FaultSchedule{}

	// Base probabilities per fault type - increased for hard mode
	type faultSpec struct {
		fType    FaultType
		probEasy float64
		probMed  float64
		probHard float64
		minDur   int // steps
		maxDur   int
	}

	specs := []faultSpec{
		{FaultChillerFailure, 0.05, 0.15, 0.45, 8, 24},
		{FaultGridOutage, 0.05, 0.10, 0.45, 4, 12},
		{FaultSensorFault, 0.08, 0.15, 0.45, 6, 20},
		{FaultTariffSpike, 0.10, 0.20, 0.50, 1, 4},
	}

	for _, spec := range specs {
		prob := spec.probEasy
		switch difficulty {
		case "medium":
			prob = spec.probMed
		case "hard":
			prob = spec.probHard
		}

		if rng.Float64() > prob {
			continue // no fault of this type this episode
		}

		// Random start time (avoid very first and last 10 steps)
		maxStart := EpisodeSteps - spec.maxDur - 10
		if maxStart < 10 {
			maxStart = 10
		}
		start := 10 + rng.Intn(maxStart)
		dur := spec.minDur + rng.Intn(spec.maxDur-spec.minDur+1)
		end := start + dur
		if end > EpisodeSteps {
			end = EpisodeSteps
		}
		severity := 0.4 + rng.Float64()*0.6 // 0.4–1.0

		event := FaultEvent{
			Type:      spec.fType,
			StartStep: start,
			EndStep:   end,
			Severity:  severity,
		}

		switch spec.fType {
		case FaultChillerFailure:
			event.Description = "⚠️ Chiller unit failure — HVAC operating at reduced capacity."
		case FaultGridOutage:
			event.Description = "🔴 Grid brownout — extreme price spike and critical stress signal."
		case FaultSensorFault:
			event.Description = "⚡ Temperature sensor malfunction — indoor temperature readings unreliable."
		case FaultTariffSpike:
			event.Description = "💸 Emergency tariff spike — electricity price has surged. Minimize consumption immediately."
		}

		schedule.Events = append(schedule.Events, event)
	}

	// Force at least one fault in hard mode if schedule is empty
	if len(schedule.Events) == 0 && difficulty == "hard" {
		schedule.Events = append(schedule.Events, FaultEvent{
			Type:        FaultTariffSpike,
			StartStep:   20,
			EndStep:     23,
			Severity:    0.6,
			Description: "Unexpected tariff spike — immediate load response required",
		})
	}

	return schedule
}

// ApplyFaults modifies environment signals based on active faults for the current step.
// It returns a list of active fault descriptions for the observation.
func ApplyFaults(b *BuildingState, schedule *FaultSchedule, step int, rng *rand.Rand) []string {
	if schedule == nil {
		return nil
	}
	active := schedule.ActiveAt(step)
	if len(active) == 0 {
		// Reset noise when no fault active
		b.TempObservationNoise = 0.0
		return nil
	}

	descriptions := make([]string, 0, len(active))
	for _, fault := range active {
		switch fault.Type {
		case FaultChillerFailure:
			// Reduce effective HVAC power — the building state's max power is scaled down
			// The physics engine uses MaxHVACPower; we reduce it proportionally to severity.
			b.MaxHVACPower = MaxHVACPowerKW * (1.0 - fault.Severity*0.8)

		case FaultGridOutage:
			// Force maximum grid stress and multiply the price to simulate outage conditions
			b.GridStressSignal = 1.0
			b.CurrentPrice = b.CurrentPrice * (1.0 + fault.Severity*3.0)

		case FaultSensorFault:
			// Add noise to the indoor temperature reading (observation only, not physics)
			// This affects what the agent sees but not the actual physics
			b.TempObservationNoise = (rng.Float64()*2 - 1) * 5.0 * fault.Severity

		case FaultTariffSpike:
			b.CurrentPrice = b.CurrentPrice * (1.0 + fault.Severity*4.0)
		}

		descriptions = append(descriptions, fault.Description)
	}
	return descriptions
}