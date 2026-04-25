// Package env implements the GridMind-RL simulation core.
// It models a multi-building industrial/commercial energy management system
// with stochastic electricity prices, thermal dynamics, and batch job scheduling.
package env

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

const (
	EpisodeSteps    = 96   // 24 hours × 15-min intervals (96 × 0.25h = 24h)
	StepDurationHrs = 0.25 // each step = 15 minutes = 0.25 h
	MaxBuildings    = 3
	DefaultSetpoint = 21.0  // °C comfortable indoor temp
	TMinDefault     = 19.0  // °C lower bound
	TMaxDefault     = 23.0  // °C upper bound
	MaxHVACPowerKW  = 50.0  // kW per building
	MaxStorageKWh   = 100.0 // kWh thermal storage capacity
	StorageLossRate = 0.005 // fraction lost per step (thermal dissipation)
	MaxBatchJobs    = 5     // max concurrent batch jobs per building
)

// Environment is the thread-safe top-level simulation manager.
type Environment struct {
	mu           sync.RWMutex
	rng          *rand.Rand
	seed         int64
	episode      int
	step         int
	done         bool
	taskID       int
	difficulty   string
	numBuildings int

	Buildings        []*BuildingState
	PriceCurve       [EpisodeSteps]float64
	CarbonCurve      [EpisodeSteps]float64
	Replay           []ReplayEntry
	LastActions      []ActionModel
	InstructionCard  *InstructionCard // set for Task 4 episodes
	FaultSchedule    *FaultSchedule   // randomised fault events for this episode
	PriceMultipliers []float64        // per-building multipliers set by coordinator (default 1.0)

	// History for dashboard rendering (per building)
	TempHistory     [][]float64
	CostHistory     [][]float64
	HVACHistory     [][]float64
	LoadShedHistory [][]float64
	RewardHistory   [][]RewardComponents

	// Exploit detection counters
	totalShedSteps     []int
	thermalCycleCounts []int
	prevChargeRates    []float64
}

// NewEnvironment creates an initialised (but not reset) environment.
func NewEnvironment() *Environment {
	seed := time.Now().UnixNano()
	return &Environment{
		rng:          rand.New(rand.NewSource(seed)),
		seed:         seed,
		taskID:       1,
		difficulty:   "easy",
		numBuildings: 1,
	}
}

// Reset initializes a new episode. Thread-safe.
func (e *Environment) Reset(req ResetRequest) ResetResponse {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Apply seed
	if req.Seed != nil {
		e.seed = *req.Seed
	} else {
		e.seed = time.Now().UnixNano()
	}
	e.rng = rand.New(rand.NewSource(e.seed))

	// Apply task and difficulty
	e.taskID = req.TaskID
	if e.taskID < 1 || e.taskID > 4 {
		e.taskID = 1
	}
	e.difficulty = req.Difficulty
	if e.difficulty == "" {
		switch e.taskID {
		case 1:
			e.difficulty = "easy"
		case 2:
			e.difficulty = "medium"
		case 3, 4:
			e.difficulty = "hard"
		}
	}

	// Number of buildings (federation)
	e.numBuildings = req.NumBuildings
	if e.numBuildings < 1 {
		e.numBuildings = 1
	}
	if e.numBuildings > MaxBuildings {
		e.numBuildings = MaxBuildings
	}

	e.episode++
	e.step = 0
	e.done = false
	e.Replay = make([]ReplayEntry, 0, EpisodeSteps)
	e.LastActions = make([]ActionModel, e.numBuildings)

	// Generate price and carbon curves for this episode
	e.generatePriceCurve()
	e.generateCarbonCurve()

	// Initialise buildings
	e.Buildings = make([]*BuildingState, e.numBuildings)
	e.TempHistory = make([][]float64, e.numBuildings)
	e.CostHistory = make([][]float64, e.numBuildings)
	e.HVACHistory = make([][]float64, e.numBuildings)
	e.LoadShedHistory = make([][]float64, e.numBuildings)
	e.RewardHistory = make([][]RewardComponents, e.numBuildings)
	e.totalShedSteps = make([]int, e.numBuildings)
	e.thermalCycleCounts = make([]int, e.numBuildings)
	e.prevChargeRates = make([]float64, e.numBuildings)

	for i := range e.Buildings {
		e.Buildings[i] = e.newBuildingState(i)
		e.TempHistory[i] = make([]float64, 0, EpisodeSteps)
		e.CostHistory[i] = make([]float64, 0, EpisodeSteps)
		e.HVACHistory[i] = make([]float64, 0, EpisodeSteps)
		e.LoadShedHistory[i] = make([]float64, 0, EpisodeSteps)
		e.RewardHistory[i] = make([]RewardComponents, 0, EpisodeSteps)
	}

	// Initialise coordinator price multipliers to 1.0
	e.PriceMultipliers = make([]float64, e.numBuildings)
	for i := range e.PriceMultipliers {
		e.PriceMultipliers[i] = 1.0
	}

	// Generate instruction card for Task 4
	e.InstructionCard = nil
	if e.taskID == 4 {
		e.InstructionCard = GenerateInstructionCard(e.rng)
	}

	// Generate fault schedule for all tasks (probability varies by difficulty)
	e.FaultSchedule = GenerateFaultSchedule(e.rng, e.difficulty)

	obs := make([]ObservationModel, e.numBuildings)
	for i, b := range e.Buildings {
		obs[i] = e.buildObservation(b)
	}

	return ResetResponse{
		Observations:    obs,
		Episode:         e.episode,
		TaskID:          e.taskID,
		Seed:            e.seed,
		InstructionCard: e.InstructionCard,
	}
}

// Step advances the simulation by one timestep for all buildings. Thread-safe.
func (e *Environment) Step(actions []ActionModel) ([]StepResponse, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.done {
		return nil, true
	}

	// Validate and clamp actions
	for i := range actions {
		e.clampAction(&actions[i])
		if i < e.numBuildings {
			e.LastActions[i] = actions[i]
		}
	}

	responses := make([]StepResponse, e.numBuildings)
	for i, b := range e.Buildings {
		var act ActionModel
		// Find action for this building (by building_id or by index)
		act = e.findAction(actions, i)
		responses[i] = e.stepBuilding(b, act, i)
	}

	e.step++
	if e.step >= EpisodeSteps {
		e.done = true
	}

	// Record replay entry (aggregate of all buildings, first building primary)
	if len(responses) > 0 {
		entry := ReplayEntry{
			Step:        e.step - 1,
			Observation: responses[0].Observation,
			Action:      e.LastActions[0],
			Reward:      responses[0].Reward,
			Components:  responses[0].Info.RewardComponents,
			Done:        e.done,
		}
		e.Replay = append(e.Replay, entry)
	}

	return responses, e.done
}

// GetState returns a full snapshot of environment state. Thread-safe (read lock).
func (e *Environment) GetState() StateResponse {
	e.mu.RLock()
	defer e.mu.RUnlock()

	buildings := make([]BuildingStatePublic, e.numBuildings)
	for i, b := range e.Buildings {
		pub := BuildingStatePublic{
			ObservationModel:    e.buildObservation(b),
			OutdoorTemperature:  b.OutdoorTemperature,
			SetpointTemperature: b.SetpointTemperature,
			BaselineCost:        b.BaselineCost,
			BaselineCarbon:      b.BaselineCarbon,
			CumulativeCarbon:    b.CumulativeCarbon,
			Jobs:                b.Jobs,
		}
		if i < len(e.TempHistory) {
			pub.TempHistory = e.TempHistory[i]
			pub.CostHistory = e.CostHistory[i]
			pub.HVACHistory = e.HVACHistory[i]
			pub.LoadShedHistory = e.LoadShedHistory[i]
			pub.RewardHistory = e.RewardHistory[i]
		}
		buildings[i] = pub
	}

	priceCurve := make([]float64, EpisodeSteps/4)
	carbonCurve := make([]float64, EpisodeSteps/4)
	for h := 0; h < EpisodeSteps/4; h++ {
		stepIdx := h * 4
		if stepIdx < EpisodeSteps {
			priceCurve[h] = e.PriceCurve[stepIdx]
			carbonCurve[h] = e.CarbonCurve[stepIdx]
		}
	}

	return StateResponse{
		Buildings:       buildings,
		PriceCurve:      priceCurve,
		CarbonCurve:     carbonCurve,
		Episode:         e.episode,
		Step:            e.step,
		TaskID:          e.taskID,
		Done:            e.done,
		Seed:            e.seed,
		InstructionCard: e.InstructionCard,
	}
}

// GetReplay returns the full episode replay. Thread-safe.
func (e *Environment) GetReplay() []ReplayEntry {
	e.mu.RLock()
	defer e.mu.RUnlock()
	result := make([]ReplayEntry, len(e.Replay))
	copy(result, e.Replay)
	return result
}

// ──────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────

func (e *Environment) newBuildingState(id int) *BuildingState {
	// Randomise initial conditions slightly
	initTemp := DefaultSetpoint + (e.rng.Float64()-0.5)*2.0
	storageLevel := 0.3 + e.rng.Float64()*0.4  // start 30–70% full
	outdoorTemp := 15.0 + e.rng.Float64()*15.0 // 15–30 °C

	b := &BuildingState{
		BuildingID:          id,
		IndoorTemperature:   initTemp,
		ThermalStorageLevel: storageLevel,
		ProcessDemand:       10.0 + e.rng.Float64()*20.0,
		CurrentPrice:        e.PriceCurve[0],
		GridStressSignal:    0.0,
		CarbonIntensity:     e.CarbonCurve[0],
		HourOfDay:           0,
		Step:                0,
		BatchQueue:          []int{},
		CumulativeCost:      0.0,
		CumulativeCarbon:    0.0,
		OutdoorTemperature:  outdoorTemp,
		PrevHVACLevel:       0.5,
		BaselineCost:        0.0,
		BaselineCarbon:      0.0,
		SetpointTemperature: DefaultSetpoint,
		MaxHVACPower:        MaxHVACPowerKW,
		MaxStorageCapacity:  MaxStorageKWh,
		ThermalLossRate:     StorageLossRate,
		HVACEfficiency:      1.0,
		HVACDegradationRate: 0.0005 + e.rng.Float64()*0.001, // 0.05% to 0.15% per step
	}

	// Spawn batch jobs based on difficulty
	b.Jobs = e.generateBatchJobs()
	b.BatchQueue = pendingDeadlines(b.Jobs)
	return b
}

func (e *Environment) generateBatchJobs() []BatchJob {
	numJobs := 3
	switch e.difficulty {
	case "medium":
		numJobs = 4
	case "hard":
		numJobs = 5
	}

	jobs := make([]BatchJob, numJobs)
	for i := range jobs {
		// Deadline spread across episode (leave slack at end for duration)
		span := EpisodeSteps - 12
		if span < 8 {
			span = 8
		}
		deadline := 4 + e.rng.Intn(span)
		jobs[i] = BatchJob{
			ID:             i + 1,
			DeadlineSlot:   deadline,
			Duration:       1 + e.rng.Intn(3),
			PowerDraw:      5.0 + e.rng.Float64()*15.0,
			Scheduled:      false,
			ScheduledAt:    -1,
			Completed:      false,
			MissedDeadline: false,
		}
	}
	return jobs
}

// generatePriceCurve creates a stochastic Time-of-Use price curve for the episode.
func (e *Environment) generatePriceCurve() {
	// Base ToU: low overnight, moderate morning, high peak (8-12, 17-21), low night
	volatility := 0.1
	switch e.difficulty {
	case "medium":
		volatility = 0.2
	case "hard":
		volatility = 0.35
	}

	// Random peak window shift (±2 hours) for stochasticity
	morningPeakShift := e.rng.Intn(5) - 2
	eveningPeakShift := e.rng.Intn(5) - 2

	for s := 0; s < EpisodeSteps; s++ {
		hour := (s / 4)
		base := touPrice(hour, morningPeakShift, eveningPeakShift)
		noise := (e.rng.Float64()*2 - 1) * volatility * base
		price := math.Max(0.02, base+noise)
		e.PriceCurve[s] = price
	}
}

// touPrice returns the base time-of-use price for a given hour.
// Price schedule ($/kWh):
//   00:00–06:00  0.035  Deep off-peak (lowest demand)
//   06:00–08:00  0.070  Ramp-up (industry + residential start)
//   08:00–12:00  0.200  Morning peak (industrial load high)
//   12:00–17:00  0.120  Solar + stabilised demand shoulder
//   17:00–21:00  0.310  True peak (highest grid stress)
//   21:00–24:00  0.093  Declining demand
func touPrice(hour, morningShift, eveningShift int) float64 {
	morningPeakStart := 8 + morningShift
	morningPeakEnd := 12 + morningShift
	eveningPeakStart := 17 + eveningShift
	eveningPeakEnd := 21 + eveningShift

	switch {
	case hour >= morningPeakStart && hour < morningPeakEnd:
		return 0.20 // Morning peak: 0.16–0.24 $/kWh
	case hour >= eveningPeakStart && hour < eveningPeakEnd:
		return 0.31 // Evening peak: 0.26–0.36 $/kWh
	case hour >= morningPeakEnd && hour < eveningPeakStart:
		return 0.12 // Solar/shoulder: 0.09–0.15 $/kWh
	case hour >= 6 && hour < morningPeakStart:
		return 0.07 // Ramp-up: 0.055–0.085 $/kWh
	case hour >= eveningPeakEnd:
		return 0.093 // Declining: 0.075–0.11 $/kWh
	default:
		return 0.035 // Deep off-peak (0–6 AM): 0.028–0.042 $/kWh
	}
}

// generateCarbonCurve creates a realistic carbon intensity curve (gCO2/kWh).
// Correlates roughly with price: higher price = more peaker plants = higher carbon.
func (e *Environment) generateCarbonCurve() {
	for s := 0; s < EpisodeSteps; s++ {
		price := e.PriceCurve[s]
		// Map price range [0.028, 0.36] → carbon [150, 600] gCO2/kWh
		carbon := 150.0 + (price-0.028)/(0.36-0.028)*(600.0-150.0)
		noise := (e.rng.Float64()*2 - 1) * 30.0
		e.CarbonCurve[s] = math.Max(100.0, carbon+noise)
	}
}

// stepBuilding advances a single building by one timestep.
func (e *Environment) stepBuilding(b *BuildingState, act ActionModel, idx int) StepResponse {
	s := e.step

	// Update environmental signals from curves
	b.CurrentPrice = e.PriceCurve[s] * e.PriceMultipliers[idx]
	b.CarbonIntensity = e.CarbonCurve[s]
	b.HourOfDay = (s / 4) % 24

	// Restore defaults before applying faults (allows recovery when fault ends)
	b.MaxHVACPower = MaxHVACPowerKW

	// Apply fault events for this step (modifies price, stress, HVAC capacity)
	activeFaultDescs := ApplyFaults(b, e.FaultSchedule, s, e.rng)
	_ = activeFaultDescs // stored for use in buildObservation via FaultSchedule.ActiveAt

	// Stochastic grid stress events (more frequent in hard mode).
	// Note: FaultGridOutage sets GridStressSignal=1.0 inside ApplyFaults.
	// We only overwrite it from the stochastic model if no outage is active.
	hasGridFault := false
	if e.FaultSchedule != nil {
		for _, f := range e.FaultSchedule.ActiveAt(s) {
			if f.Type == FaultGridOutage {
				hasGridFault = true
				break
			}
		}
	}
	if !hasGridFault {
		b.GridStressSignal = e.updateGridStress(s)
	}

	// Weather perturbation: outdoor temp drifts sinusoidally + noise
	b.OutdoorTemperature = e.updateOutdoorTemp(s)

	// Process demand fluctuation
	b.ProcessDemand = e.updateProcessDemand(s)

	// ----- Apply actions -----

	// 0. Degrade HVAC efficiency
	b.HVACEfficiency = math.Max(0.5, b.HVACEfficiency-b.HVACDegradationRate)

	// 1. HVAC: heats/cools building toward setpoint
	hvacPower := act.HVACPowerLevel * b.MaxHVACPower * b.HVACEfficiency // kW

	// 2. Thermal storage: charge or discharge
	chargeKW := act.ThermalChargeRate * b.MaxHVACPower * 0.3 // max 30% of HVAC for storage
	newStorageEnergy := b.ThermalStorageLevel*b.MaxStorageCapacity + chargeKW*StepDurationHrs
	// Apply thermal losses
	newStorageEnergy *= (1.0 - b.ThermalLossRate)
	newStorageEnergy = math.Max(0, math.Min(b.MaxStorageCapacity, newStorageEnergy))
	b.ThermalStorageLevel = newStorageEnergy / b.MaxStorageCapacity

	// 3. Load shedding
	clampedShed := math.Max(0, math.Min(0.5, act.LoadShedFraction))
	shedKW := clampedShed * b.ProcessDemand

	// 4. Batch job scheduling
	batchCompleted, batchMissed := e.updateBatchJobs(b, act.BatchJobSlot, s)

	// ----- Thermal dynamics -----
	// First-order setpoint-driven model:
	// HVAC drives temperature toward setpoint; higher power = stronger effect.
	// At HVACPowerLevel=1.0, HVAC strongly pushes toward setpoint.
	// At HVACPowerLevel=0.0, HVAC is off — temp drifts with environment.
	hvacEffect := (b.SetpointTemperature - b.IndoorTemperature) * act.HVACPowerLevel * 0.15

	// Outdoor infiltration: building slowly equilibrates with outside
	infiltration := (b.OutdoorTemperature - b.IndoorTemperature) * 0.03

	// Thermal storage discharge provides supplemental conditioning toward setpoint
	storageEffect := 0.0
	if act.ThermalChargeRate < 0 {
		storageEffect = (b.SetpointTemperature - b.IndoorTemperature) * math.Abs(act.ThermalChargeRate) * 0.05
	}

	// Process equipment waste heat (always warms the building)
	processHeat := b.ProcessDemand * 0.002 // kW→°C rough factor

	deltaT := hvacEffect + infiltration + storageEffect + processHeat
	b.IndoorTemperature += deltaT

	// Clamp to physically reasonable indoor range
	b.IndoorTemperature = math.Max(10.0, math.Min(40.0, b.IndoorTemperature))

	// ----- Energy & cost accounting -----
	batchPowerDraw := e.batchRunningPower(b)
	totalKW := hvacPower + math.Max(0, chargeKW) + batchPowerDraw - shedKW
	totalKW = math.Max(0, totalKW)
	energyKWh := totalKW * StepDurationHrs
	stepCost := energyKWh * b.CurrentPrice
	stepCarbon := energyKWh * b.CarbonIntensity

	b.CumulativeCost += stepCost
	b.CumulativeCarbon += stepCarbon

	// Baseline (always-on at 70% HVAC, no storage/shedding)
	baselineKW := 0.7*b.MaxHVACPower + b.ProcessDemand
	baselineEnergy := baselineKW * StepDurationHrs
	b.BaselineCost += baselineEnergy * b.CurrentPrice
	b.BaselineCarbon += baselineEnergy * b.CarbonIntensity

	// ----- Reward computation -----
	// Get active faults for fault mitigation reward
	var activeFaults []FaultEvent
	if e.FaultSchedule != nil {
		activeFaults = e.FaultSchedule.ActiveAt(s)
	}
	rc := ComputeReward(ComputeRewardInput{
		B:               b,
		Act:             act,
		StepCost:        stepCost,
		EnergyKWh:       energyKWh,
		TMin:            TMinDefault,
		TMax:            TMaxDefault,
		StepCarbon:      stepCarbon,
		BatchMissed:     len(batchMissed),
		GridStress:      b.GridStressSignal,
		ShedFraction:    clampedShed,
		TaskID:          e.taskID,
		PrevHVACLevel:   b.PrevHVACLevel,
		ChargeRate:      act.ThermalChargeRate,
		PrevChargeRate:  e.prevChargeRates[idx],
		StorageDelta:    act.ThermalChargeRate,
		PriceCurve:      e.PriceCurve[:],
		CurrentStep:     s,
		InstructionCard: e.InstructionCard,
		ActiveFaults:    activeFaults,
	})
	b.PrevHVACLevel = act.HVACPowerLevel
	e.prevChargeRates[idx] = act.ThermalChargeRate

	// Update batch queue
	b.BatchQueue = pendingDeadlines(b.Jobs)

	// Exploit detection
	if clampedShed > 0.4 {
		e.totalShedSteps[idx]++
	}
	if len(e.thermalCycleCounts) > idx {
		if len(e.Replay) > 0 {
			prev := e.prevChargeRates[idx]
			if prev > 0.3 && act.ThermalChargeRate < -0.3 || prev < -0.3 && act.ThermalChargeRate > 0.3 {
				e.thermalCycleCounts[idx]++
			}
		}
	}

	// Per-building step index matches global timestep for this transition (0 .. EpisodeSteps-1)
	b.Step = s

	// Record history
	if idx < len(e.TempHistory) {
		e.TempHistory[idx] = append(e.TempHistory[idx], b.IndoorTemperature)
		e.CostHistory[idx] = append(e.CostHistory[idx], b.CumulativeCost)
		e.HVACHistory[idx] = append(e.HVACHistory[idx], act.HVACPowerLevel)
		e.LoadShedHistory[idx] = append(e.LoadShedHistory[idx], clampedShed)
		e.RewardHistory[idx] = append(e.RewardHistory[idx], rc)
	}

	obs := e.buildObservation(b)

	return StepResponse{
		Observation: obs,
		Reward:      rc.Total,
		Done:        e.done || s+1 >= EpisodeSteps,
		Info: StepInfo{
			RewardComponents: rc,
			EnergyUsed:       energyKWh,
			CarbonEmitted:    stepCarbon,
			PriceSignal:      b.CurrentPrice,
			GridStress:       b.GridStressSignal,
			BatchCompleted:   batchCompleted,
			BatchMissed:      batchMissed,
			Episode:          e.episode,
			Step:             s,
		},
		Rewards: rc,
	}
}

func (e *Environment) updateGridStress(s int) float64 {
	// Grid stress is elevated during price peaks and stochastic demand spikes
	price := e.PriceCurve[s]
	priceNorm := (price - 0.028) / (0.36 - 0.028)

	// Random stress events
	stressProb := 0.05
	switch e.difficulty {
	case "medium":
		stressProb = 0.1
	case "hard":
		stressProb = 0.2
	}
	spike := 0.0
	if e.rng.Float64() < stressProb {
		spike = 0.3 + e.rng.Float64()*0.5
	}
	stress := math.Min(1.0, priceNorm*0.6+spike)
	return math.Max(0, stress)
}

func (e *Environment) updateOutdoorTemp(s int) float64 {
	// Sinusoidal daily temperature cycle + noise
	hour := float64(s) / 4.0
	baseTemp := 15.0 + 8.0*math.Sin(2*math.Pi*(hour-6)/24.0)
	noise := (e.rng.Float64()*2 - 1) * 1.5
	return baseTemp + noise
}

func (e *Environment) updateProcessDemand(s int) float64 {
	// Process demand shifts with business hours
	hour := s / 4
	base := 10.0
	if hour >= 8 && hour <= 18 {
		base = 20.0 + 10.0*math.Sin(math.Pi*float64(hour-8)/10.0)
	}
	noise := (e.rng.Float64()*2 - 1) * 3.0
	return math.Max(0, base+noise)
}

func (e *Environment) updateBatchJobs(b *BuildingState, slot int, step int) (completed []int, missed []int) {
	completed = []int{}
	missed = []int{}

	// Schedule the first pending job into the chosen slot
	for i := range b.Jobs {
		job := &b.Jobs[i]
		if !job.Scheduled && !job.Completed && !job.MissedDeadline {
			schedAt := step + slot
			job.Scheduled = true
			job.ScheduledAt = schedAt
			break // only schedule one job per step
		}
	}

	// Advance running or completed jobs
	for i := range b.Jobs {
		job := &b.Jobs[i]
		if job.Completed || job.MissedDeadline {
			continue
		}
		// Check deadline miss
		if step >= job.DeadlineSlot && !job.Completed {
			job.MissedDeadline = true
			missed = append(missed, job.ID)
			continue
		}
		// Mark as completed if scheduled and past its start
		if job.Scheduled && step >= job.ScheduledAt {
			if step >= job.ScheduledAt+job.Duration-1 {
				job.Completed = true
				completed = append(completed, job.ID)
			}
		}
	}
	return
}

func (e *Environment) batchRunningPower(b *BuildingState) float64 {
	total := 0.0
	for _, job := range b.Jobs {
		if job.Scheduled && !job.Completed && !job.MissedDeadline {
			if e.step >= job.ScheduledAt && e.step < job.ScheduledAt+job.Duration {
				total += job.PowerDraw
			}
		}
	}
	return total
}

func (e *Environment) buildObservation(b *BuildingState) ObservationModel {
	// Collect active fault descriptions for this step
	var activeFaults []string
	if e.FaultSchedule != nil {
		for _, f := range e.FaultSchedule.ActiveAt(b.Step) {
			activeFaults = append(activeFaults, f.Description)
		}
	}

	// Apply sensor fault noise to observation (not physics) - if sensor fault is active, agent sees wrong temp
	reportedTemp := b.IndoorTemperature + b.TempObservationNoise

	taskCardStr := ""
	if e.taskID == 4 && e.InstructionCard != nil {
		taskCardStr = e.InstructionCard.Text
	} else if e.taskID == 1 {
		taskCardStr = "Task 1 (Easy - Cost Minimization): Minimize total energy cost over 24 hours. No temperature or batch constraints. Use cheap off-peak periods and thermal storage."
	} else if e.taskID == 2 {
		taskCardStr = "Task 2 (Medium - Temperature Management): Minimize cost AND keep indoor temperature within 19-23°C at all times. Balance comfort vs cost."
	} else if e.taskID == 3 {
		taskCardStr = "Task 3 (Hard - Full Demand Response): Minimize cost, maintain temperature, respond to grid stress (shed when grid_stress_signal > 0.7), schedule batch jobs, minimize carbon."
	} else {
		taskCardStr = "Maintain operations and minimize cost."
	}

	priceForecast := make([]float64, 4)
	for i := 0; i < 4; i++ {
		idx := b.Step + i
		if idx < EpisodeSteps {
			priceForecast[i] = math.Round(e.PriceCurve[idx]*10000) / 10000
		} else {
			priceForecast[i] = math.Round(e.PriceCurve[EpisodeSteps-1]*10000) / 10000
		}
	}

	return ObservationModel{
		IndoorTemperature:   math.Round(reportedTemp*100) / 100,
		ThermalStorageLevel: math.Round(b.ThermalStorageLevel*1000) / 1000,
		ProcessDemand:       math.Round(b.ProcessDemand*100) / 100,
		CurrentPrice:        math.Round(b.CurrentPrice*10000) / 10000,
		GridStressSignal:    math.Round(b.GridStressSignal*1000) / 1000,
		CarbonIntensity:     math.Round(b.CarbonIntensity*10) / 10,
		HourOfDay:           b.HourOfDay,
		BatchQueue:          pendingDeadlines(b.Jobs),
		CumulativeCost:      math.Round(b.CumulativeCost*10000) / 10000,
		Step:                b.Step,
		BuildingID:          b.BuildingID,
		HVACEfficiency:      math.Round(b.HVACEfficiency*1000) / 1000,
		InstructionCard:     e.InstructionCard,
		ActiveFaults:        activeFaults,
		TaskCard:            taskCardStr,
		NLSummary:           "GridMind simulation state.",
		MarketType:          "tou",
		Season:              "summer",
		PriceVolatility:     0.2,
		PriceForecast:       priceForecast,
		DemandChargeActive:  false,
	}
}

func (e *Environment) clampAction(a *ActionModel) {
	a.HVACPowerLevel = math.Max(0, math.Min(1.0, a.HVACPowerLevel))
	a.ThermalChargeRate = math.Max(-1.0, math.Min(1.0, a.ThermalChargeRate))
	a.BatchJobSlot = max(0, min(4, a.BatchJobSlot))
	a.LoadShedFraction = math.Max(0, math.Min(0.5, a.LoadShedFraction))
}

func (e *Environment) findAction(actions []ActionModel, buildingIdx int) ActionModel {
	// Try to find an action with matching building_id, else use positional
	for _, a := range actions {
		if a.BuildingID == buildingIdx {
			return a
		}
	}
	if buildingIdx < len(actions) {
		return actions[buildingIdx]
	}
	// Default: do-nothing action
	return ActionModel{HVACPowerLevel: 0.5, ThermalChargeRate: 0.0, BatchJobSlot: 0, LoadShedFraction: 0.0}
}

// pendingDeadlines returns a slice of deadline slots for all incomplete, unscheduled jobs.
func pendingDeadlines(jobs []BatchJob) []int {
	result := []int{}
	for _, j := range jobs {
		if !j.Completed && !j.MissedDeadline {
			result = append(result, j.DeadlineSlot)
		}
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ExploitDetected returns whether the current episode shows signs of degenerate strategies.
func (e *Environment) ExploitDetected(buildingIdx int) (bool, float64) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if buildingIdx >= len(e.totalShedSteps) {
		return false, 0.0
	}
	// Flag if agent always sheds > 40% load (more than 70% of steps)
	shedRatio := float64(e.totalShedSteps[buildingIdx]) / float64(e.step+1)
	cycleRatio := float64(e.thermalCycleCounts[buildingIdx]) / float64(e.step+1)
	exploited := shedRatio > 0.7 || cycleRatio > 0.4
	penalty := 0.0
	if exploited {
		penalty = math.Max(shedRatio-0.7, 0)*0.5 + math.Max(cycleRatio-0.4, 0)*0.3
	}
	return exploited, penalty
}

// GetFeederState returns the aggregate fleet view for the coordinator.
func (e *Environment) GetFeederState() FeederState {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var totalDemand float64
	buildings := make([]BuildingSummary, len(e.Buildings))
	for i, b := range e.Buildings {
		demand := b.ProcessDemand + b.MaxHVACPower*b.PrevHVACLevel
		totalDemand += demand
		buildings[i] = BuildingSummary{
			BuildingID:          b.BuildingID,
			CurrentDemandKW:     math.Round(demand*100) / 100,
			IndoorTemperature:   math.Round(b.IndoorTemperature*100) / 100,
			ThermalStorageLevel: math.Round(b.ThermalStorageLevel*1000) / 1000,
			CumulativeCost:      math.Round(b.CumulativeCost*100) / 100,
			GridStressSignal:    math.Round(b.GridStressSignal*100) / 100,
			PriceMultiplier:     e.PriceMultipliers[i],
		}
	}

	limit := float64(120 * len(e.Buildings)) // Simplistic soft cap

	// Downsample price curve to 24 hourly points
	hourlyCurve := make([]float64, 24)
	for h := 0; h < 24; h++ {
		hourlyCurve[h] = e.PriceCurve[h*4]
	}

	return FeederState{
		TotalDemandKW:    math.Round(totalDemand*100) / 100,
		FeederLimitKW:    limit,
		FeederOverload:   totalDemand > limit,
		UtilizationPct:   math.Round((totalDemand/limit)*1000) / 10,
		Buildings:        buildings,
		PriceCurveHourly: hourlyCurve,
		Step:             e.step,
		Episode:          e.episode,
	}
}

// SetCoordinatorSignals applies per-building price multipliers.
func (e *Environment) SetCoordinatorSignals(multipliers []float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for i, val := range multipliers {
		if i < len(e.PriceMultipliers) {
			e.PriceMultipliers[i] = math.Max(0.1, math.Min(10.0, val)) // Clamp safety
		}
	}
}

// cloneBuilding creates a deep copy of a BuildingState
func cloneBuilding(b *BuildingState) *BuildingState {
	c := *b
	c.BatchQueue = make([]int, len(b.BatchQueue))
	copy(c.BatchQueue, b.BatchQueue)
	c.Jobs = make([]BatchJob, len(b.Jobs))
	copy(c.Jobs, b.Jobs)
	return &c
}

// SimulateStep predicts the next state and reward without modifying the actual environment.
// It performs a deep copy of the required state, applies the actions, and returns the expected result.
func (e *Environment) SimulateStep(actions []ActionModel) ([]StepResponse, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.done {
		return nil, true
	}

	// Create a temporary mock environment for a single step
	mock := &Environment{
		rng:              rand.New(rand.NewSource(e.rng.Int63())), // local PRNG to not desync main
		episode:          e.episode,
		step:             e.step,
		taskID:           e.taskID,
		seed:             e.seed,
		difficulty:       e.difficulty,
		numBuildings:     e.numBuildings,
		Buildings:        make([]*BuildingState, e.numBuildings),
		PriceCurve:       e.PriceCurve,
		CarbonCurve:      e.CarbonCurve,
		InstructionCard:  e.InstructionCard,
		FaultSchedule:    e.FaultSchedule,
		PriceMultipliers: e.PriceMultipliers,
		prevChargeRates:  make([]float64, len(e.prevChargeRates)),
	}
	copy(mock.prevChargeRates, e.prevChargeRates)

	for i, b := range e.Buildings {
		mock.Buildings[i] = cloneBuilding(b)
	}

	// Clamp and apply actions
	mockActions := make([]ActionModel, len(actions))
	copy(mockActions, actions)
	for i := range mockActions {
		mock.clampAction(&mockActions[i])
	}

	responses := make([]StepResponse, mock.numBuildings)
	for i, b := range mock.Buildings {
		act := mock.findAction(mockActions, i)
		responses[i] = mock.stepBuilding(b, act, i)
	}

	mockDone := (mock.step + 1) >= EpisodeSteps
	return responses, mockDone
}
