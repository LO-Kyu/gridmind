// Package env implements the multi-component dense reward function for GridMind-RL.
package env

import "math"

// ComputeRewardInput bundles all inputs needed to compute the reward for one step.
type ComputeRewardInput struct {
	B               *BuildingState
	Act             ActionModel
	StepCost        float64   // $ cost incurred this step
	EnergyKWh       float64   // kWh consumed this step
	TMin            float64   // lower temperature bound (°C)
	TMax            float64   // upper temperature bound (°C)
	StepCarbon      float64   // gCO2 emitted this step
	BatchMissed     int       // number of batch jobs that missed deadline this step
	GridStress      float64   // 0.0–1.0 grid stress signal
	ShedFraction    float64   // clamped load shed fraction
	TaskID          int       // 1, 2, or 3
	PrevHVACLevel   float64   // previous step's HVAC power level (for stability)
	ChargeRate      float64   // current thermal charge rate
	PrevChargeRate  float64   // previous step's thermal charge rate
	StorageDelta    float64   // change in storage level (+ = charging)
	PriceCurve      []float64 // full episode price curve for arbitrage calc
	CurrentStep     int       // current step index
}

// ComputeReward returns a dense RewardComponents struct from the current step inputs.
// All 7 reward components are always computed for rich per-step signal.
// Task-specific weighting is handled by the GRADING system (tasks.go), not here.
func ComputeReward(inp ComputeRewardInput) RewardComponents {
	rc := RewardComponents{}

	// ── 1. Cost Savings ─────────────────────────────────────────────────────
	// Positive baseline minus relative cost: smart agents save money.
	typicalCost := 4.0
	rc.CostSavings = 1.5 - (inp.StepCost/typicalCost)*2.0

	// ── 2. Temperature Constraint ────────────────────────────────────────────
	// Gaussian bonus for being near setpoint; penalty outside comfort bounds.
	temp := inp.B.IndoorTemperature
	rc.TempConstraint = computeTempReward(temp, inp.B.SetpointTemperature, inp.TMin, inp.TMax)

	// ── 3. Grid Stress Response ──────────────────────────────────────────────
	// Rewards proactive grid awareness and demand-response compliance.
	rc.GridResponse = computeGridResponse(inp.GridStress, inp.ShedFraction)

	// ── 4. Deadline Penalty / Bonus ──────────────────────────────────────────
	// Penalise missed batch jobs, reward on-track pending jobs.
	if inp.BatchMissed > 0 {
		rc.DeadlinePenalty = -float64(inp.BatchMissed) * 1.5
	}
	// Positive signal: reward for jobs still on track (not missed yet)
	onTrackJobs := 0
	for _, job := range inp.B.Jobs {
		if !job.Completed && !job.MissedDeadline {
			onTrackJobs++
		}
		if job.Completed && !job.MissedDeadline {
			onTrackJobs++ // completed on time is even better
		}
	}
	if onTrackJobs > 0 && inp.BatchMissed == 0 {
		rc.DeadlinePenalty += float64(onTrackJobs) * 0.08
	}

	// ── 5. Efficiency Bonus (thermal storage utilization) ─────────────────────
	// Rewards smart storage use: arbitrage + maintaining useful storage levels.
	if len(inp.PriceCurve) > inp.CurrentStep {
		rc.EfficiencyBonus = computeArbitrageBonus(
			inp.ChargeRate,
			inp.PriceCurve[inp.CurrentStep],
			inp.PriceCurve,
			inp.CurrentStep,
		)
	}
	// Baseline: reward maintaining a balanced storage level (not empty, not always full)
	storageLevel := inp.B.ThermalStorageLevel
	if storageLevel > 0.2 && storageLevel < 0.85 {
		rc.EfficiencyBonus += 0.15 // good operating range
	} else if storageLevel <= 0.05 || storageLevel >= 0.98 {
		rc.EfficiencyBonus -= 0.1 // extremes are wasteful
	}

	// ── 6. Stability Reward/Penalty ──────────────────────────────────────────
	// Smooth operation earns a bonus; rapid oscillation earns a penalty.
	hvacDelta := math.Abs(inp.Act.HVACPowerLevel - inp.PrevHVACLevel)
	chargeDelta := math.Abs(inp.ChargeRate - inp.PrevChargeRate)
	oscillation := hvacDelta*0.5 + chargeDelta*0.3
	if oscillation > 0.3 {
		rc.StabilityPenalty = -(oscillation - 0.3) * 0.8
	} else {
		// Positive reward for smooth, stable control
		rc.StabilityPenalty = (0.3 - oscillation) * 0.4
	}

	// ── 7. Carbon Reward ─────────────────────────────────────────────────────
	// Rewards low-carbon operation based on grid carbon intensity.
	carbonNorm := math.Max(0, (inp.B.CarbonIntensity-100.0)/600.0)
	// Baseline bonus, reduced by carbon-heavy consumption
	rc.CarbonReward = 0.6 - (inp.EnergyKWh * carbonNorm * 0.25)
	// Extra bonus for operating during genuinely clean grid periods
	if carbonNorm < 0.3 {
		rc.CarbonReward += 0.15
	}

	// ── Aggregate ────────────────────────────────────────────────────────────
	rc.Total = rc.CostSavings + rc.TempConstraint + rc.GridResponse +
		rc.DeadlinePenalty + rc.EfficiencyBonus + rc.StabilityPenalty + rc.CarbonReward

	return rc
}

// computeTempReward returns a reward based on how close the indoor temperature
// is to the setpoint, with a hard penalty outside [TMin, TMax].
func computeTempReward(temp, setpoint, tMin, tMax float64) float64 {
	if temp >= tMin && temp <= tMax {
		// Gaussian-shaped bonus: maximum at setpoint, degrades toward bounds
		deviation := math.Abs(temp - setpoint)
		sigma := (tMax - tMin) / 4.0
		return math.Exp(-0.5*(deviation/sigma)*(deviation/sigma)) * 1.5 // Increased positive reward
	}
	// Outside bounds: proportional penalty
	excess := math.Max(temp-tMax, tMin-temp)
	return -excess * 0.6
}

// computeGridResponse returns a reward for grid-aware behavior:
// bonus for shedding during stress, baseline for readiness, penalty for waste.
func computeGridResponse(stress, shedFraction float64) float64 {
	if stress > 0.7 {
		// High stress: large bonus proportional to shed fraction
		if shedFraction > 0.1 {
			return shedFraction * stress * 1.5
		}
		// High stress but not shedding: penalty
		return -0.2 * stress
	}
	if stress > 0.3 {
		// Moderate stress: small bonus for readiness, small bonus for proactive shedding
		if shedFraction > 0.05 {
			return shedFraction * 0.5 // proactive shedding during moderate stress
		}
		return 0.08 // grid-aware readiness bonus
	}
	// Low stress: mild penalty for unnecessary shedding, baseline for normal operation
	if shedFraction > 0.1 {
		return -shedFraction * 0.3
	}
	return 0.1 // small positive signal for operating normally under low stress
}

// computeArbitrageBonus rewards storage use when current price is low vs recent history
// (causal: uses only past prices, no future curve leakage).
func computeArbitrageBonus(chargeRate, currentPrice float64, curve []float64, step int) float64 {
	lookBack := 8
	pastSum := 0.0
	count := 0
	for i := step - lookBack; i < step && i >= 0; i++ {
		pastSum += curve[i]
		count++
	}
	if count == 0 {
		return 0.0
	}
	pastAvg := pastSum / float64(count)

	if chargeRate > 0 && currentPrice < pastAvg {
		return chargeRate * (pastAvg - currentPrice) * 2.0
	}
	if chargeRate < 0 && currentPrice > pastAvg {
		return math.Abs(chargeRate) * (currentPrice - pastAvg) * 2.0
	}
	return 0.0
}
