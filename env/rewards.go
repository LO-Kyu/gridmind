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
// The reward is task-aware: task 1 only cares about cost, task 2 adds temperature,
// task 3 adds grid response, batch deadlines, and carbon.
func ComputeReward(inp ComputeRewardInput) RewardComponents {
	rc := RewardComponents{}

	// ── 1. Cost Savings ─────────────────────────────────────────────────────
	// Shift from pure penalty to a positive baseline: standardizing operations gives positive reward.
	// Baseline reward of 1.5, minus the relative cost.
	typicalCost := 4.0
	rc.CostSavings = 1.5 - (inp.StepCost / typicalCost) * 2.0

	// ── 2. Temperature Constraint ────────────────────────────────────────────
	// Only active for task 2 and 3.
	if inp.TaskID >= 2 {
		temp := inp.B.IndoorTemperature
		rc.TempConstraint = computeTempReward(temp, inp.B.SetpointTemperature, inp.TMin, inp.TMax)
	}

	// ── 3. Grid Stress Response ──────────────────────────────────────────────
	// Only active for task 3.
	if inp.TaskID >= 3 {
		rc.GridResponse = computeGridResponse(inp.GridStress, inp.ShedFraction)
	}

	// ── 4. Deadline Penalty ──────────────────────────────────────────────────
	// Task 1 is cost-only; batch jobs are not part of the objective.
	if inp.BatchMissed > 0 && inp.TaskID >= 2 {
		rc.DeadlinePenalty = -float64(inp.BatchMissed) * 1.5
	}

	// ── 5. Efficiency Bonus (thermal storage arbitrage) ───────────────────────
	// Reward for charging storage during cheap periods and discharging during expensive ones.
	if len(inp.PriceCurve) > inp.CurrentStep {
		rc.EfficiencyBonus = computeArbitrageBonus(
			inp.ChargeRate,
			inp.PriceCurve[inp.CurrentStep],
			inp.PriceCurve,
			inp.CurrentStep,
		)
	}

	// ── 6. Stability Penalty ─────────────────────────────────────────────────
	// Penalise rapid oscillation in HVAC setpoint and thermal charge rate.
	hvacDelta := math.Abs(inp.Act.HVACPowerLevel - inp.PrevHVACLevel)
	chargeDelta := math.Abs(inp.ChargeRate - inp.PrevChargeRate)
	oscillation := hvacDelta*0.5 + chargeDelta*0.3
	if oscillation > 0.3 {
		rc.StabilityPenalty = -(oscillation - 0.3) * 0.8
	}

	// ── 7. Carbon Reward ─────────────────────────────────────────────────────
	// Low-carbon bonus: active for task 3.
	if inp.TaskID >= 3 {
		// Normalise carbon: iso-ne range roughly 100–700 gCO2/kWh
		carbonNorm := (inp.B.CarbonIntensity - 100.0) / 600.0
		// Provide a baseline positive score, reduced by carbon footprint
		rc.CarbonReward = 0.5 - (inp.EnergyKWh * carbonNorm * 0.3)
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

// computeGridResponse returns a bonus for shedding load during high grid stress,
// and a mild penalty for shedding when the grid is fine.
func computeGridResponse(stress, shedFraction float64) float64 {
	if stress > 0.7 {
		// Bonus proportional to shed fraction
		return shedFraction * stress * 1.5
	}
	// Mild penalty for unnecessary shedding (reduces productivity without benefit)
	return -shedFraction * (0.7 - stress) * 0.3
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
