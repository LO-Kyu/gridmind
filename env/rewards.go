// Package env implements the multi-component dense reward function for GridMind-RL.
package env

import "math"

// ComputeRewardInput bundles all inputs needed to compute the reward for one step.
type ComputeRewardInput struct {
	B               *BuildingState
	Act             ActionModel
	StepCost        float64          // $ cost incurred this step
	EnergyKWh       float64          // kWh consumed this step
	TMin            float64          // lower temperature bound (°C)
	TMax            float64          // upper temperature bound (°C)
	StepCarbon      float64          // gCO2 emitted this step
	BatchMissed     int              // number of batch jobs that missed deadline this step
	GridStress      float64          // 0.0–1.0 grid stress signal
	ShedFraction    float64          // clamped load shed fraction
	TaskID          int              // 1, 2, 3, or 4
	PrevHVACLevel   float64          // previous step's HVAC power level (for stability)
	ChargeRate      float64          // current thermal charge rate
	PrevChargeRate  float64          // previous step's thermal charge rate
	StorageDelta    float64          // change in storage level (+ = charging)
	PriceCurve      []float64        // full episode price curve for arbitrage calc
	CurrentStep     int              // current step index
	InstructionCard *InstructionCard // non-nil for Task 4 episodes
	ActiveFaults    []FaultEvent      // currently active fault events for Track 3
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

	// ── 8. Instruction-Following Reward (Task 4 only) ─────────────────────────
	if inp.TaskID == 4 && inp.InstructionCard != nil {
		rc.InstructionReward = computeInstructionReward(inp.InstructionCard, inp.B, inp.ShedFraction, inp.GridStress)
	}

	// ── 9. Fault Mitigation Reward (Track 3) ──────────────────────────────
	if len(inp.ActiveFaults) > 0 {
		rc.FaultMitigation = computeFaultMitigationReward(inp.B, inp.ActiveFaults)
	}

	// ── Aggregate ────────────────────────────────────────────────────────────
	// Total is the sum of all 9 reward components. Each component is computed
	// independently above and contributes directly to the total signal.
	rc.Total = rc.CostSavings + rc.TempConstraint + rc.GridResponse +
		rc.DeadlinePenalty + rc.EfficiencyBonus + rc.StabilityPenalty + rc.CarbonReward +
		rc.InstructionReward + rc.FaultMitigation

	return rc
}

// computeInstructionReward scores per-step progress against the instruction card targets.
// Returns a value in roughly [-0.5, 1.0] depending on how well the agent tracks targets.
func computeInstructionReward(card *InstructionCard, b *BuildingState, shedFraction, gridStress float64) float64 {
	if card == nil {
		return 0.0
	}

	score := 0.0
	weight := card.Weights["task_completion"]
	if weight == 0 {
		weight = 0.5
	}

	components := 0
	total := 0.0

	// KPI: energy cost cap
	if maxCost, ok := card.Targets["max_cost"]; ok && maxCost > 0 {
		components++
		if b.CumulativeCost <= maxCost {
			total += 1.0 // on track
		} else {
			// Proportional penalty for how far over budget we are
			overRatio := (b.CumulativeCost - maxCost) / maxCost
			total += math.Max(-1.0, -overRatio)
		}
	}

	// KPI: temperature bounds
	if tMin, okMin := card.Targets["t_min"]; okMin {
		if tMax, okMax := card.Targets["t_max"]; okMax {
			components++
			temp := b.IndoorTemperature
			if temp >= tMin && temp <= tMax {
				total += 1.0
			} else {
				excess := math.Max(temp-tMax, tMin-temp)
				total += math.Max(-1.0, -excess*0.3)
			}
		}
	}

	// KPI: minimum load shed during grid stress
	if minShed, ok := card.Targets["min_shed_fraction"]; ok {
		components++
		if gridStress > 0.7 {
			if shedFraction >= minShed {
				total += 1.0
			} else {
				total += (shedFraction / minShed) - 1.0 // partial credit
			}
		} else {
			total += 0.5 // no stress event this step — neutral
		}
	}

	// KPI: carbon reduction (vs baseline, approximated by carbon intensity signal)
	if _, ok := card.Targets["carbon_reduction"]; ok {
		components++
		// Proxy: reward operating when carbon intensity is low
		carbonNorm := math.Max(0, (b.CarbonIntensity-100.0)/600.0)
		if carbonNorm < 0.4 {
			total += 1.0
		} else {
			total += 1.0 - carbonNorm
		}
	}

	if components == 0 {
		return 0.0
	}
	score = (total / float64(components)) * weight
	return math.Max(-0.5, math.Min(1.0, score))
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

// computeFaultMitigationReward returns reward/penalty for proper fault response behavior.
// Tracks Track 3 (fault handling) in the hackathon theme.
func computeFaultMitigationReward(b *BuildingState, activeFaults []FaultEvent) float64 {
	if len(activeFaults) == 0 {
		return 0.0
	}

	score := 0.0
	for _, fault := range activeFaults {
		switch fault.Type {
		case FaultGridOutage:
			// Reward for shedding load during grid outage
			// High load_shed_fraction = good. Low = bad.
			if b.LoadShedFraction > 0.5 {
				score += 0.3 * b.LoadShedFraction
			} else {
				score -= 0.2
			}
		case FaultChillerFailure:
			// Reward for reducing HVAC during chiller fault
			hvacLevel := b.PrevHVACLevel
			if hvacLevel < 0.4 {
				score += 0.2
			} else {
				score -= 0.15
			}
		}
	}

	// Critical penalty: building 0 overheating during any fault
	if b.BuildingID == 0 && b.IndoorTemperature > 28.0 && len(activeFaults) > 0 {
		score -= 0.5
	}

	return math.Max(-0.5, math.Min(0.3, score))
}
