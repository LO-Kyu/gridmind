// Package env defines the four GridMind-RL tasks and their deterministic graders.
package env

import (
	"fmt"
	"math"
	"math/rand"
)

// clampOpenInterval clamps a score to the open interval (0, 1), strictly excluding 0.0 and 1.0.
// This ensures all scores satisfy the requirement: 0 < score < 1
func clampOpenInterval(score float64) float64 {
	const epsilon = 1e-6
	if score <= 0.0 {
		return epsilon
	}
	if score >= 1.0 {
		return 1.0 - epsilon
	}
	return score
}

// TaskConfig describes a single task.
type TaskConfig struct {
	ID          int                `json:"id"`
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Difficulty  string             `json:"difficulty"`
	Weights     map[string]float64 `json:"weights"`
}

// AllTasks returns the ordered list of task configurations.
func AllTasks() []TaskConfig {
	return []TaskConfig{
		{
			ID:          1,
			Name:        "Cost Minimization",
			Description: "Minimize total energy cost over a 24-hour episode with no process constraints. Beat the always-on flat policy baseline.",
			Difficulty:  "easy",
			Weights:     map[string]float64{"cost": 1.0},
		},
		{
			ID:          2,
			Name:        "Constrained Temperature Management",
			Description: "Minimize cost while keeping indoor temperature within ±2°C of setpoint at all times.",
			Difficulty:  "medium",
			Weights:     map[string]float64{"cost": 0.6, "temperature": 0.4},
		},
		{
			ID:          3,
			Name:        "Full Demand-Response with Batch Scheduling",
			Description: "Minimize cost, maintain temperature, respond to grid stress events, schedule all batch jobs before their deadlines, and minimize carbon emissions.",
			Difficulty:  "hard",
			Weights:     map[string]float64{"cost": 0.28, "temperature": 0.20, "grid_response": 0.20, "batch_deadline": 0.12, "carbon": 0.20},
		},
		{
			ID:          4,
			Name:        "Instruction-Following Operator",
			Description: "Complete a randomly sampled natural-language objective card. The agent must parse the instruction, plan accordingly, and satisfy all stated KPI targets.",
			Difficulty:  "hard",
			Weights:     map[string]float64{"task_completion": 0.50, "cost": 0.30, "temperature": 0.20},
		},
	}
}

// instructionTemplate is a parameterised instruction card template.
type instructionTemplate struct {
	makeText    func(params map[string]float64) string
	targets     map[string]float64
	weights     map[string]float64
}

// GenerateInstructionCard samples a random instruction card for Task 4.
// The card contains a human-readable text objective plus machine-readable targets.
func GenerateInstructionCard(rng *rand.Rand) *InstructionCard {
	// Pool of parameterised templates
	templates := []instructionTemplate{
		{
			// Template 1: hard energy cap
			makeText: func(p map[string]float64) string {
				return fmt.Sprintf("Keep total energy cost under $%.2f for this 24-hour episode while maintaining comfort.", p["cost_cap"])
			},
			targets:  map[string]float64{"max_cost": 0.0}, // filled in below
			weights:  map[string]float64{"task_completion": 0.5, "cost": 0.3, "temperature": 0.2},
		},
		{
			// Template 2: aggressive temperature constraint
			makeText: func(p map[string]float64) string {
				return fmt.Sprintf("Never allow indoor temperature to exceed %.0f°C or drop below %.0f°C at any point during the episode.", p["t_max"], p["t_min"])
			},
			targets:  map[string]float64{"t_min": 0.0, "t_max": 0.0},
			weights:  map[string]float64{"task_completion": 0.5, "temperature": 0.4, "cost": 0.1},
		},
		{
			// Template 3: grid response SLA
			makeText: func(p map[string]float64) string {
				return fmt.Sprintf("Respond to all grid stress events (signal > 0.7) by shedding at least %.0f%% of non-critical load.", p["min_shed_pct"]*100)
			},
			targets:  map[string]float64{"min_shed_fraction": 0.0},
			weights:  map[string]float64{"task_completion": 0.5, "cost": 0.2, "temperature": 0.3},
		},
		{
			// Template 4: carbon reduction
			makeText: func(p map[string]float64) string {
				return fmt.Sprintf("Reduce carbon emissions to at least %.0f%% below the always-on baseline policy.", p["carbon_reduction_pct"]*100)
			},
			targets:  map[string]float64{"carbon_reduction": 0.0},
			weights:  map[string]float64{"task_completion": 0.5, "cost": 0.2, "temperature": 0.2, "carbon": 0.1},
		},
		{
			// Template 5: combined cost + temperature + grid
			makeText: func(p map[string]float64) string {
				return fmt.Sprintf("Keep energy cost under $%.2f, temperature between %.0f–%.0f°C, and respond to all grid stress events.", p["cost_cap"], p["t_min"], p["t_max"])
			},
			targets:  map[string]float64{"max_cost": 0.0, "t_min": 0.0, "t_max": 0.0, "min_shed_fraction": 0.25},
			weights:  map[string]float64{"task_completion": 0.5, "cost": 0.2, "temperature": 0.2, "grid_response": 0.1},
		},
	}

	// Pick a random template
	tmpl := templates[rng.Intn(len(templates))]

	// Randomise numeric parameters
	params := map[string]float64{
		"cost_cap":             1.5 + rng.Float64()*2.0,   // $1.50 – $3.50
		"t_min":               18.0 + rng.Float64()*2.0,  // 18–20 °C
		"t_max":               23.0 + rng.Float64()*2.0,  // 23–25 °C
		"min_shed_pct":        0.2 + rng.Float64()*0.2,   // 20–40 %
		"carbon_reduction_pct": 0.15 + rng.Float64()*0.2, // 15–35 %
	}

	// Fill targets from params
	targets := make(map[string]float64)
	for k := range tmpl.targets {
		switch k {
		case "max_cost":
			targets[k] = params["cost_cap"]
		case "t_min":
			targets[k] = params["t_min"]
		case "t_max":
			targets[k] = params["t_max"]
		case "min_shed_fraction":
			targets[k] = params["min_shed_pct"]
		case "carbon_reduction":
			targets[k] = params["carbon_reduction_pct"]
		}
	}

	weights := make(map[string]float64)
	for k, v := range tmpl.weights {
		weights[k] = v
	}

	return &InstructionCard{
		Text:    tmpl.makeText(params),
		Targets: targets,
		Weights: weights,
	}
}

// GradeEpisodeInput collects all data needed to score a completed episode.
type GradeEpisodeInput struct {
	TaskID           int
	Buildings        []*BuildingState
	Replay           []ReplayEntry
	TempHistory      [][]float64 // per building, per step
	TMin             float64
	TMax             float64
	ExploitPenalties []float64
	InstructionCard  *InstructionCard // set for Task 4 episodes
}

// GradeEpisode computes a deterministic 0.0–1.0 score for a completed episode.
// Given a fixed random seed, this function is fully deterministic.
func GradeEpisode(inp GradeEpisodeInput) EpisodeGrade {
	grade := EpisodeGrade{
		TaskID:    inp.TaskID,
		SubScores: map[string]float64{},
		Details:   map[string]interface{}{},
	}

	switch inp.TaskID {
	case 1:
		grade = gradeTask1(inp, grade)
	case 2:
		grade = gradeTask2(inp, grade)
	case 3:
		grade = gradeTask3(inp, grade)
	case 4:
		grade = gradeTask4(inp, grade)
	default:
		grade = gradeTask1(inp, grade)
	}

	// Exploit detection: reduce score by penalty
	totalPenalty := 0.0
	for i, b := range inp.Buildings {
		_ = b
		if i < len(inp.ExploitPenalties) {
			totalPenalty += inp.ExploitPenalties[i]
		}
	}
	if totalPenalty > 0 {
		grade.ExploitDetected = true
		grade.PenaltyApplied = math.Min(totalPenalty, 0.3) // max 30% penalty
		grade.Score = math.Max(0, grade.Score-grade.PenaltyApplied)
	}

	// Clamp AFTER rounding to ensure boundary values are handled
	grade.Score = clampOpenInterval(math.Round(grade.Score*10000) / 10000) // 4 decimal places

	// Also ensure all sub-scores are properly clamped after rounding
	for key, val := range grade.SubScores {
		grade.SubScores[key] = clampOpenInterval(math.Round(val*10000) / 10000)
	}
	return grade
}

// ── Task 1: Cost Minimization ───────────────────────────────────────────────

func gradeTask1(inp GradeEpisodeInput, grade EpisodeGrade) EpisodeGrade {
	agentCost := 0.0
	baselineCost := 0.0
	for _, b := range inp.Buildings {
		agentCost += b.CumulativeCost
		baselineCost += b.BaselineCost
	}

	var costScore float64
	if baselineCost > 0 {
		// score = max(0, 1 - agent_cost / baseline_cost)
		// 0.0 if agent costs same or more, 1.0 if agent costs nothing
		ratio := agentCost / baselineCost
		costScore = math.Max(0, 1.0-ratio)
	}

	// Clamp after min operation
	clamped := clampOpenInterval(math.Min(1.0, costScore))
	grade.SubScores["cost"] = clampOpenInterval(math.Round(clamped*10000) / 10000)
	grade.Score = grade.SubScores["cost"]
	grade.Details["agent_cost"] = agentCost
	grade.Details["baseline_cost"] = baselineCost
	grade.Details["cost_ratio"] = agentCost / math.Max(baselineCost, 0.01)
	return grade
}

// ── Task 2: Constrained Temperature Management ──────────────────────────────

func gradeTask2(inp GradeEpisodeInput, grade EpisodeGrade) EpisodeGrade {
	// Cost sub-score (same as task 1)
	grade = gradeTask1(inp, grade)
	costScore := grade.SubScores["cost"]

	// Temperature constraint sub-score
	totalSteps := 0
	withinBounds := 0
	for i, history := range inp.TempHistory {
		_ = i
		for _, temp := range history {
			totalSteps++
			if temp >= inp.TMin && temp <= inp.TMax {
				withinBounds++
			}
		}
	}
	constraintScore := 0.0
	if totalSteps > 0 {
		constraintScore = float64(withinBounds) / float64(totalSteps)
	}

	// Clamp sub-scores and final score after rounding
	grade.SubScores["cost"] = clampOpenInterval(math.Round(costScore*10000) / 10000)
	grade.SubScores["temperature"] = clampOpenInterval(math.Round(constraintScore*10000) / 10000)
	finalScore := costScore*0.6 + constraintScore*0.4
	grade.Score = clampOpenInterval(math.Round(finalScore*10000) / 10000)
	grade.Details["within_bounds_steps"] = withinBounds
	grade.Details["total_steps"] = totalSteps
	return grade
}

// ── Task 3: Full Demand-Response with Batch Scheduling ──────────────────────

func gradeTask3(inp GradeEpisodeInput, grade EpisodeGrade) EpisodeGrade {
	// Reuse task 2 for cost + temperature scores
	grade = gradeTask2(inp, grade)
	costScore := grade.SubScores["cost"]
	tempScore := grade.SubScores["temperature"]

	// Grid response sub-score
	// Count steps where stress > 0.7 and shed_fraction > 0.15
	gridStressSteps := 0
	gridResponseSteps := 0
	for _, entry := range inp.Replay {
		if entry.Observation.GridStressSignal > 0.7 {
			gridStressSteps++
			if entry.Action.LoadShedFraction > 0.15 {
				gridResponseSteps++
			}
		}
	}
	gridScore := 0.5 // default neutral if no stress events
	if gridStressSteps > 0 {
		gridScore = float64(gridResponseSteps) / float64(gridStressSteps)
	}

	// Batch deadline sub-score
	totalJobs := 0
	completedOnTime := 0
	for _, b := range inp.Buildings {
		for _, job := range b.Jobs {
			totalJobs++
			if job.Completed && !job.MissedDeadline {
				completedOnTime++
			}
		}
	}
	batchScore := 0.0
	if totalJobs > 0 {
		batchScore = float64(completedOnTime) / float64(totalJobs)
	}

	// Carbon sub-score vs baseline always-on policy (same spirit as cost)
	agentCarbon := 0.0
	baselineCarbon := 0.0
	for _, b := range inp.Buildings {
		agentCarbon += b.CumulativeCarbon
		baselineCarbon += b.BaselineCarbon
	}
	carbonScore := 0.0
	if baselineCarbon > 0 {
		carbonScore = math.Max(0, 1.0-agentCarbon/baselineCarbon)
	}

	// Clamp all sub-scores after rounding
	grade.SubScores["cost"] = clampOpenInterval(math.Round(costScore*10000) / 10000)
	grade.SubScores["temperature"] = clampOpenInterval(math.Round(tempScore*10000) / 10000)
	grade.SubScores["grid_response"] = clampOpenInterval(math.Round(gridScore*10000) / 10000)
	grade.SubScores["batch_deadline"] = clampOpenInterval(math.Round(batchScore*10000) / 10000)
	grade.SubScores["carbon"] = clampOpenInterval(math.Round(math.Min(1.0, carbonScore)*10000) / 10000)

	finalScore := costScore*0.28 + tempScore*0.20 + gridScore*0.20 + batchScore*0.12 + carbonScore*0.20
	grade.Score = clampOpenInterval(math.Round(finalScore*10000) / 10000)

	grade.Details["grid_stress_steps"] = gridStressSteps
	grade.Details["grid_response_steps"] = gridResponseSteps
	grade.Details["total_jobs"] = totalJobs
	grade.Details["completed_on_time"] = completedOnTime
	grade.Details["agent_carbon"] = agentCarbon
	grade.Details["baseline_carbon"] = baselineCarbon
	return grade
}

// ── Task 4: Instruction-Following Operator ───────────────────────────────────

// gradeTask4 evaluates how well the agent satisfied the natural-language
// instruction card issued at reset. It reads the InstructionCard from Building 0,
// checks each target that appears in the card, and computes a weighted score.
// Falls back to Task 3 grading when no instruction card is available.
func gradeTask4(inp GradeEpisodeInput, grade EpisodeGrade) EpisodeGrade {
	// Require an instruction card — passed from the environment at grade time
	if inp.InstructionCard == nil {
		// Fallback: grade as Task 3 (no card to evaluate)
		return gradeTask3(inp, grade)
	}

	card := inp.InstructionCard
	weights := card.Weights
	targets := card.Targets

	// Always compute base sub-scores — reuse existing graders
	base := gradeTask3(inp, EpisodeGrade{
		TaskID:    inp.TaskID,
		SubScores: map[string]float64{},
		Details:   map[string]interface{}{},
	})
	costScore := base.SubScores["cost"]
	tempScore := base.SubScores["temperature"]
	gridScore := base.SubScores["grid_response"]
	carbonScore := base.SubScores["carbon"]

	// ── Card-specific KPI checks ─────────────────────────────────────────────

	// KPI 1: Cost cap — did the agent stay under max_cost?
	taskCompletionScore := 0.5 // default partial credit
	if maxCost, ok := targets["max_cost"]; ok && maxCost > 0 {
		agentCost := 0.0
		for _, b := range inp.Buildings {
			agentCost += b.CumulativeCost
		}
		if agentCost <= maxCost {
			taskCompletionScore = 1.0
		} else {
			// Partial credit: how close were they?
			taskCompletionScore = math.Max(0, 1.0-(agentCost-maxCost)/maxCost)
		}
		grade.Details["target_max_cost"] = maxCost
		grade.Details["actual_cost"] = agentCost
	}

	// KPI 2: Temperature bounds — never violated t_min / t_max
	if tMin, hasTMin := targets["t_min"]; hasTMin {
		tMax, hasTMax := targets["t_max"]
		if hasTMax {
			totalSteps := 0
			withinBounds := 0
			for _, history := range inp.TempHistory {
				for _, temp := range history {
					totalSteps++
					if temp >= tMin && temp <= tMax {
						withinBounds++
					}
				}
			}
			if totalSteps > 0 {
				adherence := float64(withinBounds) / float64(totalSteps)
				// Strict: full credit only if ALWAYS within bounds
				taskCompletionScore = adherence
			}
			grade.Details["target_t_min"] = tMin
			grade.Details["target_t_max"] = tMax
		}
	}

	// KPI 3: Grid response SLA — shed >= min_shed_fraction when stress > 0.7
	if minShed, ok := targets["min_shed_fraction"]; ok {
		stressSteps := 0
		compliantSteps := 0
		for _, entry := range inp.Replay {
			if entry.Observation.GridStressSignal > 0.7 {
				stressSteps++
				if entry.Action.LoadShedFraction >= minShed {
					compliantSteps++
				}
			}
		}
		if stressSteps > 0 {
			taskCompletionScore = float64(compliantSteps) / float64(stressSteps)
		}
		grade.Details["target_min_shed"] = minShed
		grade.Details["stress_steps"] = stressSteps
		grade.Details["compliant_steps"] = compliantSteps
	}

	// KPI 4: Carbon reduction — did agent beat baseline by carbon_reduction target?
	if carbonTarget, ok := targets["carbon_reduction"]; ok {
		agentCarbon := 0.0
		baselineCarbon := 0.0
		for _, b := range inp.Buildings {
			agentCarbon += b.CumulativeCarbon
			baselineCarbon += b.BaselineCarbon
		}
		if baselineCarbon > 0 {
			actualReduction := 1.0 - agentCarbon/baselineCarbon
			if actualReduction >= carbonTarget {
				taskCompletionScore = 1.0
			} else {
				taskCompletionScore = math.Max(0, actualReduction/carbonTarget)
			}
		}
		grade.Details["target_carbon_reduction"] = carbonTarget
	}

	// ── Weighted final score ─────────────────────────────────────────────────
	// Use weights from the card; fall back to Task 4 defaults if missing
	wTask := getWeight(weights, "task_completion", 0.50)
	wCost := getWeight(weights, "cost", 0.20)
	wTemp := getWeight(weights, "temperature", 0.20)
	wGrid := getWeight(weights, "grid_response", 0.05)
	wCarbon := getWeight(weights, "carbon", 0.05)

	finalScore := taskCompletionScore*wTask +
		costScore*wCost +
		tempScore*wTemp +
		gridScore*wGrid +
		carbonScore*wCarbon

	grade.SubScores["task_completion"] = clampOpenInterval(math.Round(taskCompletionScore*10000) / 10000)
	grade.SubScores["cost"] = clampOpenInterval(math.Round(costScore*10000) / 10000)
	grade.SubScores["temperature"] = clampOpenInterval(math.Round(tempScore*10000) / 10000)
	grade.SubScores["grid_response"] = clampOpenInterval(math.Round(gridScore*10000) / 10000)
	grade.SubScores["carbon"] = clampOpenInterval(math.Round(carbonScore*10000) / 10000)
	grade.Score = clampOpenInterval(math.Round(finalScore*10000) / 10000)

	grade.Details["instruction_card_text"] = card.Text
	return grade
}

// getWeight safely retrieves a weight from a map, returning defaultVal if missing.
func getWeight(weights map[string]float64, key string, defaultVal float64) float64 {
	if v, ok := weights[key]; ok {
		return v
	}
	return defaultVal
}
