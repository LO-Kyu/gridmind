// Package env defines the three GridMind-RL tasks and their deterministic graders.
package env

import "math"

// TaskConfig describes a single task.
type TaskConfig struct {
	ID          int    `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Difficulty  string `json:"difficulty"`
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
			Weights:     map[string]float64{"cost": 0.35, "temperature": 0.25, "grid_response": 0.25, "batch_deadline": 0.15},
		},
	}
}

// GradeEpisodeInput collects all data needed to score a completed episode.
type GradeEpisodeInput struct {
	TaskID       int
	Buildings    []*BuildingState
	Replay       []ReplayEntry
	TempHistory  [][]float64 // per building, per step
	TMin         float64
	TMax         float64
	ExploitPenalties []float64
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

	grade.Score = math.Round(grade.Score*10000) / 10000 // 4 decimal places
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

	grade.SubScores["cost"] = math.Min(1.0, costScore)
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

	grade.SubScores["cost"] = costScore
	grade.SubScores["temperature"] = constraintScore
	grade.Score = costScore*0.6 + constraintScore*0.4
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

	grade.SubScores["cost"] = costScore
	grade.SubScores["temperature"] = tempScore
	grade.SubScores["grid_response"] = gridScore
	grade.SubScores["batch_deadline"] = batchScore

	// Weighted composite score
	grade.Score = costScore*0.35 + tempScore*0.25 + gridScore*0.25 + batchScore*0.15

	grade.Details["grid_stress_steps"] = gridStressSteps
	grade.Details["grid_response_steps"] = gridResponseSteps
	grade.Details["total_jobs"] = totalJobs
	grade.Details["completed_on_time"] = completedOnTime
	return grade
}
