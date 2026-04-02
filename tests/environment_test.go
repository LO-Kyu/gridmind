// tests/test_environment.go — unit tests for GridMind-RL environment
package tests

import (
	"testing"
	"math"

	"gridmind-rl/env"
)

// TestResetProducesValidObservation checks that reset returns sane initial observations.
func TestResetProducesValidObservation(t *testing.T) {
	e := env.NewEnvironment()
	var seed int64 = 42
	resp := e.Reset(env.ResetRequest{Seed: &seed, TaskID: 1, NumBuildings: 1})

	if len(resp.Observations) != 1 {
		t.Fatalf("expected 1 observation, got %d", len(resp.Observations))
	}
	obs := resp.Observations[0]

	if obs.IndoorTemperature < 10 || obs.IndoorTemperature > 40 {
		t.Errorf("indoor_temperature out of range: %.2f", obs.IndoorTemperature)
	}
	if obs.ThermalStorageLevel < 0 || obs.ThermalStorageLevel > 1 {
		t.Errorf("thermal_storage_level out of [0,1]: %.3f", obs.ThermalStorageLevel)
	}
	if obs.CurrentPrice <= 0 {
		t.Errorf("current_price must be positive, got %.4f", obs.CurrentPrice)
	}
	if obs.HourOfDay < 0 || obs.HourOfDay > 23 {
		t.Errorf("hour_of_day out of [0,23]: %d", obs.HourOfDay)
	}
	if obs.GridStressSignal < 0 || obs.GridStressSignal > 1 {
		t.Errorf("grid_stress_signal out of [0,1]: %.3f", obs.GridStressSignal)
	}
}

// TestStepAdvancesState verifies that step increments the step counter.
func TestStepAdvancesState(t *testing.T) {
	e := env.NewEnvironment()
	var seed int64 = 1
	e.Reset(env.ResetRequest{Seed: &seed, TaskID: 1, NumBuildings: 1})

	action := []env.ActionModel{{HVACPowerLevel: 0.5, ThermalChargeRate: 0.0, BatchJobSlot: 0}}
	resps, done := e.Step(action)

	if done {
		t.Error("episode should not be done after first step")
	}
	if len(resps) != 1 {
		t.Fatalf("expected 1 step response, got %d", len(resps))
	}
	state := e.GetState()
	if state.Step != 1 {
		t.Errorf("expected step=1 after one step, got %d", state.Step)
	}
	if resps[0].Observation.Step != 0 {
		t.Errorf("expected observation.step=0 after first transition, got %d", resps[0].Observation.Step)
	}
}

// TestEpisodeLengthIs96 verifies the episode terminates after 96 steps (24h).
func TestEpisodeLengthIs96(t *testing.T) {
	e := env.NewEnvironment()
	var seed int64 = 99
	e.Reset(env.ResetRequest{Seed: &seed, TaskID: 1, NumBuildings: 1})

	action := []env.ActionModel{{HVACPowerLevel: 0.5}}
	var lastDone bool
	for i := 0; i < env.EpisodeSteps; i++ {
		_, lastDone = e.Step(action)
	}
	if !lastDone {
		t.Errorf("episode should be done after %d steps", env.EpisodeSteps)
	}
}

// TestDeterministicWithSeed verifies that two runs with the same seed produce identical rewards.
func TestDeterministicWithSeed(t *testing.T) {
	action := []env.ActionModel{{HVACPowerLevel: 0.4, ThermalChargeRate: 0.1, BatchJobSlot: 1}}
	var seed int64 = 1337

	run := func() float64 {
		e := env.NewEnvironment()
		e.Reset(env.ResetRequest{Seed: &seed, TaskID: 2, NumBuildings: 1})
		resps, _ := e.Step(action)
		return resps[0].Reward
	}

	r1 := run()
	r2 := run()
	if math.Abs(r1-r2) > 1e-9 {
		t.Errorf("non-deterministic rewards with same seed: %.6f vs %.6f", r1, r2)
	}
}

// TestActionClamping verifies out-of-range actions are clamped.
func TestActionClamping(t *testing.T) {
	e := env.NewEnvironment()
	var seed int64 = 7
	e.Reset(env.ResetRequest{Seed: &seed, TaskID: 1})

	// Over-range action
	action := []env.ActionModel{{HVACPowerLevel: 2.0, ThermalChargeRate: -5.0, LoadShedFraction: 0.9}}
	resps, _ := e.Step(action)
	if len(resps) == 0 {
		t.Fatal("no responses returned")
	}
	// After step, state should still be valid
	state := e.GetState()
	if len(state.Buildings) == 0 {
		t.Fatal("no buildings in state")
	}
	b := state.Buildings[0]
	if b.ThermalStorageLevel < 0 || b.ThermalStorageLevel > 1 {
		t.Errorf("thermal storage out of bounds: %.3f", b.ThermalStorageLevel)
	}
}

// TestMultiBuildingFederation checks that 3-building reset + step works.
func TestMultiBuildingFederation(t *testing.T) {
	e := env.NewEnvironment()
	var seed int64 = 5
	resp := e.Reset(env.ResetRequest{Seed: &seed, TaskID: 3, NumBuildings: 3})

	if len(resp.Observations) != 3 {
		t.Fatalf("expected 3 observations for 3 buildings, got %d", len(resp.Observations))
	}

	actions := []env.ActionModel{
		{HVACPowerLevel: 0.3, BuildingID: 0},
		{HVACPowerLevel: 0.5, BuildingID: 1},
		{HVACPowerLevel: 0.7, BuildingID: 2},
	}
	resps, _ := e.Step(actions)
	if len(resps) != 3 {
		t.Fatalf("expected 3 step responses, got %d", len(resps))
	}
}

// TestRewardComponentsAreFinite verifies no NaN/Inf in rewards.
func TestRewardComponentsAreFinite(t *testing.T) {
	e := env.NewEnvironment()
	var seed int64 = 42
	e.Reset(env.ResetRequest{Seed: &seed, TaskID: 3})

	action := []env.ActionModel{{HVACPowerLevel: 0.5, ThermalChargeRate: 0.2, BatchJobSlot: 2, LoadShedFraction: 0.3}}
	resps, _ := e.Step(action)

	rc := resps[0].Info.RewardComponents
	vals := []float64{rc.CostSavings, rc.TempConstraint, rc.GridResponse,
		rc.DeadlinePenalty, rc.EfficiencyBonus, rc.StabilityPenalty, rc.CarbonReward, rc.Total}
	for i, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("reward component %d is not finite: %v", i, v)
		}
	}
}

// TestGraderTask1ScoreRange verifies Task 1 score is always in [0, 1].
func TestGraderTask1ScoreRange(t *testing.T) {
	e := env.NewEnvironment()
	var seed int64 = 101
	e.Reset(env.ResetRequest{Seed: &seed, TaskID: 1})

	action := []env.ActionModel{{HVACPowerLevel: 0.3}}
	for i := 0; i < env.EpisodeSteps; i++ {
		e.Step(action)
	}

	state := e.GetState()
	replay := e.GetReplay()

	buildings := make([]*env.BuildingState, len(state.Buildings))
	for i, pub := range state.Buildings {
		jobsCopy := make([]env.BatchJob, len(pub.Jobs))
		copy(jobsCopy, pub.Jobs)
		buildings[i] = &env.BuildingState{
			CumulativeCost:   pub.CumulativeCost,
			BaselineCost:     pub.BaselineCost,
			CumulativeCarbon: pub.CumulativeCarbon,
			BaselineCarbon:   pub.BaselineCarbon,
			Jobs:             jobsCopy,
		}
	}

	grade := env.GradeEpisode(env.GradeEpisodeInput{
		TaskID:    1,
		Buildings: buildings,
		Replay:    replay,
		TMin:      env.TMinDefault,
		TMax:      env.TMaxDefault,
	})

	if grade.Score < 0 || grade.Score > 1 {
		t.Errorf("Task 1 score out of [0,1]: %.4f", grade.Score)
	}
}
