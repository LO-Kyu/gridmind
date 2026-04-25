// main.go — GridMind-RL HTTP server (OpenEnv-compliant)
// Exposes: POST /step, POST /reset, GET /state, GET /health, GET /replay, GET /grade, GET /metrics
// Port: 7860 (Hugging Face Spaces compatible)
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"gridmind-rl/env"

	"github.com/gorilla/websocket"
)

// ──────────────────────────────────────────────
// Prometheus-style metrics (OpenTelemetry)
// ──────────────────────────────────────────────

type Metrics struct {
	mu               sync.Mutex
	stepCount        int64
	stepLatencySum   float64
	stepLatencyCount int64
	rewardSum        float64
	rewardCount      int64
	rewardMin        float64
	rewardMax        float64
	// Histograms
	actionBuckets map[string]int64 // hvac bucket counts
	errorCount    int64
}

var metrics = &Metrics{
	rewardMin:     math.MaxFloat64,
	rewardMax:     -math.MaxFloat64,
	actionBuckets: map[string]int64{"low": 0, "mid": 0, "high": 0},
}

func (m *Metrics) recordStep(latencyMs float64, reward float64) {
	atomic.AddInt64(&m.stepCount, 1)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stepLatencySum += latencyMs
	m.stepLatencyCount++
	m.rewardSum += reward
	m.rewardCount++
	if reward < m.rewardMin {
		m.rewardMin = reward
	}
	if reward > m.rewardMax {
		m.rewardMax = reward
	}
}

func (m *Metrics) recordAction(hvac float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case hvac < 0.33:
		m.actionBuckets["low"]++
	case hvac < 0.66:
		m.actionBuckets["mid"]++
	default:
		m.actionBuckets["high"]++
	}
}

func (m *Metrics) prometheus() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	avgLatency := 0.0
	if m.stepLatencyCount > 0 {
		avgLatency = m.stepLatencySum / float64(m.stepLatencyCount)
	}
	avgReward := 0.0
	if m.rewardCount > 0 {
		avgReward = m.rewardSum / float64(m.rewardCount)
	}
	return fmt.Sprintf(`# HELP gridmind_steps_total Total environment steps taken
# TYPE gridmind_steps_total counter
gridmind_steps_total %d

# HELP gridmind_step_latency_ms_avg Average step latency (ms)
# TYPE gridmind_step_latency_ms_avg gauge
gridmind_step_latency_ms_avg %.4f

# HELP gridmind_reward_avg Average reward per step
# TYPE gridmind_reward_avg gauge
gridmind_reward_avg %.4f

# HELP gridmind_reward_min Minimum reward seen
# TYPE gridmind_reward_min gauge
gridmind_reward_min %.4f

# HELP gridmind_reward_max Maximum reward seen
# TYPE gridmind_reward_max gauge
gridmind_reward_max %.4f

# HELP gridmind_action_hvac_bucket HVAC power level distribution
# TYPE gridmind_action_hvac_bucket counter
gridmind_action_hvac_bucket{bin="low"} %d
gridmind_action_hvac_bucket{bin="mid"} %d
gridmind_action_hvac_bucket{bin="high"} %d

# HELP gridmind_errors_total Total request errors
# TYPE gridmind_errors_total counter
gridmind_errors_total %d
`,
		atomic.LoadInt64(&m.stepCount),
		avgLatency, avgReward,
		m.rewardMin, m.rewardMax,
		m.actionBuckets["low"], m.actionBuckets["mid"], m.actionBuckets["high"],
		atomic.LoadInt64(&m.errorCount),
	)
}

// ──────────────────────────────────────────────
// Server
// ──────────────────────────────────────────────

type Server struct {
	envMgr *env.Environment
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func newServer() *Server {
	return &Server{envMgr: env.NewEnvironment()}
}

func (s *Server) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRoot)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/ping", s.handlePing)
	mux.HandleFunc("/reset", s.handleReset)
	mux.HandleFunc("/step", s.handleStep)
	mux.HandleFunc("/coordinator/reset", s.handleCoordinatorReset)
	mux.HandleFunc("/coordinator/step", s.handleCoordinatorStep)
	mux.HandleFunc("/state", s.handleState)
	mux.HandleFunc("/replay", s.handleReplay)
	mux.HandleFunc("/grade", s.handleGrade)
	mux.HandleFunc("/feeder", s.handleFeeder)
	mux.HandleFunc("/coordinate", s.handleCoordinate)
	mux.HandleFunc("/simulate", s.handleSimulate)
	mux.HandleFunc("/tasks", s.handleTasks)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/ws", s.handleWebSocket)
	mux.HandleFunc("/info", s.handleInfo)
	// Reverse proxy for dashboard (runs on port 7861 internally)
	mux.HandleFunc("/dashboard", s.handleDashboardProxy)
	mux.HandleFunc("/dashboard/", s.handleDashboardProxy)
	return mux
}

// ── / (Root) ────────────────────────────────────────────────────────────────

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	html := `<!DOCTYPE html>
<html>
<head><title>GridMind-RL</title>
<style>
  body { font-family: monospace; background: #0d1117; color: #e6edf3; padding: 40px; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .badge { background: #238636; padding: 4px 10px; border-radius: 4px; color: white; display: inline-block; }
  ul { line-height: 1.8; }
  pre { background: #0d1117; border: 1px solid #30363d; padding: 16px; border-radius: 6px; overflow-x: auto; }
</style>
</head>
<body>
<h1>⚡ GridMind-RL</h1>
<p><span class="badge">● RUNNING</span></p>
<p>Industrial building energy management RL environment — OpenEnv compliant.</p>
<h3>🔗 Quick Links</h3>
<ul>
  <li><a href="/dashboard">→ Live Dashboard</a></li>
  <li><a href="/health">→ API Health Check</a></li>
  <li><a href="/tasks">→ Available Tasks</a></li>
  <li><a href="/metrics">→ Prometheus Metrics</a></li>
</ul>
<h3>📡 API Endpoints</h3>
<pre>GET  /health           → health check
GET  /ping             → ping pong
GET  /state            → current environment state
GET  /replay           → episode replay data
GET  /grade            → episode grade score
GET  /feeder           → aggregate fleet status (for coordinator)
POST /coordinate       → apply price multipliers (for coordinator)
POST /simulate {action}→ predict next state (world model API)
POST /reset {task_id}  → start new episode
POST /step {action}    → take action</pre>
<h3>📚 Links</h3>
<ul>
  <li><a href="https://github.com/LO-Kyu/gridmind">GitHub Repository</a></li>
  <li><a href="https://openenv.org">OpenEnv Specification</a></li>
</ul>
</body>
</html>`
	w.Write([]byte(html))
}

// ── /health ──────────────────────────────────────────────────────────────────

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "version": "1.0.0"})
}

func (s *Server) handlePing(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// ── /reset ───────────────────────────────────────────────────────────────────

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req env.ResetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Allow empty body → defaults
		req = env.ResetRequest{TaskID: 1}
	}
	if req.TaskID == 0 {
		req.TaskID = 1
	}
	resp := s.envMgr.Reset(req)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ── /step ────────────────────────────────────────────────────────────────────

func (s *Server) handleStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	// Accept both single action and array of actions
	var actions []env.ActionModel
	var single env.ActionModel

	body := make([]byte, 0, 512)
	buf := make([]byte, 512)
	for {
		n, err := r.Body.Read(buf)
		body = append(body, buf[:n]...)
		if err != nil {
			break
		}
	}

	if len(body) > 0 && body[0] == '[' {
		if err := json.Unmarshal(body, &actions); err != nil {
			atomic.AddInt64(&metrics.errorCount, 1)
			http.Error(w, "invalid action array: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		if err := json.Unmarshal(body, &single); err != nil {
			atomic.AddInt64(&metrics.errorCount, 1)
			http.Error(w, "invalid action: "+err.Error(), http.StatusBadRequest)
			return
		}
		actions = []env.ActionModel{single}
	}

	responses, done := s.envMgr.Step(actions)

	latency := float64(time.Since(start).Microseconds()) / 1000.0
	for _, resp := range responses {
		metrics.recordStep(latency, resp.Reward)
	}
	if len(actions) > 0 {
		metrics.recordAction(actions[0].HVACPowerLevel)
	}

	w.Header().Set("Content-Type", "application/json")
	if done && len(responses) == 1 {
		responses[0].Done = true
	}
	// Return single response if single building, array otherwise
	if len(responses) == 1 {
		json.NewEncoder(w).Encode(responses[0])
	} else {
		json.NewEncoder(w).Encode(responses)
	}
}

// ── /coordinator/reset ──────────────────────────────────────────────────────

func (s *Server) handleCoordinatorReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req env.ResetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Allow empty body → defaults
		req = env.ResetRequest{TaskID: 1, NumBuildings: 3}
	}
	if req.TaskID == 0 {
		req.TaskID = 1
	}
	if req.NumBuildings == 0 {
		req.NumBuildings = 3
	}
	resp := s.envMgr.Reset(req)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ── /coordinator/step ───────────────────────────────────────────────────────

func (s *Server) handleCoordinatorStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	// Accept array of actions (one per building)
	var actions []env.ActionModel

	body := make([]byte, 0, 512)
	buf := make([]byte, 512)
	for {
		n, err := r.Body.Read(buf)
		body = append(body, buf[:n]...)
		if err != nil {
			break
		}
	}

	if err := json.Unmarshal(body, &actions); err != nil {
		atomic.AddInt64(&metrics.errorCount, 1)
		http.Error(w, "invalid action array: "+err.Error(), http.StatusBadRequest)
		return
	}

	// If empty array provided, use defaults
	if len(actions) == 0 {
		actions = []env.ActionModel{{HVACPowerLevel: 0.5, BuildingID: 0}}
	}

	responses, _ := s.envMgr.Step(actions)

	latency := float64(time.Since(start).Microseconds()) / 1000.0
	for _, resp := range responses {
		metrics.recordStep(latency, resp.Reward)
	}
	if len(actions) > 0 {
		metrics.recordAction(actions[0].HVACPowerLevel)
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Always return array format for coordinator
	json.NewEncoder(w).Encode(responses)
}

// ── /state ───────────────────────────────────────────────────────────────────

func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	state := s.envMgr.GetState()
	// Add CORS for dashboard
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(state)
}

// ── /replay ──────────────────────────────────────────────────────────────────

func (s *Server) handleReplay(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	replay := s.envMgr.GetReplay()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"replay": replay,
		"steps":  len(replay),
	})
}

// ── /grade ───────────────────────────────────────────────────────────────────

func (s *Server) handleGrade(w http.ResponseWriter, r *http.Request) {
	state := s.envMgr.GetState()
	replay := s.envMgr.GetReplay()

	// Collect per-building exploit penalties
	penalties := make([]float64, len(state.Buildings))
	for i := range state.Buildings {
		_, pen := s.envMgr.ExploitDetected(i)
		penalties[i] = pen
	}

	// Build building states from public state
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

	// Reconstruct temp history from public state
	tempHistory := make([][]float64, len(state.Buildings))
	for i, pub := range state.Buildings {
		tempHistory[i] = pub.TempHistory
	}

	grade := env.GradeEpisode(env.GradeEpisodeInput{
		TaskID:           state.TaskID,
		Buildings:        buildings,
		Replay:           replay,
		TempHistory:      tempHistory,
		TMin:             env.TMinDefault,
		TMax:             env.TMaxDefault,
		ExploitPenalties: penalties,
		InstructionCard:  state.InstructionCard,
	})

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(grade)
}

// ── /feeder ──────────────────────────────────────────────────────────────────

func (s *Server) handleFeeder(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	state := s.envMgr.GetFeederState()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(state)
}

// ── /coordinate ──────────────────────────────────────────────────────────────

func (s *Server) handleCoordinate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req env.CoordinateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.envMgr.SetCoordinatorSignals(req.PriceMultipliers)
	w.WriteHeader(http.StatusOK)
}

// ── /simulate ────────────────────────────────────────────────────────────────

func (s *Server) handleSimulate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var actions []env.ActionModel
	if err := json.NewDecoder(r.Body).Decode(&actions); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	responses, done := s.envMgr.SimulateStep(actions)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"results": responses,
		"done":    done,
	})
}

// ── /tasks ───────────────────────────────────────────────────────────────────

func (s *Server) handleTasks(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(env.AllTasks())
}

// ── /metrics ─────────────────────────────────────────────────────────────────

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprint(w, metrics.prometheus())
}

// ── /dashboard (Reverse Proxy) ────────────────────────────────────────────

func (s *Server) handleDashboardProxy(w http.ResponseWriter, r *http.Request) {
	// Target URL for the dashboard service (running on localhost:7861)
	target, err := url.Parse("http://localhost:7861")
	if err != nil {
		http.Error(w, "proxy configuration error", http.StatusInternalServerError)
		return
	}

	// Create a custom director to modify the request
	director := func(req *http.Request) {
		// Strip /dashboard prefix
		path := req.URL.Path
		if strings.HasPrefix(path, "/dashboard") {
			path = strings.TrimPrefix(path, "/dashboard")
			if path == "" {
				path = "/"
			}
		}

		// Set up the new request
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		if target.Path != "" {
			req.URL.Path = target.Path + path
		} else {
			req.URL.Path = path
		}
		req.RequestURI = ""

		// Preserve original host header for dashboard API calls
		if req.Header.Get("X-Forwarded-Host") == "" {
			req.Header.Set("X-Forwarded-For", getClientIP(r))
			req.Header.Set("X-Forwarded-Proto", "https")
		}
	}

	// Use ReverseProxy with custom director
	proxy := &httputil.ReverseProxy{Director: director}
	proxy.ServeHTTP(w, r)
}

// Helper: extract client IP from request
func getClientIP(r *http.Request) string {
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}

// ── /ws (WebSocket) ───────────────────────────────────────────────────────────

type WSMessage struct {
	Type   string          `json:"type"`
	Data   json.RawMessage `json:"data,omitempty"`
	Seed   *int64          `json:"seed,omitempty"`
	TaskID int             `json:"task_id,omitempty"`
}

type WSResetMessage struct {
	Seed         *int64 `json:"seed,omitempty"`
	TaskID       int    `json:"task_id,omitempty"`
	NumBuildings int    `json:"num_buildings,omitempty"`
}

type WSStepMessage struct {
	Action json.RawMessage `json:"action"`
}

func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	for {
		// Read message from client
		_, msgBytes, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		var msg WSMessage
		if err := json.Unmarshal(msgBytes, &msg); err != nil {
			errMsg, _ := json.Marshal(map[string]string{"error": "invalid message format"})
			conn.WriteMessage(websocket.TextMessage, errMsg)
			continue
		}

		switch msg.Type {
		case "reset":
			// GenericEnvClient sends: {"type": "reset", "data": {"seed": 42}}
			// We need to handle data payload if present
			if len(msg.Data) > 0 {
				s.handleWSReset(conn, msg.Data)
			} else {
				// Fallback to top-level fields (seed, task_id)
				s.handleWSResetDirect(conn, msg.Seed, msg.TaskID)
			}
		case "step":
			// GenericEnvClient sends: {"type": "step", "data": {"action": {...}}}
			if len(msg.Data) > 0 {
				s.handleWSStep(conn, msg.Data)
			} else {
				// Fallback to top-level action
				s.handleWSStepDirect(conn, msgBytes)
			}
		case "state":
			s.handleWSState(conn)
		case "close":
			return
		default:
			errMsg, _ := json.Marshal(map[string]string{"error": "unknown message type: " + msg.Type})
			conn.WriteMessage(websocket.TextMessage, errMsg)
		}
	}
}

func (s *Server) handleWSReset(conn *websocket.Conn, data json.RawMessage) {
	// GenericEnvClient sends: {"data": {"seed": 42}}
	// Or: {"data": {"task_id": 1, "seed": 42}}
	var reqData map[string]interface{}
	if err := json.Unmarshal(data, &reqData); err != nil {
		errMsg, _ := json.Marshal(map[string]string{"error": "invalid reset data: " + err.Error()})
		conn.WriteMessage(websocket.TextMessage, errMsg)
		return
	}

	var seed *int64
	if seedVal, ok := reqData["seed"].(float64); ok {
		s := int64(seedVal)
		seed = &s
	} else if seedVal, ok := reqData["seed"].(int64); ok {
		seed = &seedVal
	} else if seedVal, ok := reqData["seed"].(int); ok {
		s := int64(seedVal)
		seed = &s
	}

	taskID := 1
	if taskIDVal, ok := reqData["task_id"].(float64); ok {
		taskID = int(taskIDVal)
	} else if taskIDVal, ok := reqData["task_id"].(int64); ok {
		taskID = int(taskIDVal)
	} else if taskIDVal, ok := reqData["task_id"].(int); ok {
		taskID = taskIDVal
	}

	numBuildings := 1
	if nbVal, ok := reqData["num_buildings"].(float64); ok {
		numBuildings = int(nbVal)
	} else if nbVal, ok := reqData["num_buildings"].(int64); ok {
		numBuildings = int(nbVal)
	} else if nbVal, ok := reqData["num_buildings"].(int); ok {
		numBuildings = nbVal
	}

	resp := s.envMgr.Reset(env.ResetRequest{
		Seed:         seed,
		TaskID:       taskID,
		NumBuildings: numBuildings,
	})

	// Build observation response
	obs := resp.Observations[0]
	respData := map[string]interface{}{
		"observation": map[string]interface{}{
			"indoor_temperature":    obs.IndoorTemperature,
			"thermal_storage_level": obs.ThermalStorageLevel,
			"process_demand":        obs.ProcessDemand,
			"current_price":         obs.CurrentPrice,
			"grid_stress_signal":    obs.GridStressSignal,
			"carbon_intensity":      obs.CarbonIntensity,
			"hour_of_day":           obs.HourOfDay,
			"batch_queue":           obs.BatchQueue,
			"cumulative_cost":       obs.CumulativeCost,
			"step":                  obs.Step,
			"building_id":           obs.BuildingID,
		},
		"reward": nil,
		"done":   false,
		"info":   map[string]interface{}{"episode": resp.Episode, "task_id": resp.TaskID},
	}

	// Wrap in "data" field for GenericEnvClient compatibility
	response := map[string]interface{}{
		"data": respData,
	}

	respBytes, _ := json.Marshal(response)
	conn.WriteMessage(websocket.TextMessage, respBytes)
}

func (s *Server) handleWSStep(conn *websocket.Conn, data json.RawMessage) {
	// GenericEnvClient sends action directly in data: {"data": {...action fields...}}
	var reqData map[string]interface{}
	if err := json.Unmarshal(data, &reqData); err != nil {
		errMsg, _ := json.Marshal(map[string]string{"error": "invalid step data: " + err.Error()})
		conn.WriteMessage(websocket.TextMessage, errMsg)
		return
	}

	// Handle two formats:
	// 1. Direct action: {"data": {"hvac_power_level": 0.5, ...}}
	// 2. Wrapped action: {"data": {"action": {"hvac_power_level": 0.5, ...}}}
	var actionBytes []byte
	if actionData, ok := reqData["action"]; ok {
		// Wrapped format
		actionBytes, _ = json.Marshal(actionData)
	} else {
		// Direct format - use the whole reqData as action
		actionBytes = data
	}

	var action env.ActionModel
	if err := json.Unmarshal(actionBytes, &action); err != nil {
		errMsg, _ := json.Marshal(map[string]string{"error": "invalid action: " + err.Error()})
		conn.WriteMessage(websocket.TextMessage, errMsg)
		return
	}

	responses, done := s.envMgr.Step([]env.ActionModel{action})

	// Record metrics
	if len(responses) > 0 {
		metrics.recordStep(0, responses[0].Reward)
		metrics.recordAction(action.HVACPowerLevel)
	}

	obs := responses[0]
	respData := map[string]interface{}{
		"observation": map[string]interface{}{
			"indoor_temperature":    obs.Observation.IndoorTemperature,
			"thermal_storage_level": obs.Observation.ThermalStorageLevel,
			"process_demand":        obs.Observation.ProcessDemand,
			"current_price":         obs.Observation.CurrentPrice,
			"grid_stress_signal":    obs.Observation.GridStressSignal,
			"carbon_intensity":      obs.Observation.CarbonIntensity,
			"hour_of_day":           obs.Observation.HourOfDay,
			"batch_queue":           obs.Observation.BatchQueue,
			"cumulative_cost":       obs.Observation.CumulativeCost,
			"step":                  obs.Observation.Step,
			"building_id":           obs.Observation.BuildingID,
		},
		"reward": obs.Reward,
		"done":   done,
		"info":   obs.Info,
	}
	response := map[string]interface{}{"data": respData}

	respBytes, _ := json.Marshal(response)
	conn.WriteMessage(websocket.TextMessage, respBytes)
}

func (s *Server) handleWSState(conn *websocket.Conn) {
	state := s.envMgr.GetState()
	stateBytes, _ := json.Marshal(state)
	conn.WriteMessage(websocket.TextMessage, stateBytes)
}

// Direct handlers for OpenEnv client format (action at top level)

func (s *Server) handleWSResetDirect(conn *websocket.Conn, seed *int64, taskID int) {
	if seed == nil {
		var s int64 = 42
		seed = &s
	}
	if taskID == 0 {
		taskID = 1
	}

	resp := s.envMgr.Reset(env.ResetRequest{
		Seed:         seed,
		TaskID:       taskID,
		NumBuildings: 1,
	})

	obs := resp.Observations[0]
	respData := map[string]interface{}{
		"observation": map[string]interface{}{
			"indoor_temperature":    obs.IndoorTemperature,
			"thermal_storage_level": obs.ThermalStorageLevel,
			"process_demand":        obs.ProcessDemand,
			"current_price":         obs.CurrentPrice,
			"grid_stress_signal":    obs.GridStressSignal,
			"carbon_intensity":      obs.CarbonIntensity,
			"hour_of_day":           obs.HourOfDay,
			"batch_queue":           obs.BatchQueue,
			"cumulative_cost":       obs.CumulativeCost,
			"step":                  obs.Step,
			"building_id":           obs.BuildingID,
		},
		"reward": nil,
		"done":   false,
		"info":   map[string]interface{}{"episode": resp.Episode, "task_id": resp.TaskID},
	}
	response := map[string]interface{}{"data": respData}

	respBytes, _ := json.Marshal(response)
	conn.WriteMessage(websocket.TextMessage, respBytes)
}

func (s *Server) handleWSStepDirect(conn *websocket.Conn, msgBytes []byte) {
	// Parse the original message to get action directly
	var rawMsg map[string]interface{}
	if err := json.Unmarshal(msgBytes, &rawMsg); err != nil {
		errMsg, _ := json.Marshal(map[string]string{"error": "invalid step message: " + err.Error()})
		conn.WriteMessage(websocket.TextMessage, errMsg)
		return
	}

	actionData, ok := rawMsg["action"]
	if !ok {
		errMsg, _ := json.Marshal(map[string]string{"error": "missing action field"})
		conn.WriteMessage(websocket.TextMessage, errMsg)
		return
	}

	actionBytes, err := json.Marshal(actionData)
	if err != nil {
		errMsg, _ := json.Marshal(map[string]string{"error": "invalid action format"})
		conn.WriteMessage(websocket.TextMessage, errMsg)
		return
	}

	var action env.ActionModel
	if err := json.Unmarshal(actionBytes, &action); err != nil {
		errMsg, _ := json.Marshal(map[string]string{"error": "invalid action: " + err.Error()})
		conn.WriteMessage(websocket.TextMessage, errMsg)
		return
	}

	responses, done := s.envMgr.Step([]env.ActionModel{action})

	if len(responses) > 0 {
		metrics.recordStep(0, responses[0].Reward)
		metrics.recordAction(action.HVACPowerLevel)
	}

	obs := responses[0]
	respData := map[string]interface{}{
		"observation": map[string]interface{}{
			"indoor_temperature":    obs.Observation.IndoorTemperature,
			"thermal_storage_level": obs.Observation.ThermalStorageLevel,
			"process_demand":        obs.Observation.ProcessDemand,
			"current_price":         obs.Observation.CurrentPrice,
			"grid_stress_signal":    obs.Observation.GridStressSignal,
			"carbon_intensity":      obs.Observation.CarbonIntensity,
			"hour_of_day":           obs.Observation.HourOfDay,
			"batch_queue":           obs.Observation.BatchQueue,
			"cumulative_cost":       obs.Observation.CumulativeCost,
			"step":                  obs.Observation.Step,
			"building_id":           obs.Observation.BuildingID,
		},
		"reward": obs.Reward,
		"done":   done,
		"info":   obs.Info,
	}
	response := map[string]interface{}{"data": respData}

	respBytes, _ := json.Marshal(response)
	conn.WriteMessage(websocket.TextMessage, respBytes)
}

// ──────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "7860"
	}
	// Validate port
	if _, err := strconv.Atoi(port); err != nil {
		log.Fatalf("invalid PORT: %s", port)
	}

	srv := newServer()

	// Perform initial reset so /state is always valid
	var seed int64 = 42
	srv.envMgr.Reset(env.ResetRequest{Seed: &seed, TaskID: 1, NumBuildings: 1})

	log.Printf("GridMind-RL environment server starting on :%s", port)
	log.Printf("Endpoints: GET / (landing) | GET /health /ping /state /replay /grade /tasks /metrics /dashboard | POST /reset /step")

	mux := withCORS(withLogging(srv.routes()))
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// ──────────────────────────────────────────────
// Middleware
// ──────────────────────────────────────────────

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// handleInfo returns OpenEnv-standard metadata for automated validators and judges.
func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	info := map[string]interface{}{
		"name":        "gridmind-rl",
		"version":     "2.0.0",
		"description": "Multi-building industrial energy management RL environment with instruction-following, world modeling, fault injection, and curriculum learning.",
		"multi_agent": true,
		"themes": []string{
			"multi-agent",
			"long-horizon-planning",
			"world-modeling",
			"self-improvement",
		},
		"observation_space": map[string]interface{}{
			"type": "dict",
			"fields": []string{
				"indoor_temperature", "thermal_storage_level", "current_price",
				"grid_stress_signal", "carbon_intensity", "hour_of_day", "step",
				"hvac_efficiency", "process_demand", "cumulative_cost",
				"batch_queue", "active_faults", "instruction_card",
			},
		},
		"action_space": map[string]interface{}{
			"type": "dict",
			"fields": map[string]string{
				"hvac_power_level":    "float [0.0, 1.0]",
				"thermal_charge_rate": "float [-1.0, 1.0]",
				"batch_job_slot":      "int [0, 4]",
				"load_shed_fraction":  "float [0.0, 0.5]",
				"building_id":         "int [0, N_buildings-1]",
			},
		},
		"endpoints": []string{
			"POST /reset", "POST /step", "GET /grade", "GET /tasks",
			"GET /state", "POST /simulate", "GET /feeder", "POST /coordinate",
			"GET /health", "GET /info",
		},
		"hf_space": "https://lo-kyu-gridmind.hf.space",
		"github":   "https://github.com/LO-Kyu/gridmind",
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(info)
}
