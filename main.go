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
	mux.HandleFunc("/state", s.handleState)
	mux.HandleFunc("/replay", s.handleReplay)
	mux.HandleFunc("/grade", s.handleGrade)
	mux.HandleFunc("/tasks", s.handleTasks)
	mux.HandleFunc("/metrics", s.handleMetrics)
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
	w.Header().Set("Content-Type", "text/html")
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
GET  /tasks            → list of tasks
GET  /metrics          → prometheus metrics
POST /reset {task_id}  → start new episode
POST /step {action}    → take action</pre>
<h3>📚 Links</h3>
<ul>
  <li><a href="https://github.com/shreeshantkhade/GridMind-RL">GitHub Repository</a></li>
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
	})

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(grade)
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
