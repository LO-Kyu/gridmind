/**
 * GridMind-RL Dashboard — Chart.js real-time visualization
 * Polls /api/state every 500ms and updates all charts.
 */

'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
const POLL_MS        = 500;
const EPISODE_STEPS  = 96;    // 24h × 4 steps/h (15-min)
const HISTORY_LEN    = EPISODE_STEPS;
const CURVE_POINTS   = 24;    // hourly downsample (EpisodeSteps/4)
const API_BASE       = '/api';
const TASK_NAMES = {
  1: 'Task 1 — Cost Minimization (Easy)',
  2: 'Task 2 — Temperature Management (Medium)',
  3: 'Task 3 — Full Demand Response (Hard)',
};

let currentBuilding = 0;
let pollTimer = null;
let connected = false;

// ── Chart.js global defaults ─────────────────────────────────────────────────
Chart.defaults.color = '#8899b4';
Chart.defaults.borderColor = 'rgba(56,139,253,0.1)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation.duration = 300;

const COLORS = {
  blue:   '#388bfd',
  green:  '#3fb950',
  amber:  '#d29922',
  red:    '#f85149',
  purple: '#bc8cff',
  cyan:   '#39d0d8',
  orange: '#ff7c39',
  dimBlue: 'rgba(56,139,253,0.15)',
};

function rgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Chart factory helpers ────────────────────────────────────────────────────
function makeLineChart(id, labels, datasets, opts = {}) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: 'rgba(56,139,253,0.06)' }, ticks: { maxTicksLimit: 8 } },
        y: { grid: { color: 'rgba(56,139,253,0.06)' }, ...opts.yAxis },
      },
      plugins: {
        legend: { display: opts.legend || false },
        tooltip: { backgroundColor: '#0f1829', borderColor: 'rgba(56,139,253,0.3)', borderWidth: 1 },
      },
      ...opts.extra,
    },
  });
}

function makeAreaChart(id, labels, datasets) {
  return makeLineChart(id, labels, datasets, {
    extra: { fill: true },
  });
}

function makeBarChart(id, labels, datasets) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: 'rgba(56,139,253,0.06)' }, ticks: { maxTicksLimit: 8 } },
        y: { stacked: true, grid: { color: 'rgba(56,139,253,0.06)' } },
      },
      plugins: {
        legend: { display: true, position: 'bottom', labels: { usePointStyle: true, padding: 10 } },
        tooltip: { backgroundColor: '#0f1829', borderColor: 'rgba(56,139,253,0.3)', borderWidth: 1 },
      },
    },
  });
}

// ── Initialise all charts ─────────────────────────────────────────────────────
const emptyLabels = Array.from({ length: CURVE_POINTS }, (_, i) => `${i}h`);
const emptyData   = Array(CURVE_POINTS).fill(null);

// 1. Price curve
const priceChart = makeLineChart('chart-price',
  emptyLabels,
  [
    {
      label: 'Price ($/kWh)',
      data: [...emptyData],
      borderColor: COLORS.amber,
      backgroundColor: rgba(COLORS.amber, 0.15),
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'Current',
      data: [...emptyData],
      borderColor: COLORS.red,
      backgroundColor: 'transparent',
      borderWidth: 0,
      pointRadius: 6,
      pointBackgroundColor: COLORS.red,
    },
  ],
  { legend: true, yAxis: { title: { display: true, text: '$/kWh' } } }
);

// 2. Temperature
const tempChart = makeLineChart('chart-temp',
  [],
  [
    {
      label: 'Indoor Temp (°C)',
      data: [],
      borderColor: COLORS.cyan,
      backgroundColor: rgba(COLORS.cyan, 0.1),
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'T_max (23°C)',
      data: [],
      borderColor: rgba(COLORS.red, 0.5),
      borderWidth: 1,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
    },
    {
      label: 'T_min (19°C)',
      data: [],
      borderColor: rgba(COLORS.blue, 0.5),
      borderWidth: 1,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
    },
  ],
  { legend: true, yAxis: { suggestedMin: 15, suggestedMax: 30, title: { display: true, text: '°C' } } }
);

// 3. Storage history (mini)
const storageChart = makeLineChart('chart-storage',
  [],
  [{
    label: 'Storage Level',
    data: [],
    borderColor: COLORS.cyan,
    backgroundColor: rgba(COLORS.cyan, 0.2),
    borderWidth: 2,
    fill: true,
    tension: 0.4,
    pointRadius: 0,
  }],
  { yAxis: { min: 0, max: 1 } }
);

// 4. HVAC + Load Shed stacked area
const hvacChart = makeBarChart('chart-hvac',
  [],
  [
    {
      label: 'HVAC Power',
      data: [],
      backgroundColor: rgba(COLORS.blue, 0.7),
      borderColor: COLORS.blue,
      borderWidth: 1,
    },
    {
      label: 'Load Shed',
      data: [],
      backgroundColor: rgba(COLORS.red, 0.7),
      borderColor: COLORS.red,
      borderWidth: 1,
    },
  ]
);

// 5. Cumulative cost vs baseline
const costChart = makeLineChart('chart-cost',
  [],
  [
    {
      label: 'Agent Cost ($)',
      data: [],
      borderColor: COLORS.green,
      backgroundColor: rgba(COLORS.green, 0.1),
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'Baseline ($)',
      data: [],
      borderColor: rgba(COLORS.amber, 0.7),
      borderDash: [6, 3],
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 0,
    },
  ],
  { legend: true, yAxis: { title: { display: true, text: '$' } } }
);

// 6. Grid stress history (mini)
const stressChart = makeLineChart('chart-stress',
  [],
  [{
    label: 'Grid Stress',
    data: [],
    borderColor: COLORS.red,
    backgroundColor: rgba(COLORS.red, 0.2),
    borderWidth: 2,
    fill: true,
    tension: 0.4,
    pointRadius: 0,
  }],
  { yAxis: { min: 0, max: 1 } }
);

// 7. Carbon curve
const carbonChart = makeLineChart('chart-carbon',
  emptyLabels,
  [{
    label: 'Carbon Intensity (gCO₂/kWh)',
    data: [...emptyData],
    borderColor: COLORS.orange,
    backgroundColor: rgba(COLORS.orange, 0.15),
    borderWidth: 2,
    fill: true,
    tension: 0.4,
    pointRadius: 0,
  }],
  { yAxis: { title: { display: true, text: 'gCO₂/kWh' } } }
);

// 8. Reward timeline curve
const rewardChart = makeLineChart('chart-reward',
  [],
  [
    { label: 'Step Reward',   data: [], borderColor: COLORS.green, backgroundColor: rgba(COLORS.green, 0.1), borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 },
  ],
  { yAxis: { title: { display: true, text: 'Reward' } } }
);

// ── Stress meter bars ────────────────────────────────────────────────────────
function buildStressMeter() {
  const el = document.getElementById('stress-meter');
  el.innerHTML = '';
  for (let i = 0; i < 20; i++) {
    const bar = document.createElement('div');
    bar.className = 'stress-bar';
    bar.id = `sm-${i}`;
    el.appendChild(bar);
  }
}
buildStressMeter();

function updateStressMeter(stress) {
  const bars = 20;
  const active = Math.round(stress * bars);
  for (let i = 0; i < bars; i++) {
    const bar = document.getElementById(`sm-${i}`);
    if (!bar) continue;
    const pct = (i / bars) * 100;
    bar.style.height = `${20 + pct * 0.8}%`;
    if (i < active) {
      const color = stress > 0.7 ? COLORS.red : stress > 0.4 ? COLORS.amber : COLORS.green;
      bar.style.background = color;
      bar.style.opacity = '1';
    } else {
      bar.style.background = 'rgba(255,255,255,0.05)';
      bar.style.opacity = '1';
    }
  }
}

// ── Batch Gantt renderer ─────────────────────────────────────────────────────
function renderGantt(jobs, currentStep) {
  const wrap = document.getElementById('gantt-wrap');
  if (!jobs || jobs.length === 0) {
    wrap.innerHTML = '<div style="color:var(--text-dim);font-size:0.8rem">No batch jobs in this episode.</div>';
    return;
  }
  const totalSlots = EPISODE_STEPS;
  wrap.innerHTML = '';
  jobs.forEach(job => {
    const row = document.createElement('div');
    row.className = 'gantt-row';

    const label = document.createElement('div');
    label.className = 'gantt-label';
    label.textContent = `J${job.id}`;
    row.appendChild(label);

    const track = document.createElement('div');
    track.className = 'gantt-track';

    // Deadline marker
    const deadlinePct = (job.deadline_slot / totalSlots) * 100;
    const deadline = document.createElement('div');
    deadline.className = 'gantt-deadline';
    deadline.style.left = `${deadlinePct}%`;
    deadline.title = `Deadline: step ${job.deadline_slot}`;
    track.appendChild(deadline);

    // Job block
    if (job.scheduled) {
      const startPct = (job.scheduled_at / totalSlots) * 100;
      const widthPct = (job.duration / totalSlots) * 100;
      const block = document.createElement('div');
      block.className = 'gantt-block ' + (job.completed ? 'completed' : job.missed_deadline ? 'missed' : 'scheduled');
      block.style.left = `${startPct}%`;
      block.style.width = `${Math.max(widthPct, 1)}%`;
      track.appendChild(block);
    }

    // Current step marker
    const curPct = (currentStep / totalSlots) * 100;
    const curMarker = document.createElement('div');
    curMarker.style.cssText = `position:absolute;top:0;bottom:0;width:1px;background:rgba(56,139,253,0.6);left:${curPct}%`;
    track.appendChild(curMarker);

    row.appendChild(track);

    // Status badge
    const statusWrap = document.createElement('div');
    statusWrap.className = 'gantt-status';
    let badgeClass = 'pending', badgeText = 'pending';
    if (job.completed)       { badgeClass = 'ok';      badgeText = 'done'; }
    else if (job.missed_deadline) { badgeClass = 'missed';  badgeText = 'missed'; }
    else if (job.scheduled && !job.completed) { badgeClass = 'running'; badgeText = 'running'; }
    statusWrap.innerHTML = `<span class="badge ${badgeClass}">${badgeText}</span>`;
    row.appendChild(statusWrap);

    wrap.appendChild(row);
  });
}

// ── Reward breakdown rows ─────────────────────────────────────────────────────
function renderRewardRows(rc) {
  if (!rc) return;
  const container = document.getElementById('reward-rows');
  const components = [
    { key: 'cost_savings',      label: 'Cost Savings',  color: COLORS.green,  sign: 1 },
    { key: 'temp_constraint',   label: 'Temp Constr.',  color: COLORS.cyan,   sign: 1 },
    { key: 'grid_response',     label: 'Grid DR',       color: COLORS.blue,   sign: 1 },
    { key: 'efficiency_bonus',  label: 'Efficiency',    color: COLORS.purple, sign: 1 },
    { key: 'stability_penalty', label: 'Stability',     color: COLORS.amber,  sign: -1 },
    { key: 'deadline_penalty',  label: 'Deadlines',     color: COLORS.red,    sign: -1 },
    { key: 'carbon_reward',     label: 'Carbon',        color: COLORS.orange, sign: 1 },
  ];
  container.innerHTML = '';
  components.forEach(c => {
    const val = rc[c.key] || 0;
    const absVal = Math.abs(val);
    const pct = Math.min(100, absVal * 30); // scale 0–~3 reward to 0–100%
    container.innerHTML += `
      <div class="reward-row">
        <div class="reward-label">${c.label}</div>
        <div class="reward-bar-wrap">
          <div class="reward-bar" style="width:${pct}%;background:${c.color};opacity:0.8"></div>
        </div>
        <div class="reward-val" style="color:${val >= 0 ? COLORS.green : COLORS.red}">${val.toFixed(3)}</div>
      </div>`;
  });
}

// ── KPI color logic ──────────────────────────────────────────────────────────
function colorClass(val, good, bad) {
  if (val <= good) return 'good';
  if (val >= bad)  return 'bad';
  return 'warn';
}

// ── Main state update ─────────────────────────────────────────────────────────
let lastStep = -1;

async function fetchAndUpdate() {
  try {
    const res = await fetch(`${API_BASE}/state`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const state = await res.json();
    connected = true;
    document.getElementById('conn-banner').classList.remove('show');
    document.getElementById('status-dot').style.background = 'var(--accent-green)';

    const b = state.buildings && state.buildings[currentBuilding];
    if (!b) return;

    const step = state.step;
    const hourOfDay = b.hour_of_day || 0;

    // ── Header ──
    document.getElementById('ep-step').textContent = `ep:${state.episode} step:${step}/${EPISODE_STEPS - 1}`;
    document.getElementById('task-badge').textContent = TASK_NAMES[state.task_id] || 'Task 1';

    // ── KPIs ──
    const priceEl = document.getElementById('kpi-price');
    priceEl.textContent = `$${b.current_price.toFixed(4)}`;
    priceEl.className = 'kpi-value ' + colorClass(b.current_price, 0.08, 0.16);

    const tempEl = document.getElementById('kpi-temp');
    tempEl.textContent = `${b.indoor_temperature.toFixed(1)}°C`;
    const inBounds = b.indoor_temperature >= 19 && b.indoor_temperature <= 23;
    tempEl.className = 'kpi-value ' + (inBounds ? 'good' : 'bad');

    const stressEl = document.getElementById('kpi-stress');
    stressEl.textContent = b.grid_stress_signal.toFixed(3);
    stressEl.className = 'kpi-value ' + colorClass(b.grid_stress_signal, 0.4, 0.7);

    const costEl = document.getElementById('kpi-cost');
    const savings = b.baseline_cost - b.cumulative_cost;
    costEl.textContent = `$${b.cumulative_cost.toFixed(2)}`;
    costEl.className = 'kpi-value ' + (savings > 0 ? 'good' : 'warn');
    document.getElementById('kpi-baseline').textContent = `$${b.baseline_cost.toFixed(2)}`;

    document.getElementById('kpi-carbon').textContent = `${b.carbon_intensity.toFixed(0)}`;
    document.getElementById('kpi-demand').textContent = `${b.process_demand.toFixed(1)}`;
    document.getElementById('kpi-storage').textContent = `${(b.thermal_storage_level * 100).toFixed(1)}`;

    // ── Price curve chart ──
    if (state.price_curve_episode && state.price_curve_episode.length === CURVE_POINTS) {
      const labels = Array.from({ length: CURVE_POINTS }, (_, i) => `${i}:00`);
      priceChart.data.labels = labels;
      priceChart.data.datasets[0].data = state.price_curve_episode;
      const marker = Array(CURVE_POINTS).fill(null);
      marker[Math.floor(step / 4)] = state.price_curve_episode[Math.floor(step / 4)];
      priceChart.data.datasets[1].data = marker;
      priceChart.update('none');
    }

    // ── Carbon curve ──
    if (state.carbon_curve_episode && state.carbon_curve_episode.length === CURVE_POINTS) {
      carbonChart.data.labels = Array.from({ length: CURVE_POINTS }, (_, i) => `${i}:00`);
      carbonChart.data.datasets[0].data = state.carbon_curve_episode;
      carbonChart.update('none');
    }

    // ── Grid stress ──
    document.getElementById('stress-big').textContent = b.grid_stress_signal.toFixed(3);
    updateStressMeter(b.grid_stress_signal);
    const cardStress = document.getElementById('card-stress');
    if (b.grid_stress_signal > 0.7) {
      cardStress.classList.add('alert-active');
    } else {
      cardStress.classList.remove('alert-active');
    }

    // ── Thermal storage bar ──
    const storagePct = (b.thermal_storage_level * 100).toFixed(1);
    document.getElementById('storage-pct').textContent = storagePct;
    document.getElementById('storage-fill').style.width = `${storagePct}%`;

    // ── History-based charts (only update when step changes) ──
    if (step !== lastStep) {
      lastStep = step;
      const stepLabels = Array.from({ length: b.temp_history.length }, (_, i) => i);

      // Temperature chart
      if (b.temp_history.length > 0) {
        tempChart.data.labels = stepLabels;
        tempChart.data.datasets[0].data = b.temp_history;
        tempChart.data.datasets[1].data = b.temp_history.map(() => 23);
        tempChart.data.datasets[2].data = b.temp_history.map(() => 19);
        tempChart.update('none');
      }

      // Storage history
      if (b.hvac_history && b.hvac_history.length > 0) {
        storageChart.data.labels = stepLabels;
        storageChart.data.datasets[0].data = Array.from({ length: b.hvac_history.length }, (_, i) =>
          b.thermal_storage_level // simplify: use current level as placeholder
        );
        storageChart.update('none');
      }

      // HVAC + load shed (bar)
      if (b.hvac_history && b.load_shed_history) {
        const n = Math.min(b.hvac_history.length, HISTORY_LEN);
        hvacChart.data.labels = Array.from({ length: n }, (_, i) => i);
        hvacChart.data.datasets[0].data = b.hvac_history.slice(0, n);
        hvacChart.data.datasets[1].data = b.load_shed_history.slice(0, n);
        hvacChart.update('none');
      }

      // Cost vs baseline
      if (b.cost_history && b.cost_history.length > 0) {
        const n = b.cost_history.length;
        costChart.data.labels = Array.from({ length: n }, (_, i) => i);
        costChart.data.datasets[0].data = b.cost_history;
        // Generate approximate baseline curve (linear ramp to b.baseline_cost)
        const baselineStep = b.baseline_cost / Math.max(step, 1);
        costChart.data.datasets[1].data = b.cost_history.map((_, i) => baselineStep * (i + 1));
        costChart.update('none');
      }

      // Grid stress history
      if (b.reward_history && b.reward_history.length > 0) {
        const n = b.reward_history.length;
        stressChart.data.labels = Array.from({ length: n }, (_, i) => i);
        stressChart.data.datasets[0].data = b.reward_history.map(r => Math.max(0, r.grid_response || 0));
        stressChart.update('none');

        // Total reward timeline chart (full episode)
        rewardChart.data.labels = Array.from({ length: n }, (_, i) => i);
        rewardChart.data.datasets[0].data = b.reward_history.map(r => r.total || 0);
        rewardChart.update('none');

        // Reward rows (last step)
        renderRewardRows(b.reward_history[b.reward_history.length - 1]);
      }

      // Batch Gantt
      renderGantt(b.jobs || [], step);
    }

  } catch (err) {
    connected = false;
    document.getElementById('conn-banner').classList.add('show');
    document.getElementById('status-dot').style.background = 'var(--accent-red)';
    // console.error('Poll error:', err);
  }
}

// ── Episode controls ─────────────────────────────────────────────────────────

async function doReset() {
  const taskId = parseInt(document.getElementById('task-select').value, 10);
  const btn = document.getElementById('btn-reset');
  btn.textContent = 'Resetting...';
  btn.disabled = true;
  lastStep = -1;
  try {
    await fetch(`${API_BASE}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: taskId, num_buildings: 1 }),
    });
  } catch (e) {
    console.error(e);
  }
  btn.textContent = '↺ New Episode';
  btn.disabled = false;
  document.getElementById('grade-result').textContent = '';
}

let liveSimTimer = null;
let isLiveSimulating = false;

function toggleLiveSim() {
  const btn = document.getElementById('btn-live');
  if (isLiveSimulating) {
    // Stop live sim
    clearInterval(liveSimTimer);
    isLiveSimulating = false;
    btn.textContent = '▶ Start Live Simulation';
    btn.style.background = 'var(--accent-green)';
  } else {
    // Start live sim
    isLiveSimulating = true;
    btn.textContent = '⏸ Pause Live Simulation';
    btn.style.background = 'var(--accent-amber)';
    
    liveSimTimer = setInterval(async () => {
      // Step the environment automatically with a simple heuristic policy
      const taskId = parseInt(document.getElementById('task-select').value, 10);
      try {
        await fetch(`${API_BASE}/step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            hvac_power_level: 0.5,
            thermal_charge_rate: 0.0,
            batch_job_slot: 0,
            load_shed_fraction: 0.0,
            building_id: currentBuilding
          }),
        });
        // fetchAndUpdate() will catch the change via polling
      } catch (e) {
        console.error(e);
      }
    }, 400); // 400ms per step
  }
}

async function doGrade() {
  try {
    const res = await fetch(`${API_BASE}/grade`);
    const grade = await res.json();
    const score = (grade.score * 100).toFixed(2);
    const el = document.getElementById('grade-result');
    el.textContent = `Score: ${score}% ${grade.exploit_detected ? '⚠ exploit!' : ''}`;
    el.style.color = grade.score > 0.6 ? 'var(--accent-green)' : grade.score > 0.3 ? 'var(--accent-amber)' : 'var(--accent-red)';
  } catch (e) {
    console.error(e);
  }
}

function onTaskChange() {
  // Reset chart histories on task change
  [tempChart, storageChart, hvacChart, costChart, stressChart, rewardChart].forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(d => d.data = []);
    c.update('none');
  });
}

function onBuildingChange() {
  currentBuilding = parseInt(document.getElementById('building-select').value, 10);
  lastStep = -1;
}

// ── Start polling ────────────────────────────────────────────────────────────
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  fetchAndUpdate(); // immediate first fetch
  pollTimer = setInterval(fetchAndUpdate, POLL_MS);
}

startPolling();
