/**
 * GridMind-RL Dashboard — Premium Chart.js real-time visualization
 * Polls /api/state every 500ms and updates all charts + KPIs.
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

// ── Chart.js Premium Theme ──────────────────────────────────────────────────
Chart.defaults.color = '#5a6478';
Chart.defaults.borderColor = 'rgba(255,255,255,0.03)';
Chart.defaults.font.family = "'Inter', -apple-system, system-ui, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.font.weight = 400;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation.duration = 350;
Chart.defaults.animation.easing = 'easeOutQuart';
Chart.defaults.elements.line.borderCapStyle = 'round';
Chart.defaults.elements.line.borderJoinStyle = 'round';

const C = {
  blue:    '#5b9cf6',
  green:   '#4ade80',
  amber:   '#f5a623',
  red:     '#f06e6e',
  purple:  '#a78bfa',
  cyan:    '#34d4e4',
  orange:  '#fb923c',
  teal:    '#2dd4bf',
  rose:    '#fb7185',
  grid:    'rgba(255,255,255,0.025)',
  surface: '#1a1f2e',
};

function rgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Chart factory — refined styling ──────────────────────────────────────────
function makeLineChart(id, labels, datasets, opts = {}) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      layout: { padding: { top: 4, right: 4, bottom: 0, left: 4 } },
      scales: {
        x: {
          grid: { color: C.grid, drawBorder: false },
          ticks: {
            maxTicksLimit: 8,
            font: { size: 10, family: "'JetBrains Mono', monospace" },
            color: '#3d4558',
            padding: 4,
          },
          border: { display: false },
        },
        y: {
          grid: { color: C.grid, drawBorder: false },
          ticks: {
            font: { size: 10, family: "'JetBrains Mono', monospace" },
            color: '#3d4558',
            padding: 8,
          },
          border: { display: false },
          ...opts.yAxis,
        },
      },
      plugins: {
        legend: {
          display: opts.legend || false,
          position: 'bottom',
          labels: {
            usePointStyle: true,
            pointStyle: 'circle',
            padding: 16,
            font: { size: 10, weight: 500 },
            color: '#8a94a8',
          },
        },
        tooltip: {
          backgroundColor: '#12151e',
          titleColor: '#e8ecf4',
          bodyColor: '#8a94a8',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 10,
          displayColors: true,
          boxPadding: 4,
          titleFont: { size: 11, weight: 600 },
          bodyFont: { size: 11 },
        },
      },
      ...opts.extra,
    },
  });
}

function makeBarChart(id, labels, datasets) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 4, right: 4, bottom: 0, left: 4 } },
      scales: {
        x: {
          stacked: true,
          grid: { color: C.grid, drawBorder: false },
          ticks: { maxTicksLimit: 8, font: { size: 10, family: "'JetBrains Mono', monospace" }, color: '#3d4558' },
          border: { display: false },
        },
        y: {
          stacked: true,
          grid: { color: C.grid, drawBorder: false },
          ticks: { font: { size: 10, family: "'JetBrains Mono', monospace" }, color: '#3d4558' },
          border: { display: false },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            usePointStyle: true,
            pointStyle: 'circle',
            padding: 16,
            font: { size: 10, weight: 500 },
            color: '#8a94a8',
          },
        },
        tooltip: {
          backgroundColor: '#12151e',
          titleColor: '#e8ecf4',
          bodyColor: '#8a94a8',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 10,
        },
      },
    },
  });
}

// ── Gradient helper ──────────────────────────────────────────────────────────
function createGradient(ctx, hex, startAlpha, endAlpha) {
  const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.clientHeight);
  gradient.addColorStop(0, rgba(hex, startAlpha));
  gradient.addColorStop(1, rgba(hex, endAlpha));
  return gradient;
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
      borderColor: C.amber,
      backgroundColor: rgba(C.amber, 0.08),
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'Current',
      data: [...emptyData],
      borderColor: C.red,
      backgroundColor: 'transparent',
      borderWidth: 0,
      pointRadius: 5,
      pointBackgroundColor: C.red,
      pointBorderColor: rgba(C.red, 0.3),
      pointBorderWidth: 6,
    },
  ],
  { legend: true, yAxis: { title: { display: true, text: '$/kWh', color: '#3d4558', font: { size: 10 } } } }
);

// 2. Temperature
const tempChart = makeLineChart('chart-temp',
  [],
  [
    {
      label: 'Indoor Temp (°C)',
      data: [],
      borderColor: C.cyan,
      backgroundColor: rgba(C.cyan, 0.06),
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'T_max (23°C)',
      data: [],
      borderColor: rgba(C.red, 0.35),
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
      fill: false,
    },
    {
      label: 'T_min (19°C)',
      data: [],
      borderColor: rgba(C.blue, 0.35),
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
      fill: false,
    },
  ],
  { legend: true, yAxis: { suggestedMin: 15, suggestedMax: 30, title: { display: true, text: '°C', color: '#3d4558', font: { size: 10 } } } }
);

// 3. Storage history
const storageChart = makeLineChart('chart-storage',
  [],
  [{
    label: 'Storage Level',
    data: [],
    borderColor: C.teal,
    backgroundColor: rgba(C.teal, 0.1),
    borderWidth: 2,
    fill: true,
    tension: 0.4,
    pointRadius: 0,
  }],
  { yAxis: { min: 0, max: 1 } }
);

// 4. HVAC + Load Shed
const hvacChart = makeBarChart('chart-hvac',
  [],
  [
    {
      label: 'HVAC Power',
      data: [],
      backgroundColor: rgba(C.blue, 0.6),
      borderColor: C.blue,
      borderWidth: 1,
      borderRadius: 3,
    },
    {
      label: 'Load Shed',
      data: [],
      backgroundColor: rgba(C.red, 0.6),
      borderColor: C.red,
      borderWidth: 1,
      borderRadius: 3,
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
      borderColor: C.green,
      backgroundColor: rgba(C.green, 0.06),
      borderWidth: 2,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    },
    {
      label: 'Baseline ($)',
      data: [],
      borderColor: rgba(C.amber, 0.6),
      borderDash: [5, 3],
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 0,
    },
  ],
  { legend: true, yAxis: { title: { display: true, text: '$', color: '#3d4558', font: { size: 10 } } } }
);

// 6. Grid stress history
const stressChart = makeLineChart('chart-stress',
  [],
  [{
    label: 'Grid Stress',
    data: [],
    borderColor: C.red,
    backgroundColor: rgba(C.red, 0.1),
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
    borderColor: C.orange,
    backgroundColor: rgba(C.orange, 0.08),
    borderWidth: 2,
    fill: true,
    tension: 0.4,
    pointRadius: 0,
  }],
  { yAxis: { title: { display: true, text: 'gCO₂/kWh', color: '#3d4558', font: { size: 10 } } } }
);

// 8. Reward timeline
const rewardChart = makeLineChart('chart-reward',
  [],
  [{
    label: 'Step Reward',
    data: [],
    borderColor: C.green,
    backgroundColor: rgba(C.green, 0.06),
    borderWidth: 2,
    fill: true,
    tension: 0.4,
    pointRadius: 0,
  }],
  { yAxis: { title: { display: true, text: 'Reward', color: '#3d4558', font: { size: 10 } } } }
);

// ── Stress meter bars ────────────────────────────────────────────────────────
function buildStressMeter() {
  const el = document.getElementById('stress-meter');
  if (!el) return;
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
      const color = stress > 0.7 ? C.red : stress > 0.4 ? C.amber : C.green;
      bar.style.background = color;
      bar.style.boxShadow = `0 0 6px ${rgba(color === C.red ? C.red : color === C.amber ? C.amber : C.green, 0.3)}`;
    } else {
      bar.style.background = 'rgba(255,255,255,0.03)';
      bar.style.boxShadow = 'none';
    }
  }
}

// ── Batch Gantt renderer ─────────────────────────────────────────────────────
function renderGantt(jobs, currentStep) {
  const wrap = document.getElementById('gantt-wrap');
  if (!wrap) return;
  if (!jobs || jobs.length === 0) {
    wrap.innerHTML = '<div class="gantt-empty">No batch jobs in this episode</div>';
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
    curMarker.style.cssText = `position:absolute;top:2px;bottom:2px;width:2px;background:rgba(91,156,246,0.5);left:${curPct}%;border-radius:1px;box-shadow:0 0 4px rgba(91,156,246,0.3)`;
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
  if (!container) return;
  const components = [
    { key: 'cost_savings',      label: 'Cost Savings',  color: C.green },
    { key: 'temp_constraint',   label: 'Temp Constr.',  color: C.cyan },
    { key: 'grid_response',     label: 'Grid DR',       color: C.blue },
    { key: 'efficiency_bonus',  label: 'Efficiency',    color: C.purple },
    { key: 'stability_penalty', label: 'Stability',     color: C.amber },
    { key: 'deadline_penalty',  label: 'Deadlines',     color: C.red },
    { key: 'carbon_reward',     label: 'Carbon',        color: C.orange },
  ];
  container.innerHTML = '';
  components.forEach(c => {
    const val = rc[c.key] || 0;
    const absVal = Math.abs(val);
    const pct = Math.min(100, absVal * 30);
    container.innerHTML += `
      <div class="reward-row">
        <div class="reward-label">${c.label}</div>
        <div class="reward-bar-track">
          <div class="reward-bar" style="width:${pct}%;background:${c.color};opacity:0.7"></div>
        </div>
        <div class="reward-val" style="color:${val >= 0 ? C.green : C.red}">${val.toFixed(3)}</div>
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
    document.getElementById('status-label').textContent = 'Live';

    const b = state.buildings && state.buildings[currentBuilding];
    if (!b) return;

    const step = state.step;

    // ── Header ──
    document.getElementById('ep-step').textContent = `ep:${state.episode} step:${step}/${EPISODE_STEPS}`;
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
      const markerIdx = Math.min(Math.floor(step / 4), CURVE_POINTS - 1);
      marker[markerIdx] = state.price_curve_episode[markerIdx];
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
    const stressBig = document.getElementById('stress-big');
    stressBig.textContent = b.grid_stress_signal.toFixed(3);
    stressBig.className = 'stress-value ' + 
      (b.grid_stress_signal > 0.7 ? 'high' : b.grid_stress_signal > 0.4 ? 'mid' : 'low');
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

    // ── History-based charts (only when step changes) ──
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
        storageChart.data.datasets[0].data = Array.from({ length: b.hvac_history.length }, () =>
          b.thermal_storage_level
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
        const baselineStep = b.baseline_cost / Math.max(step, 1);
        costChart.data.datasets[1].data = b.cost_history.map((_, i) => baselineStep * (i + 1));
        costChart.update('none');
      }

      // Grid stress + reward history
      if (b.reward_history && b.reward_history.length > 0) {
        const n = b.reward_history.length;
        stressChart.data.labels = Array.from({ length: n }, (_, i) => i);
        stressChart.data.datasets[0].data = b.reward_history.map(r => Math.max(0, r.grid_response || 0));
        stressChart.update('none');

        rewardChart.data.labels = Array.from({ length: n }, (_, i) => i);
        rewardChart.data.datasets[0].data = b.reward_history.map(r => r.total || 0);
        rewardChart.update('none');

        renderRewardRows(b.reward_history[b.reward_history.length - 1]);
      }

      // Batch Gantt
      renderGantt(b.jobs || [], step);
    }

  } catch (err) {
    connected = false;
    document.getElementById('conn-banner').classList.add('show');
    document.getElementById('status-dot').style.background = 'var(--accent-red)';
    document.getElementById('status-label').textContent = 'Offline';
  }
}

// ── Episode controls ─────────────────────────────────────────────────────────

async function doReset() {
  const taskId = parseInt(document.getElementById('task-select').value, 10);
  const btn = document.getElementById('btn-reset');
  btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="spin"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg> Resetting...';
  btn.disabled = true;
  btn.style.opacity = '0.6';
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
  btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg> New Episode';
  btn.disabled = false;
  btn.style.opacity = '1';
  document.getElementById('grade-result').textContent = '';
  document.getElementById('grade-result').classList.remove('show');
}

let liveSimTimer = null;
let isLiveSimulating = false;
let lastLiveState = null;

// ── Smart Heuristic Agent ───────────────────────────────────────────────────
// Mirrors the Python _heuristic_action() from inference.py.
// Reads the latest fetched state and generates intelligent actions that
// exercise ALL reward components: cost, temperature, grid DR, efficiency,
// stability, deadlines, and carbon.
function heuristicAction(b) {
  if (!b) {
    return { hvac_power_level: 0.5, thermal_charge_rate: 0.0, batch_job_slot: 0, load_shed_fraction: 0.0, building_id: currentBuilding };
  }

  const price   = b.current_price   || 0.10;
  const stress  = b.grid_stress_signal || 0.0;
  const temp    = b.indoor_temperature || 21.0;
  const storage = b.thermal_storage_level || 0.5;
  const queue   = b.batch_queue || [];
  const carbon  = b.carbon_intensity || 300;
  const step    = b.step || 0;

  // ── HVAC: price-aware + temperature-reactive ──
  let hvac = 0.5;
  if (price < 0.07)       hvac = 0.7;   // cheap → run more
  else if (price > 0.15)  hvac = 0.3;   // expensive → reduce
  else                    hvac = 0.5;

  // Temperature override: keep within 19–23°C
  if (temp > 23.0)      hvac = Math.max(hvac, 0.8);
  else if (temp > 22.0) hvac = Math.max(hvac, 0.6);
  else if (temp < 19.0) hvac = Math.min(hvac, 0.2);
  else if (temp < 20.0) hvac = Math.min(hvac, 0.35);

  // ── Thermal storage: arbitrage ──
  let charge = 0.0;
  if (price < 0.07 && storage < 0.8) {
    charge = 0.6;          // charge during cheap periods
  } else if (price > 0.15 && storage > 0.3) {
    charge = -0.5;         // discharge during expensive periods
  } else if (price < 0.10 && storage < 0.5) {
    charge = 0.3;          // moderate charge at mid-low prices
  } else if (price > 0.12 && storage > 0.6) {
    charge = -0.3;         // moderate discharge at mid-high prices
  }

  // Carbon-aware: prefer charging when carbon is low
  if (carbon < 250 && storage < 0.7) {
    charge = Math.max(charge, 0.4);
  }

  // ── Load shedding: grid stress response ──
  let shed = 0.0;
  if (stress > 0.8)       shed = 0.45;
  else if (stress > 0.7)  shed = 0.35;
  else if (stress > 0.5)  shed = 0.15;
  else if (stress > 0.3)  shed = 0.05;

  // ── Batch scheduling: urgency-aware ──
  let slot = 2;  // default: moderate defer
  if (queue.length > 0) {
    const minDeadline = Math.min(...queue);
    const stepsLeft = minDeadline - step;
    if (stepsLeft < 4)       slot = 0;  // urgent: run now
    else if (stepsLeft < 8)  slot = 1;  // soon: start soon
    else if (stepsLeft < 16) slot = 2;  // moderate
    else if (price < 0.08)   slot = 0;  // cheap: might as well run now
    else                     slot = 3;  // defer
  }

  return {
    hvac_power_level: Math.max(0, Math.min(1, hvac)),
    thermal_charge_rate: Math.max(-1, Math.min(1, charge)),
    batch_job_slot: Math.max(0, Math.min(4, slot)),
    load_shed_fraction: Math.max(0, Math.min(0.5, shed)),
    building_id: currentBuilding,
  };
}

function toggleLiveSim() {
  const btn = document.getElementById('btn-live');
  if (isLiveSimulating) {
    clearInterval(liveSimTimer);
    isLiveSimulating = false;
    btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Live Simulation';
    btn.classList.remove('active');
  } else {
    isLiveSimulating = true;
    btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause Simulation';
    btn.classList.add('active');

    liveSimTimer = setInterval(async () => {
      try {
        // Fetch current state to make informed actions
        const stateRes = await fetch(`${API_BASE}/state`);
        if (stateRes.ok) {
          const state = await stateRes.json();
          lastLiveState = state.buildings && state.buildings[currentBuilding];
        }

        // Use smart heuristic agent based on current state
        const action = heuristicAction(lastLiveState);

        await fetch(`${API_BASE}/step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(action),
        });
      } catch (e) {
        console.error(e);
      }
    }, 400);
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
    el.style.background = grade.score > 0.6 ? 'rgba(74,222,128,0.08)' : grade.score > 0.3 ? 'rgba(245,166,35,0.08)' : 'rgba(240,110,110,0.08)';
    el.classList.add('show');
  } catch (e) {
    console.error(e);
  }
}

function onTaskChange() {
  [tempChart, storageChart, hvacChart, costChart, stressChart, rewardChart].forEach(c => {
    if (!c) return;
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
  fetchAndUpdate();
  pollTimer = setInterval(fetchAndUpdate, POLL_MS);
}

startPolling();
