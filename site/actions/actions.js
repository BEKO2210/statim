// Statim Plays: Replay Player & Interactive Showcase

document.documentElement.classList.add("js");

(function () {
  "use strict";

  const STRATEGIES = {
    "2048": "Keep the largest tile in a corner; prefer merges and many empty cells.",
    "snake": "Reach food safely while preserving enough open space to avoid trapping the snake.",
    "othello": "Take corners, avoid giving corners, limit opponent mobility, and gain discs.",
    "tetris": "Clear lines while keeping the stack low, even, and free of holes."
  };

  const GAMES = ["2048", "snake", "othello", "tetris"];
  const BASE_INTERVAL = 1200; // ms per step at 1x

  // Check prefers-reduced-motion
  const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  let prefersReducedMotion = reducedMotionQuery.matches;

  // State
  const state = {
    viewMode: "overview", // "overview" | "focused"
    activeGame: "2048",
    isPlaying: !prefersReducedMotion,
    speed: 1.0,
    traces: {}, // { "2048": [...], "snake": [...], ... }
    steps: { "2048": 0, "snake": 0, "othello": 0, "tetris": 0 },
    prevSteps: { "2048": -1, "snake": -1, "othello": -1, "tetris": -1 },
    timerId: null
  };

  // Helper: HiDPI Canvas setup
  function getCanvasContext(canvas) {
    if (!canvas) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || canvas.width;
    const h = rect.height || canvas.height;
    
    // Check if backing store needs resizing
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    return { ctx, width: w, height: h };
  }

  // ---------- RENDERERS ----------

  // 2048 Renderer
  const TILE_COLORS = {
    0: { bg: "#161c24", text: "#475569" },
    2: { bg: "#202a36", text: "#cbd5e1" },
    4: { bg: "#283647", text: "#e2e8f0" },
    8: { bg: "#163c32", text: "#a7f3d0" },
    16: { bg: "#154d3e", text: "#6ee7b7" },
    32: { bg: "#125f4b", text: "#34d399" },
    64: { bg: "#107257", text: "#ecfdf5", glow: "rgba(52, 211, 153, 0.25)" },
    128: { bg: "#0d8463", text: "#ffffff", glow: "rgba(52, 211, 153, 0.4)" },
    256: { bg: "#0b9670", text: "#ffffff", glow: "rgba(52, 211, 153, 0.5)" },
    512: { bg: "#0aa87d", text: "#ffffff", glow: "rgba(52, 211, 153, 0.6)" },
    1024: { bg: "#0ebf8d", text: "#04261a", glow: "rgba(52, 211, 153, 0.7)" },
    2048: { bg: "#34d399", text: "#04261a", glow: "rgba(52, 211, 153, 0.9)" }
  };

  function render2048(canvas, record) {
    const setup = getCanvasContext(canvas);
    if (!setup || !record) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);

    // Board background
    const pad = Math.max(6, Math.round(width * 0.035));
    const radius = 10;
    ctx.fillStyle = "#11161d";
    roundRect(ctx, 0, 0, width, height, radius);
    ctx.fill();

    const board = record.board || [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    const gridCols = 4;
    const cellSize = (width - pad * (gridCols + 1)) / gridCols;

    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const val = (board[r] && board[r][c]) || 0;
        const x = pad + c * (cellSize + pad);
        const y = pad + r * (cellSize + pad);

        const scheme = TILE_COLORS[val] || (val > 2048 ? TILE_COLORS[2048] : TILE_COLORS[0]);

        if (scheme.glow && val >= 64) {
          ctx.save();
          ctx.shadowColor = scheme.glow;
          ctx.shadowBlur = Math.min(16, cellSize * 0.25);
          ctx.fillStyle = scheme.bg;
          roundRect(ctx, x, y, cellSize, cellSize, 6);
          ctx.fill();
          ctx.restore();
        }

        ctx.fillStyle = scheme.bg;
        roundRect(ctx, x, y, cellSize, cellSize, 6);
        ctx.fill();

        if (val > 0) {
          ctx.fillStyle = scheme.text;
          const fontScale = val < 100 ? 0.44 : val < 1000 ? 0.36 : 0.3;
          const fontSize = Math.max(10, Math.round(cellSize * fontScale));
          ctx.font = `600 ${fontSize}px "IBM Plex Mono", monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(val), x + cellSize / 2, y + cellSize / 2 + 1);
        }
      }
    }
  }

  // Snake Renderer
  function renderSnake(canvas, record) {
    const setup = getCanvasContext(canvas);
    if (!setup || !record) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);

    // Board background
    ctx.fillStyle = "#0d1117";
    roundRect(ctx, 0, 0, width, height, 10);
    ctx.fill();

    const size = (record.board && record.board.size) || 8;
    const pad = Math.max(4, Math.round(width * 0.03));
    const boardW = width - pad * 2;
    const cellSize = boardW / size;

    // Grid lines
    ctx.strokeStyle = "#161d26";
    ctx.lineWidth = 1;
    for (let i = 0; i <= size; i++) {
      const p = pad + i * cellSize;
      ctx.beginPath();
      ctx.moveTo(p, pad);
      ctx.lineTo(p, pad + boardW);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pad, p);
      ctx.lineTo(pad + boardW, p);
      ctx.stroke();
    }

    const food = (record.board && record.board.food) || [0, 0];
    const snake = (record.board && record.board.snake) || [];

    // Food
    if (food && food.length >= 2) {
      const fx = pad + food[1] * cellSize + cellSize / 2;
      const fy = pad + food[0] * cellSize + cellSize / 2;
      const fr = cellSize * 0.32;

      ctx.save();
      ctx.shadowColor = "#f43f5e";
      ctx.shadowBlur = Math.min(14, cellSize * 0.4);
      ctx.fillStyle = "#fb7185";
      ctx.beginPath();
      ctx.arc(fx, fy, fr, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Snake Body
    if (snake.length > 0) {
      const segRadius = cellSize * 0.4;
      for (let i = snake.length - 1; i >= 0; i--) {
        const seg = snake[i];
        const sx = pad + seg[1] * cellSize + cellSize / 2;
        const sy = pad + seg[0] * cellSize + cellSize / 2;

        if (i === 0) {
          // Head
          ctx.save();
          ctx.shadowColor = "#34d399";
          ctx.shadowBlur = Math.min(12, cellSize * 0.3);
          ctx.fillStyle = "#34d399";
          ctx.beginPath();
          ctx.arc(sx, sy, segRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();

          // Subtle head dot/eyes
          ctx.fillStyle = "#04261a";
          ctx.beginPath();
          ctx.arc(sx, sy, segRadius * 0.28, 0, Math.PI * 2);
          ctx.fill();
        } else {
          // Body segment gradient
          const factor = 1 - (i / snake.length) * 0.6;
          ctx.fillStyle = `rgba(16, 185, 129, ${factor})`;
          ctx.beginPath();
          ctx.arc(sx, sy, segRadius * 0.9, 0, Math.PI * 2);
          ctx.fill();

          // Connect to next segment for continuous body
          const nextSeg = snake[i - 1];
          const nx = pad + nextSeg[1] * cellSize + cellSize / 2;
          const ny = pad + nextSeg[0] * cellSize + cellSize / 2;
          ctx.strokeStyle = `rgba(16, 185, 129, ${factor})`;
          ctx.lineWidth = segRadius * 1.8;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(nx, ny);
          ctx.stroke();
        }
      }
    }
  }

  // Othello Renderer
  function renderOthello(canvas, record) {
    const setup = getCanvasContext(canvas);
    if (!setup || !record) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);

    // Felt green background
    ctx.fillStyle = "#0d1b15";
    roundRect(ctx, 0, 0, width, height, 10);
    ctx.fill();

    const pad = Math.max(6, Math.round(width * 0.04));
    const boardW = width - pad * 2;
    const cellSize = boardW / 8;

    // Grid
    ctx.strokeStyle = "#173327";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 8; i++) {
      const p = pad + i * cellSize;
      ctx.beginPath();
      ctx.moveTo(p, pad);
      ctx.lineTo(p, pad + boardW);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pad, p);
      ctx.lineTo(pad + boardW, p);
      ctx.stroke();
    }

    // 4 Star points
    const starIndices = [2, 6];
    ctx.fillStyle = "#274d3d";
    for (const r of starIndices) {
      for (const c of starIndices) {
        ctx.beginPath();
        ctx.arc(pad + c * cellSize, pad + r * cellSize, Math.max(2, cellSize * 0.06), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Discs
    const rawBoard = record.board || [];
    const discRadius = cellSize * 0.4;

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        let cell = ".";
        if (Array.isArray(rawBoard) && rawBoard[r]) {
          cell = rawBoard[r][c] || ".";
        }
        if (cell === "." || cell === 0) continue;

        const cx = pad + c * cellSize + cellSize / 2;
        const cy = pad + r * cellSize + cellSize / 2;

        if (cell === "b" || cell === 1 || cell === "B") {
          // Black disc
          const grad = ctx.createRadialGradient(cx - discRadius * 0.3, cy - discRadius * 0.3, discRadius * 0.1, cx, cy, discRadius);
          grad.addColorStop(0, "#2a313a");
          grad.addColorStop(1, "#0f1216");
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, discRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#080a0c";
          ctx.lineWidth = 1;
          ctx.stroke();
        } else if (cell === "w" || cell === -1 || cell === "W") {
          // White disc
          const grad = ctx.createRadialGradient(cx - discRadius * 0.3, cy - discRadius * 0.3, discRadius * 0.1, cx, cy, discRadius);
          grad.addColorStop(0, "#ffffff");
          grad.addColorStop(1, "#c9d1dc");
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, discRadius, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#94a3b8";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
  }

  // Tetris Renderer
  const PIECE_COLORS = {
    I: "#38bdf8",
    O: "#fbbf24",
    T: "#a78bfa",
    S: "#34d399",
    Z: "#f43f5e",
    J: "#60a5fa",
    L: "#fb923c"
  };

  function renderTetris(canvas, record) {
    const setup = getCanvasContext(canvas);
    if (!setup || !record) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);

    // Deep slate matrix background
    ctx.fillStyle = "#0c1015";
    roundRect(ctx, 0, 0, width, height, 10);
    ctx.fill();

    const rawGrid = (record.board && (record.board.grid || record.board)) || [];
    const currentPiece = record.piece || (record.board && record.board.piece) || "Z";
    const pieceColor = PIECE_COLORS[currentPiece] || "#34d399";

    // Tetris is 20 rows by 10 columns
    const pad = Math.max(6, Math.round(width * 0.035));
    const matrixH = height - pad * 2;
    const matrixW = width - pad * 2;

    const cellH = matrixH / 20;
    const cellW = matrixW / 10;
    const cellSize = Math.min(cellW, cellH);

    const startX = pad + (matrixW - cellSize * 10) / 2;
    const startY = pad;

    // Grid lines
    ctx.strokeStyle = "#151b22";
    ctx.lineWidth = 1;
    for (let c = 0; c <= 10; c++) {
      ctx.beginPath();
      ctx.moveTo(startX + c * cellSize, startY);
      ctx.lineTo(startX + c * cellSize, startY + 20 * cellSize);
      ctx.stroke();
    }
    for (let r = 0; r <= 20; r++) {
      ctx.beginPath();
      ctx.moveTo(startX, startY + r * cellSize);
      ctx.lineTo(startX + 10 * cellSize, startY + r * cellSize);
      ctx.stroke();
    }

    // Grid cells: trace row 0 is the top row (site_traces.py already flips the engine's rows)
    for (let r = 0; r < 20; r++) {
      const row = rawGrid[r] || [];
      const canvasRow = r;
      const y = startY + canvasRow * cellSize;

      for (let c = 0; c < 10; c++) {
        if (row[c] === 1) {
          const x = startX + c * cellSize;
          ctx.fillStyle = pieceColor;
          roundRect(ctx, x + 1, y + 1, cellSize - 2, cellSize - 2, 2);
          ctx.fill();

          // Bevel highlight
          ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
          ctx.fillRect(x + 1, y + 1, cellSize - 2, 2);
          ctx.fillRect(x + 1, y + 1, 2, cellSize - 2);
        }
      }
    }
  }

  // Rounded rectangle helper
  function roundRect(ctx, x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ---------- TRACE LOADER ----------
  async function loadTraces() {
    let anyMock = false;

    for (const game of GAMES) {
      try {
        const res = await fetch(`traces/${game}.jsonl`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const lines = text.trim().split("\n");
        const records = [];
        for (const line of lines) {
          if (!line.trim()) continue;
          records.push(JSON.parse(line));
        }
        state.traces[game] = records;
        if (records.length > 0 && records[0].mock) {
          anyMock = true;
        }
      } catch (err) {
        console.error(`Failed to load traces/${game}.jsonl:`, err);
      }
    }

    if (anyMock) {
      const badge = document.getElementById("mock-badge");
      if (badge) badge.removeAttribute("hidden");
    }
  }

  // ---------- UI UPDATERS ----------

  // Update Mini Tile (Overview)
  function updateTile(game) {
    const records = state.traces[game];
    if (!records || records.length === 0) return;
    const step = state.steps[game] % records.length;
    const record = records[step];

    // Canvas
    const canvas = document.getElementById(`mini-canvas-${game}`);
    if (canvas) {
      if (game === "2048") render2048(canvas, record);
      else if (game === "snake") renderSnake(canvas, record);
      else if (game === "othello") renderOthello(canvas, record);
      else if (game === "tetris") renderTetris(canvas, record);
    }

    // Step badge & stats
    const stepEl = document.getElementById(`tile-step-${game}`);
    if (stepEl) stepEl.textContent = String(record.step !== undefined ? record.step : step);

    if (game === "2048") {
      const s = document.getElementById("tile-score-2048");
      const l = document.getElementById("tile-largest-2048");
      if (s) s.textContent = (record.stats && record.stats.score) || "0";
      if (l) l.textContent = (record.stats && record.stats.largest) || "2";
    } else if (game === "snake") {
      const s = document.getElementById("tile-score-snake");
      const l = document.getElementById("tile-length-snake");
      if (s) s.textContent = (record.stats && record.stats.score) || "0";
      if (l) l.textContent = (record.stats && record.stats.length) || "3";
    } else if (game === "othello") {
      const b = document.getElementById("tile-black-othello");
      const w = document.getElementById("tile-white-othello");
      if (b) b.textContent = (record.stats && record.stats.black) || "2";
      if (w) w.textContent = (record.stats && record.stats.white) || "2";
    } else if (game === "tetris") {
      const l = document.getElementById("tile-lines-tetris");
      const s = document.getElementById("tile-score-tetris");
      if (l) l.textContent = (record.stats && record.stats.lines) || "0";
      if (s) s.textContent = (record.stats && record.stats.score) || "0";
    }

    // Probabilities Panel
    const probsContainer = document.getElementById(`tile-probs-${game}`);
    if (probsContainer && record.options) {
      const topOptions = record.options.slice(0, 4);
      let html = "";
      for (const opt of topOptions) {
        const isChosen = opt.label === record.chosen;
        const pct = Math.round(opt.p * 100);
        html += `
          <div class="prob-row ${isChosen ? "chosen" : ""}">
            <span class="prob-label">${escapeHtml(opt.label)}</span>
            <div class="prob-track">
              <div class="prob-fill" style="width: ${pct}%"></div>
            </div>
            <span class="prob-val">${(opt.p).toFixed(2)}</span>
          </div>`;
      }
      probsContainer.innerHTML = html;
    }
  }

  // Update Focused View
  function updateFocusedView() {
    const game = state.activeGame;
    const records = state.traces[game];
    if (!records || records.length === 0) return;
    const step = state.steps[game] % records.length;
    const record = records[step];

    // Update strategy text
    const stratEl = document.getElementById("strategy-directive-text");
    if (stratEl) stratEl.textContent = STRATEGIES[game] || "";

    // Big Canvas
    const canvas = document.getElementById("focused-canvas");
    if (canvas) {
      if (game === "2048") render2048(canvas, record);
      else if (game === "snake") renderSnake(canvas, record);
      else if (game === "othello") renderOthello(canvas, record);
      else if (game === "tetris") renderTetris(canvas, record);
    }

    // HUD items
    const hudStep = document.getElementById("hud-step");
    if (hudStep) hudStep.textContent = String(record.step !== undefined ? record.step : step);

    const hudScore = document.getElementById("hud-score");
    if (hudScore) hudScore.textContent = (record.stats && record.stats.score !== undefined) ? String(record.stats.score) : "0";

    const lbl1 = document.getElementById("hud-extra-lbl-1");
    const val1 = document.getElementById("hud-extra-val-1");
    const lbl2 = document.getElementById("hud-extra-lbl-2");
    const val2 = document.getElementById("hud-extra-val-2");

    if (game === "2048") {
      if (lbl1) lbl1.textContent = "Largest";
      if (val1) val1.textContent = (record.stats && record.stats.largest) || "2";
      if (lbl2) lbl2.textContent = "Action";
      if (val2) val2.textContent = record.chosen || "--";
    } else if (game === "snake") {
      if (lbl1) lbl1.textContent = "Length";
      if (val1) val1.textContent = (record.stats && record.stats.length) || "3";
      if (lbl2) lbl2.textContent = "Action";
      if (val2) val2.textContent = record.chosen || "--";
    } else if (game === "othello") {
      if (lbl1) lbl1.textContent = "Black / White";
      if (val1) val1.textContent = `${record.stats && record.stats.black || 2} : ${record.stats && record.stats.white || 2}`;
      if (lbl2) lbl2.textContent = "Action";
      if (val2) val2.textContent = record.chosen || "--";
    } else if (game === "tetris") {
      if (lbl1) lbl1.textContent = "Lines";
      if (val1) val1.textContent = (record.stats && record.stats.lines !== undefined) ? String(record.stats.lines) : "0";
      if (lbl2) lbl2.textContent = "Piece";
      if (val2) val2.textContent = record.piece || (record.board && record.board.piece) || "--";
    }

    // Full Probabilities List
    const listEl = document.getElementById("full-probs-list");
    if (listEl && record.options) {
      let html = "";
      for (const opt of record.options) {
        const isChosen = opt.label === record.chosen;
        const pct = (opt.p * 100).toFixed(1);
        const pStr = (opt.p).toFixed(3);
        html += `
          <li class="full-prob-item ${isChosen ? "chosen pulse-chosen" : ""}">
            <div class="prob-item-header">
              <div class="prob-label-group">
                <span class="action-badge">${escapeHtml(opt.label)}</span>
                ${isChosen ? '<span class="chosen-tag"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg> Chosen</span>' : ""}
              </div>
              <span class="prob-score-num">${pct}% <span style="font-size: 0.8em; color: var(--fg-3)">(${pStr})</span></span>
            </div>
            <div class="prob-features-text">${escapeHtml(opt.text)}</div>
            <div class="prob-bar-container">
              <div class="prob-bar-fill" style="width: ${pct}%"></div>
            </div>
          </li>`;
      }
      listEl.innerHTML = html;
    }

    // Scrubber & step display
    const scrubber = document.getElementById("ctrl-scrubber");
    const stepDisplay = document.getElementById("ctrl-step-display");
    if (scrubber) {
      scrubber.max = String(records.length - 1);
      scrubber.value = String(step);
    }
    if (stepDisplay) {
      stepDisplay.textContent = `Step ${step + 1} / ${records.length}`;
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ---------- ADVANCE STEP ----------
  function advanceStep(game, delta = 1) {
    const records = state.traces[game];
    if (!records || records.length === 0) return;
    const len = records.length;
    state.prevSteps[game] = state.steps[game];
    state.steps[game] = (state.steps[game] + delta + len) % len;
  }

  function tick() {
    if (!state.isPlaying) return;

    if (state.viewMode === "overview") {
      for (const game of GAMES) {
        advanceStep(game, 1);
        updateTile(game);
      }
    } else {
      advanceStep(state.activeGame, 1);
      updateFocusedView();
    }
  }

  function startTimer() {
    if (state.timerId) clearInterval(state.timerId);
    if (!state.isPlaying) return;
    const interval = Math.round(BASE_INTERVAL / state.speed);
    state.timerId = setInterval(tick, interval);
  }

  function pausePlayback() {
    state.isPlaying = false;
    if (state.timerId) clearInterval(state.timerId);
    updatePlayPauseButton();
  }

  function resumePlayback() {
    state.isPlaying = true;
    updatePlayPauseButton();
    startTimer();
  }

  function togglePlayback() {
    if (state.isPlaying) pausePlayback();
    else resumePlayback();
  }

  function updatePlayPauseButton() {
    const playBtn = document.getElementById("ctrl-play");
    const iconPause = document.getElementById("icon-pause");
    const iconPlay = document.getElementById("icon-play");
    if (!playBtn) return;
    if (state.isPlaying) {
      if (iconPause) iconPause.removeAttribute("hidden");
      if (iconPlay) iconPlay.setAttribute("hidden", "");
      playBtn.setAttribute("aria-label", "Pause replay");
    } else {
      if (iconPause) iconPause.setAttribute("hidden", "");
      if (iconPlay) iconPlay.removeAttribute("hidden");
      playBtn.setAttribute("aria-label", "Play replay");
    }
  }

  // ---------- VIEW SWITCHING ----------
  function setViewMode(mode, game = null) {
    state.viewMode = mode;
    if (game && GAMES.includes(game)) {
      state.activeGame = game;
    }

    const overviewEl = document.getElementById("overview-view");
    const focusedEl = document.getElementById("focused-view");

    if (mode === "focused") {
      if (overviewEl) overviewEl.setAttribute("hidden", "");
      if (focusedEl) {
        focusedEl.removeAttribute("hidden");
        // Scroll so the top of the focused view is in view
        const targetTop = focusedEl.getBoundingClientRect().top + window.pageYOffset - 72;
        window.scrollTo({ top: Math.max(0, targetTop), behavior: prefersReducedMotion ? "auto" : "smooth" });
      }

      // Update tabs
      for (const g of GAMES) {
        const tab = document.getElementById(`tab-${g}`);
        if (tab) {
          const isSelected = g === state.activeGame;
          tab.setAttribute("aria-pressed", isSelected ? "true" : "false");
          tab.classList.toggle("active", isSelected);
        }
      }
      updateFocusedView();
    } else {
      if (focusedEl) focusedEl.setAttribute("hidden", "");
      if (overviewEl) overviewEl.removeAttribute("hidden");

      for (const g of GAMES) {
        updateTile(g);
      }
    }
  }

  // ---------- EVENT BINDINGS ----------
  function initEvents() {
    // Tile clicks (Overview -> Focused)
    for (const game of GAMES) {
      const tile = document.querySelector(`.replay-tile[data-game="${game}"]`);
      if (tile) {
        tile.addEventListener("click", () => setViewMode("focused", game));
        tile.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setViewMode("focused", game);
          }
        });
      }

      const focusBtn = document.querySelector(`.tile-focus-btn[data-game="${game}"]`);
      if (focusBtn) {
        focusBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          setViewMode("focused", game);
        });
      }
    }

    // Back button
    const backBtn = document.getElementById("btn-back-overview");
    if (backBtn) {
      backBtn.addEventListener("click", () => setViewMode("overview"));
    }

    // Game tabs in focused view
    for (const game of GAMES) {
      const tab = document.getElementById(`tab-${game}`);
      if (tab) {
        tab.addEventListener("click", () => {
          state.activeGame = game;
          for (const g of GAMES) {
            const t = document.getElementById(`tab-${g}`);
            if (t) {
              const sel = g === game;
              t.setAttribute("aria-pressed", sel ? "true" : "false");
              t.classList.toggle("active", sel);
            }
          }
          updateFocusedView();
        });
      }
    }

    // Play/Pause button
    const playBtn = document.getElementById("ctrl-play");
    if (playBtn) {
      playBtn.addEventListener("click", togglePlayback);
    }

    // Step Prev/Next
    const prevBtn = document.getElementById("ctrl-prev");
    if (prevBtn) {
      prevBtn.addEventListener("click", () => {
        pausePlayback();
        advanceStep(state.activeGame, -1);
        updateFocusedView();
      });
    }

    const nextBtn = document.getElementById("ctrl-next");
    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        pausePlayback();
        advanceStep(state.activeGame, 1);
        updateFocusedView();
      });
    }

    // Scrubber
    const scrubber = document.getElementById("ctrl-scrubber");
    if (scrubber) {
      scrubber.addEventListener("input", (e) => {
        pausePlayback();
        const val = parseInt(e.target.value, 10);
        state.steps[state.activeGame] = val;
        updateFocusedView();
      });
    }

    // Speed buttons
    const speedBtns = document.querySelectorAll(".speed-btn");
    speedBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const spd = parseFloat(btn.dataset.speed);
        state.speed = spd;
        speedBtns.forEach((b) => b.classList.toggle("active", b === btn));
        if (state.isPlaying) startTimer();
      });
    });

    // Keyboard navigation
    window.addEventListener("keydown", (e) => {
      // Don't intercept if user is inside a form field
      // Leave keys to native controls (buttons, links, form fields) and to modified shortcuts
      const t = e.target;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      if (t && t.closest && t.closest("button, a, input, textarea, select, [contenteditable], [role=slider]")) return;

      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        togglePlayback();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        pausePlayback();
        if (state.viewMode === "focused") {
          advanceStep(state.activeGame, -1);
          updateFocusedView();
        } else {
          for (const g of GAMES) advanceStep(g, -1);
          for (const g of GAMES) updateTile(g);
        }
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        pausePlayback();
        if (state.viewMode === "focused") {
          advanceStep(state.activeGame, 1);
          updateFocusedView();
        } else {
          for (const g of GAMES) advanceStep(g, 1);
          for (const g of GAMES) updateTile(g);
        }
      }
    });

    // Resize handling for canvases
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (state.viewMode === "overview") {
          for (const g of GAMES) updateTile(g);
        } else {
          updateFocusedView();
        }
      }, 100);
    });

    // Reduced motion change listener
    reducedMotionQuery.addEventListener("change", (e) => {
      prefersReducedMotion = e.matches;
      if (prefersReducedMotion) {
        pausePlayback();
      }
    });

    // Mobile menu helper matching site.js
    const menu = document.querySelector(".menu");
    if (menu) {
      menu.addEventListener("click", (e) => {
        if (e.target.closest("a")) menu.open = false;
      });
    }
  }

  // ---------- INITIALIZATION ----------
  async function init() {
    await loadTraces();

    // Render initial tiles
    for (const g of GAMES) {
      updateTile(g);
    }

    initEvents();
    updatePlayPauseButton();

    if (state.isPlaying) {
      startTimer();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
