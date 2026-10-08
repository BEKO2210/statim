/* Browser ports of demos/actions.  Keep this file dependency-free. */

export class PythonRandom {
  constructor(seed = 0) {
    this.mt = new Uint32Array(624);
    this.index = 624;
    this.seed(seed);
  }

  seed(value) {
    let n;
    if (typeof value === "bigint") n = value < 0n ? -value : value;
    else {
      if (!Number.isSafeInteger(value)) throw new TypeError("seed must be a safe integer or bigint");
      n = BigInt(Math.abs(value));
    }
    const key = [];
    do {
      key.push(Number(n & 0xffffffffn));
      n >>= 32n;
    } while (n);
    this._initByArray(key);
  }

  _initGenrand(seed) {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(previous, 1812433253) + i) >>> 0;
    }
    this.index = 624;
  }

  _initByArray(key) {
    this._initGenrand(19650218);
    let i = 1;
    let j = 0;
    let k = Math.max(624, key.length);
    for (; k; k--) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous, 1664525)) + key[j] + j) >>> 0;
      i++;
      j++;
      if (i >= 624) {
        this.mt[0] = this.mt[623];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (k = 623; k; k--) {
      const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous, 1566083941)) - i) >>> 0;
      i++;
      if (i >= 624) {
        this.mt[0] = this.mt[623];
        i = 1;
      }
    }
    this.mt[0] = 0x80000000;
  }

  _genrand() {
    if (this.index >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = (this.mt[i] & 0x80000000) | (this.mt[(i + 1) % 624] & 0x7fffffff);
        this.mt[i] = (this.mt[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0)) >>> 0;
      }
      this.index = 0;
    }
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  random() {
    const a = this._genrand() >>> 5;
    const b = this._genrand() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  getrandbits(k) {
    if (!Number.isInteger(k) || k < 0) throw new TypeError("number of bits must be a non-negative integer");
    if (k === 0) return 0n;
    if (k <= 32) return BigInt(this._genrand() >>> (32 - k));
    const words = Math.ceil(k / 32);
    let result = 0n;
    for (let i = 0; i < words; i++) {
      let word = this._genrand();
      if (i === words - 1 && (k % 32)) word >>>= 32 - (k % 32);
      result |= BigInt(word) << BigInt(32 * i);
    }
    return result;
  }

  _randbelow(n) {
    if (!Number.isSafeInteger(n) || n <= 0) throw new RangeError("n must be a positive safe integer");
    const bits = n.toString(2).length;
    let value;
    do value = Number(this.getrandbits(bits)); while (value >= n);
    return value;
  }

  randrange(start, stop = undefined, step = 1) {
    if (![start, step].every(Number.isSafeInteger) || (stop !== undefined && !Number.isSafeInteger(stop))) {
      throw new TypeError("randrange arguments must be safe integers");
    }
    if (stop === undefined) {
      stop = start;
      start = 0;
    }
    if (step === 0) throw new RangeError("zero step for randrange()");
    const width = stop - start;
    let count;
    if (step === 1) count = width;
    else if (step > 0) count = Math.floor((width + step - 1) / step);
    else count = Math.floor((width + step + 1) / step);
    if (count <= 0) throw new RangeError("empty range for randrange()");
    return start + step * this._randbelow(count);
  }

  choice(sequence) {
    if (!sequence.length) throw new RangeError("cannot choose from an empty sequence");
    return sequence[this._randbelow(sequence.length)];
  }

  shuffle(sequence) {
    for (let i = sequence.length - 1; i > 0; i--) {
      const j = this._randbelow(i + 1);
      [sequence[i], sequence[j]] = [sequence[j], sequence[i]];
    }
    return sequence;
  }
}

const cloneBoard = board => board.map(row => row.slice());
const equalBoard = (a, b) => a.every((row, r) => row.every((cell, c) => cell === b[r][c]));

class Game2048 {
  constructor(seed = 0, board = null) {
    this.name = "2048";
    this.rng = new PythonRandom(seed);
    this.board = board ? cloneBoard(board) : Array.from({length: 4}, () => Array(4).fill(0));
    this.steps = 0;
    this.points = 0;
    if (!board) { this._spawn(); this._spawn(); }
    this.done = this.legalActions().length === 0;
  }

  static _mergeLine(line) {
    const values = line.filter(Boolean);
    const out = [];
    let merges = 0;
    let mergedValue = 0;
    for (let i = 0; i < values.length;) {
      if (i + 1 < values.length && values[i] === values[i + 1]) {
        const value = values[i] * 2;
        out.push(value); merges++; mergedValue += value; i += 2;
      } else out.push(values[i++]);
    }
    while (out.length < 4) out.push(0);
    return [out, merges, mergedValue];
  }

  _moved(action) {
    if (!["left", "right", "up", "down"].includes(action)) throw new Error(`unknown 2048 action: ${action}`);
    const result = Array.from({length: 4}, () => Array(4).fill(0));
    let merges = 0;
    let mergedValue = 0;
    for (let index = 0; index < 4; index++) {
      let line = (action === "left" || action === "right")
        ? this.board[index].slice() : this.board.map(row => row[index]);
      if (action === "right" || action === "down") line.reverse();
      let made, value;
      [line, made, value] = Game2048._mergeLine(line);
      if (action === "right" || action === "down") line.reverse();
      merges += made; mergedValue += value;
      line.forEach((cell, offset) => {
        if (action === "left" || action === "right") result[index][offset] = cell;
        else result[offset][index] = cell;
      });
    }
    return [result, merges, mergedValue];
  }

  _spawn() {
    const empty = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) if (!this.board[r][c]) empty.push([r, c]);
    const rankDraw = this.rng.random();
    const valueDraw = this.rng.random();
    if (empty.length) {
      const [r, c] = empty[Math.min(Math.trunc(rankDraw * empty.length), empty.length - 1)];
      this.board[r][c] = valueDraw < 0.9 ? 2 : 4;
    }
  }

  legalActions() {
    return ["left", "right", "up", "down"].filter(action => !equalBoard(this._moved(action)[0], this.board));
  }

  static _monotonicCount(board) {
    const lines = cloneBoard(board);
    for (let c = 0; c < 4; c++) lines.push(board.map(row => row[c]));
    let count = 0;
    for (const line of lines) {
      const values = line.filter(Boolean);
      const ascending = values.slice().sort((a, b) => a - b);
      if (values.every((v, i) => v === ascending[i]) || values.every((v, i) => v === ascending[ascending.length - 1 - i])) count++;
    }
    return count;
  }

  features(action) {
    const [board, merges, mergedValue] = this._moved(action);
    const currentLargest = Math.max(...this.board.flat());
    const currentCorners = [this.board[0][0], this.board[0][3], this.board[3][0], this.board[3][3]];
    const largest = Math.max(...board.flat());
    const corners = [board[0][0], board[0][3], board[3][0], board[3][3]];
    const wasCorner = currentCorners.includes(currentLargest);
    const isCorner = corners.includes(largest);
    const mergeValues = [];
    for (let index = 0; index < 4; index++) {
      let line = (action === "left" || action === "right") ? this.board[index].slice() : this.board.map(row => row[index]);
      if (action === "right" || action === "down") line.reverse();
      const values = line.filter(Boolean);
      for (let offset = 0; offset + 1 < values.length;) {
        if (values[offset] === values[offset + 1]) { mergeValues.push(values[offset] * 2); offset += 2; }
        else offset++;
      }
    }
    return {
      merges, merged_value: mergedValue,
      largest_merge: mergeValues.length ? Math.max(...mergeValues) : 0,
      empty: board.flat().filter(cell => cell === 0).length,
      corner: isCorner,
      corner_status: wasCorner && isCorner ? "kept" : wasCorner ? "lost" : isCorner ? "gained" : "absent",
      monotonic: Game2048._monotonicCount(board),
    };
  }

  apply(action) {
    if (!this.legalActions().includes(action)) throw new Error(`illegal 2048 action: ${action}`);
    let mergedValue;
    [this.board, , mergedValue] = this._moved(action);
    this.points += mergedValue;
    this._spawn();
    this.steps++;
    this.done = this.legalActions().length === 0;
  }

  get score() { return this.points; }
  renderText() { return this.board.map(row => row.map(cell => cell || ".").join(" ")).join("/"); }
  finalStats() { return {largest_tile: Math.max(...this.board.flat())}; }
}

const SNAKE_VECTORS = {up: [-1, 0], right: [0, 1], down: [1, 0], left: [0, -1]};
const SNAKE_OPPOSITES = {up: "down", down: "up", left: "right", right: "left"};
const cellKey = cell => `${cell[0]},${cell[1]}`;

class Snake {
  constructor(seed = 0, snake = null, food = null, direction = "right") {
    this.name = "snake";
    this.rng = new PythonRandom(seed);
    this.snake = snake ? snake.map(cell => cell.slice()) : [[4, 4], [4, 3], [4, 2]];
    this.direction = direction;
    this.food = food ? food.slice() : null;
    this.steps = 0; this.foods = 0; this.done = false; this.won = false;
    if (this.food === null) this._placeFood();
  }

  _placeFood() {
    const occupied = new Set(this.snake.map(cellKey));
    const free = [];
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) if (!occupied.has(`${r},${c}`)) free.push([r, c]);
    const rankDraw = this.rng.random();
    this.rng.random();
    this.food = free.length ? free[Math.min(Math.trunc(rankDraw * free.length), free.length - 1)] : null;
    if (!free.length) { this.done = true; this.won = true; }
  }

  legalActions() {
    if (this.done) return [];
    return Object.keys(SNAKE_VECTORS).filter(action => action !== SNAKE_OPPOSITES[this.direction]);
  }

  _next(action) {
    const [dr, dc] = SNAKE_VECTORS[action];
    const head = this.snake[0];
    const newHead = [head[0] + dr, head[1] + dc];
    const grows = this.food !== null && newHead[0] === this.food[0] && newHead[1] === this.food[1];
    const relevant = grows ? this.snake : this.snake.slice(0, -1);
    const occupied = new Set(relevant.map(cellKey));
    const inBounds = newHead[0] >= 0 && newHead[0] < 8 && newHead[1] >= 0 && newHead[1] < 8;
    const death = !inBounds || occupied.has(cellKey(newHead));
    const body = [newHead, ...(grows ? this.snake : this.snake.slice(0, -1))].map(cell => cell.slice());
    return [newHead, body, grows, death];
  }

  static _reachable(head, blockedCells) {
    const blocked = new Set(blockedCells.map(cellKey));
    if (head[0] < 0 || head[0] >= 8 || head[1] < 0 || head[1] >= 8 || blocked.has(cellKey(head))) return 0;
    const seen = new Set([cellKey(head)]);
    const queue = [head];
    for (let at = 0; at < queue.length; at++) {
      const [r, c] = queue[at];
      for (const [dr, dc] of Object.values(SNAKE_VECTORS)) {
        const next = [r + dr, c + dc];
        const key = cellKey(next);
        if (next[0] >= 0 && next[0] < 8 && next[1] >= 0 && next[1] < 8 && !blocked.has(key) && !seen.has(key)) {
          seen.add(key); queue.push(next);
        }
      }
    }
    return seen.size;
  }

  features(action) {
    if (!this.legalActions().includes(action)) throw new Error(`illegal snake action: ${action}`);
    const [head, body, grows, death] = this._next(action);
    const distance = this.food === null ? 0 : Math.abs(head[0] - this.food[0]) + Math.abs(head[1] - this.food[1]);
    const inBounds = head[0] >= 0 && head[0] < 8 && head[1] >= 0 && head[1] < 8;
    return {
      food_distance: distance,
      reachable: death ? 0 : Snake._reachable(head, body.slice(1)),
      death,
      death_reason: death ? (inBounds ? "snake" : "wall") : null,
      eats: grows && !death,
    };
  }

  apply(action) {
    if (!this.legalActions().includes(action)) throw new Error(`illegal snake action: ${action}`);
    const [, body, grows, death] = this._next(action);
    this.direction = action; this.steps++;
    if (death) { this.done = true; return; }
    this.snake = body;
    if (grows) { this.foods++; this._placeFood(); }
  }

  get score() { return this.foods; }
  renderText() {
    const cells = Array.from({length: 8}, () => Array(8).fill("."));
    if (this.food) cells[this.food[0]][this.food[1]] = "F";
    for (const [r, c] of this.snake.slice(1)) cells[r][c] = "o";
    cells[this.snake[0][0]][this.snake[0][1]] = "H";
    return cells.map(row => row.join("")).join("/");
  }
  finalStats() { return {length: this.snake.length, won: this.won}; }
}

const OTHELLO_DIRECTIONS = [];
for (const dr of [-1, 0, 1]) for (const dc of [-1, 0, 1]) if (dr || dc) OTHELLO_DIRECTIONS.push([dr, dc]);
const OTHELLO_WEIGHTS = [
  120,-20,20,5,5,20,-20,120, -20,-40,-5,-5,-5,-5,-40,-20,
  20,-5,15,3,3,15,-5,20, 5,-5,3,3,3,3,-5,5,
  5,-5,3,3,3,3,-5,5, 20,-5,15,3,3,15,-5,20,
  -20,-40,-5,-5,-5,-5,-40,-20, 120,-20,20,5,5,20,-20,120,
];

class Othello {
  constructor(seed = 0, board = null) {
    this.name = "othello";
    this.board = board ? board.slice() : Array(64).fill(0);
    if (!board) { this.board[27] = this.board[36] = -1; this.board[28] = this.board[35] = 1; }
    this.steps = 0; this.opening_moves = [];
    if (!board) {
      const rng = new PythonRandom(seed);
      let player = 1;
      for (let turn = 0; turn < 4; turn++) {
        const moves = this._moves(player);
        if (moves.length) { const move = rng.choice(moves); this._put(move, player); this.opening_moves.push(Othello._label(move)); }
        else this.opening_moves.push("pass");
        player = -player;
      }
    }
    this.done = !this._moves(1).length && !this._moves(-1).length;
  }

  static _label(index) { return String.fromCharCode(97 + index % 8) + String(Math.floor(index / 8) + 1); }
  static _index(label) {
    if (!/^[a-h][1-8]$/.test(label)) throw new Error(`bad Othello square: ${label}`);
    return (Number(label[1]) - 1) * 8 + label.charCodeAt(0) - 97;
  }
  _flips(index, player) {
    if (this.board[index] !== 0) return [];
    const row = Math.floor(index / 8), col = index % 8, result = [];
    for (const [dr, dc] of OTHELLO_DIRECTIONS) {
      const line = [];
      let r = row + dr, c = col + dc;
      while (r >= 0 && r < 8 && c >= 0 && c < 8 && this.board[r * 8 + c] === -player) {
        line.push(r * 8 + c); r += dr; c += dc;
      }
      if (line.length && r >= 0 && r < 8 && c >= 0 && c < 8 && this.board[r * 8 + c] === player) result.push(...line);
    }
    return result;
  }
  _moves(player) { return Array.from({length: 64}, (_, i) => i).filter(i => this._flips(i, player).length); }
  legalActions() {
    if (this.done) return [];
    const moves = this._moves(1);
    return moves.length ? moves.map(Othello._label) : ["pass"];
  }
  _put(index, player) {
    const flips = this._flips(index, player);
    if (!flips.length) throw new Error("illegal Othello move");
    this.board[index] = player;
    for (const flipped of flips) this.board[flipped] = player;
    return flips.length;
  }
  features(action) {
    if (action === "pass") {
      if (this._moves(1).length) throw new Error("illegal Othello pass");
      return {flips: 0, corner: false, gives_corner: false, opp_mobility: this._moves(-1).length};
    }
    const index = Othello._index(action), board = this.board.slice(), flips = this._flips(index, 1);
    if (!flips.length) throw new Error(`illegal Othello action: ${action}`);
    this._put(index, 1);
    const opponentMoves = this._moves(-1);
    const result = {flips: flips.length, corner: [0,7,56,63].includes(index),
      gives_corner: opponentMoves.some(move => [0,7,56,63].includes(move)),
      opp_mobility: opponentMoves.length, positional: OTHELLO_WEIGHTS[index]};
    this.board = board;
    return result;
  }
  apply(action) {
    if (!this.legalActions().includes(action)) throw new Error(`illegal Othello action: ${action}`);
    if (action !== "pass") this._put(Othello._index(action), 1);
    this.steps++;
    const opponent = this._moves(-1);
    if (opponent.length) {
      opponent.sort((a, b) => OTHELLO_WEIGHTS[b] - OTHELLO_WEIGHTS[a] || a - b);
      this._put(opponent[0], -1);
    }
    this.done = !this._moves(1).length && !this._moves(-1).length;
  }
  get score() { return this.board.filter(x => x === 1).length - this.board.filter(x => x === -1).length; }
  renderText() {
    const symbols = {["0"]: ".", ["1"]: "B", ["-1"]: "W"};
    return Array.from({length: 8}, (_, r) => this.board.slice(r * 8, r * 8 + 8).map(x => symbols[x]).join("")).join("/");
  }
  finalStats() {
    const black = this.board.filter(x => x === 1).length, white = this.board.filter(x => x === -1).length;
    return {black_discs: black, white_discs: white, win: black > white ? "black" : white > black ? "white" : "draw"};
  }
}

const TETROMINOES = {
  I: [[[0,0],[1,0],[2,0],[3,0]], [[0,0],[0,1],[0,2],[0,3]]],
  O: [[[0,0],[1,0],[0,1],[1,1]]],
  T: [[[0,0],[1,0],[2,0],[1,1]], [[0,0],[0,1],[0,2],[1,1]], [[1,0],[0,1],[1,1],[2,1]], [[1,0],[0,1],[1,1],[1,2]]],
  S: [[[0,0],[1,0],[1,1],[2,1]], [[1,0],[0,1],[1,1],[0,2]]],
  Z: [[[1,0],[2,0],[0,1],[1,1]], [[0,0],[0,1],[1,1],[1,2]]],
  J: [[[0,0],[1,0],[2,0],[0,1]], [[0,0],[1,0],[1,1],[1,2]], [[2,0],[0,1],[1,1],[2,1]], [[0,0],[0,1],[0,2],[1,2]]],
  L: [[[0,0],[1,0],[2,0],[2,1]], [[0,0],[0,1],[0,2],[1,0]], [[0,0],[0,1],[1,1],[2,1]], [[0,2],[1,0],[1,1],[1,2]]],
};

class Tetris {
  constructor(seed = 0, board = null, current = null) {
    this.name = "tetris";
    this.rng = new PythonRandom(seed);
    this.board = board ? cloneBoard(board) : Array.from({length: 20}, () => Array(10).fill(0));
    this.bag = []; this.steps = 0; this.lines = 0;
    this.current = current || this._draw();
    this.done = !this._canSpawn(this.current);
  }
  _draw() {
    if (!this.bag.length) { this.bag = Object.keys(TETROMINOES); this.rng.shuffle(this.bag); }
    return this.bag.pop();
  }
  static _bounds(shape) { return [Math.max(...shape.map(x => x[0])) + 1, Math.max(...shape.map(x => x[1])) + 1]; }
  _canSpawn(piece) {
    const shape = TETROMINOES[piece][0], [width, height] = Tetris._bounds(shape);
    const x = Math.floor((10 - width) / 2), y = 20 - height;
    return shape.every(([dx, dy]) => !this.board[y + dy][x + dx]);
  }
  static _label(piece, rotation, column) { return `${piece}:r${rotation}:x${column}`; }
  static _parse(action) { const [piece, rotation, column] = action.split(":"); return [piece, Number(rotation.slice(1)), Number(column.slice(1))]; }
  _dropY(shape, column) {
    const [width, height] = Tetris._bounds(shape);
    if (column < 0 || column + width > 10) return null;
    let y = 20 - height;
    if (shape.some(([dx, dy]) => this.board[y + dy][column + dx])) return null;
    while (y > 0 && shape.every(([dx, dy]) => !this.board[y - 1 + dy][column + dx])) y--;
    return y;
  }
  legalActions() {
    if (this.done) return [];
    const result = [];
    TETROMINOES[this.current].forEach((shape, rotation) => {
      const [width] = Tetris._bounds(shape);
      for (let column = 0; column < 10 - width + 1; column++) if (this._dropY(shape, column) !== null) result.push(Tetris._label(this.current, rotation, column));
    });
    return result;
  }
  _placed(action) {
    const [piece, rotation, column] = Tetris._parse(action);
    if (piece !== this.current || rotation < 0 || rotation >= TETROMINOES[piece].length) throw new Error(`illegal Tetris action: ${action}`);
    const shape = TETROMINOES[piece][rotation], y = this._dropY(shape, column);
    if (y === null) throw new Error(`illegal Tetris action: ${action}`);
    let board = cloneBoard(this.board);
    for (const [dx, dy] of shape) board[y + dy][column + dx] = 1;
    const kept = board.filter(row => !row.every(Boolean));
    const cleared = 20 - kept.length;
    board = kept.concat(Array.from({length: cleared}, () => Array(10).fill(0)));
    return [board, cleared];
  }
  static _boardFeatures(board) {
    const heights = []; let holes = 0;
    for (let column = 0; column < 10; column++) {
      const occupied = [];
      for (let row = 0; row < 20; row++) if (board[row][column]) occupied.push(row);
      const height = occupied.length ? Math.max(...occupied) + 1 : 0;
      heights.push(height);
      for (let row = 0; row < height; row++) if (!board[row][column]) holes++;
    }
    let bumpiness = 0;
    for (let i = 0; i < heights.length - 1; i++) bumpiness += Math.abs(heights[i] - heights[i + 1]);
    return {holes, aggregate_height: heights.reduce((a, b) => a + b, 0), bumpiness, max_height: Math.max(...heights)};
  }
  features(action) {
    const [board, cleared] = this._placed(action);
    const beforeHoles = Tetris._boardFeatures(this.board).holes, result = Tetris._boardFeatures(board);
    return {lines: cleared, new_holes: Math.max(0, result.holes - beforeHoles), ...result};
  }
  apply(action) {
    if (!this.legalActions().includes(action)) throw new Error(`illegal Tetris action: ${action}`);
    let cleared;
    [this.board, cleared] = this._placed(action);
    this.lines += cleared; this.steps++; this.current = this._draw(); this.done = !this._canSpawn(this.current);
  }
  get score() { return this.lines; }
  renderText() { return `piece=${this.current};` + this.board.slice().reverse().map(row => row.map(x => x ? "#" : ".").join("")).join("/"); }
  finalStats() { return {lines: this.lines}; }
}

const STRATEGIES = {
  "2048": "Keep the largest tile in a corner; prefer merges and many empty cells.",
  snake: "Reach food safely while preserving enough open space to avoid trapping the snake.",
  othello: "Take corners, avoid giving corners, limit opponent mobility, and gain discs.",
  tetris: "Clear lines while keeping the stack low, even, and free of holes.",
};
const plural = (number, singular) => number === 1 ? singular : `${singular}s`;

function sentence2048(label, facts) {
  const merge = facts.largest_merge;
  const start = merge ? `Move ${label} merges two ${Math.floor(merge / 2)} tiles into a ${merge}` : `Move ${label} merges nothing`;
  const corner = {kept: "keeps the largest tile in the corner", lost: "moves the largest tile out of the corner",
    gained: "moves the largest tile into a corner", absent: "leaves the largest tile out of the corners"}[facts.corner_status];
  return `${start} and ${corner}; ${facts.empty} cells stay empty.`;
}
function sentenceSnake(label, facts, game) {
  if (facts.death) return `Move ${label} hits the ${facts.death_reason === "wall" ? "wall" : "snake"} and the snake dies.`;
  if (facts.eats) return `Move ${label} eats the food.`;
  if (facts.reachable < game.snake.length + 2) return `Move ${label} is safe but traps the snake in ${facts.reachable} free cells.`;
  const current = Math.abs(game.snake[0][0] - game.food[0]) + Math.abs(game.snake[0][1] - game.food[1]);
  const relation = facts.food_distance < current ? "brings the head closer to" : facts.food_distance > current ? "moves the head farther from" : "keeps the head the same distance from";
  return `Move ${label} is safe and ${relation} the food (${facts.food_distance} steps away).`;
}
function sentenceOthello(label, facts) {
  if (label === "pass") return `Move pass flips no discs and leaves the opponent ${facts.opp_mobility} moves.`;
  if (facts.corner) return `Move ${label} takes a corner and flips ${facts.flips} ${plural(facts.flips, "disc")}.`;
  if (facts.gives_corner) return `Move ${label} flips ${facts.flips} ${plural(facts.flips, "disc")} but gives the opponent a corner.`;
  return `Move ${label} flips ${facts.flips} ${plural(facts.flips, "disc")} and leaves the opponent ${facts.opp_mobility} ${plural(facts.opp_mobility, "move")}.`;
}
function sentenceTetris(label, facts, action) {
  const [, rotation, column] = Tetris._parse(action);
  const lines = facts.lines ? `clears ${facts.lines} ${plural(facts.lines, "line")}` : "clears no lines";
  const holes = facts.holes ? `leaves ${facts.new_holes} new ${plural(facts.new_holes, "hole")}` : "leaves no holes";
  const display = /^p\d+$/.test(label) ? label.slice(1) : label;
  return `Placement ${display} (rotation ${rotation}, column ${column}) ${lines} and ${holes}; the stack is ${facts.max_height} ${plural(facts.max_height, "row")} high.`;
}
const tetrisPrefilterScore = f => 8 * f.lines - 7 * f.holes - f.aggregate_height - f.bumpiness - 2 * f.max_height;

function decisionOptions(game) {
  let actions = game.legalActions();
  const facts = {};
  for (const action of actions) facts[action] = game.features(action);
  const total = actions.length;
  let applied = false;
  if (game.name === "tetris" && actions.length > 12) {
    actions = actions.slice().sort((a, b) => tetrisPrefilterScore(facts[b]) - tetrisPrefilterScore(facts[a]) || (a < b ? -1 : a > b ? 1 : 0)).slice(0, 12);
    applied = true;
  }
  let choices = {};
  actions.forEach((action, i) => { choices[game.name === "tetris" ? `p${i + 1}` : action] = action; });
  return [choices, facts, {applied, legal_count: total, sent_count: actions.length}];
}

function requestParts(game) {
  let [choices, facts, prefilter] = decisionOptions(game);
  if (game.name === "tetris") {
    const moveChoices = {};
    for (const action of Object.values(choices)) moveChoices[action] = action;
    choices = moveChoices;
  }
  const sentenceMap = {};
  const texts = [];
  for (const [label, action] of Object.entries(choices)) {
    let sentence;
    if (game.name === "2048") sentence = sentence2048(label, facts[action]);
    else if (game.name === "snake") sentence = sentenceSnake(label, facts[action], game);
    else if (game.name === "othello") sentence = sentenceOthello(label, facts[action]);
    else sentence = sentenceTetris(label, facts[action], action);
    if (sentence.length > 140) throw new Error("move sentence exceeded 140 characters");
    sentenceMap[label] = sentence; texts.push(sentence);
  }
  const title = game.name.charAt(0).toUpperCase() + game.name.slice(1).toLowerCase();
  const state = `${title} game. ${texts.join(" ")}`;
  if (state.length > 1800) throw new Error("state exceeded 1,800 characters");
  const criteria = {};
  for (const label of Object.keys(choices)) criteria[label] = null;
  const payload = {state, questions: {move: {type: "choice", instructions: `${STRATEGIES[game.name]} Which move is best?`, criteria}}};
  return {payload, choices, prefilter, sentenceMap};
}

export function buildRequest(game) {
  const {payload, choices, prefilter} = requestParts(game);
  return {payload, choices, prefilter};
}

export function sentences(game) { return requestParts(game).sentenceMap; }

export function heuristicScores(game) {
  const scores = {};
  for (const action of game.legalActions()) {
    const f = game.features(action);
    if (game.name === "2048") scores[action] = 12 * f.merges + .25 * f.merged_value + 3 * f.empty + 15 * Number(f.corner) + 2 * f.monotonic;
    else if (game.name === "snake") scores[action] = -10000 * Number(f.death) + 50 * Number(f.eats) + f.reachable - 2 * f.food_distance;
    else if (game.name === "othello") scores[action] = 100 * Number(f.corner) - 100 * Number(f.gives_corner) + (f.positional || 0) + 2 * f.flips - 4 * f.opp_mobility;
    else scores[action] = 10 * f.lines - 8 * f.holes - .5 * f.aggregate_height - .8 * f.bumpiness - f.max_height;
  }
  return scores;
}

export function heuristicChoice(game) {
  const scores = heuristicScores(game), actions = Object.keys(scores);
  actions.sort((a, b) => scores[b] - scores[a] || (a < b ? -1 : a > b ? 1 : 0));
  return actions[0];
}

export function createGame(name, seed = 0) {
  const classes = {"2048": Game2048, snake: Snake, othello: Othello, tetris: Tetris};
  if (!classes[name]) throw new Error(`unknown game: ${name}`);
  return new classes[name](seed);
}
