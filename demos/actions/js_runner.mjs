import {PythonRandom, buildRequest, createGame, heuristicChoice} from "../../site/actions/engines.js";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const command = JSON.parse(input);

function snapshot(game) {
  const request = buildRequest(game);
  const board = game.name === "snake"
    ? {snake: game.snake.map(cell => cell.slice()), food: game.food && game.food.slice(), direction: game.direction}
    : (Array.isArray(game.board[0]) ? game.board.map(row => row.slice()) : game.board.slice());
  return {
    board,
    score: game.score,
    done: game.done,
    legal_actions: game.legalActions(),
    request: JSON.stringify(request.payload),
    choices: request.choices,
    heuristic_choice: game.done ? null : heuristicChoice(game),
    final_stats: game.finalStats(),
  };
}

function runCase(testCase) {
  const game = createGame(testCase.game, testCase.seed);
  const states = [];
  for (const action of testCase.moves) {
    states.push(snapshot(game));
    game.apply(action);
  }
  states.push(snapshot(game));
  return states;
}

function runRng(spec) {
  const randomRng = new PythonRandom(BigInt(spec.seed));
  const random = Array.from({length: spec.count}, () => randomRng.random());
  const resultRng = new PythonRandom(BigInt(spec.seed));
  const results = [];
  for (let i = 0; i < spec.count; i++) {
    if (i % 3 === 0) results.push(resultRng.randrange(1, 1000003, 7));
    else if (i % 3 === 1) results.push(resultRng.randrange(-5000, 8000));
    else results.push(resultRng.choice(["a", "b", "c", "d", "e", "f", "g"]));
  }
  return {random, results};
}

const output = {
  cases: (command.cases || []).map(runCase),
  rng: (command.rng || []).map(runRng),
};
process.stdout.write(JSON.stringify(output));
