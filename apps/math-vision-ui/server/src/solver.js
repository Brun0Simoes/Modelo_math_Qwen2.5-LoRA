import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const PROJECT_ROOT = process.env.PROJECT_ROOT ?? "E:\\IA_matematica";
const PYTHON_EXE =
  process.env.PYTHON_EXE ?? path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe");
const SOLVE_SCRIPT =
  process.env.SOLVE_SCRIPT ?? path.join(PROJECT_ROOT, "scripts", "solve_problem_json.py");
const DEFAULT_MODEL = process.env.MODEL_NAME ?? "Qwen/Qwen2.5-Math-7B-Instruct";
const SOLVER_TIMEOUT_MS = Number.parseInt(process.env.SOLVER_TIMEOUT_MS ?? "600000", 10);
const DEFAULT_BACKEND = process.env.DEFAULT_BACKEND ?? "transformers";
const LOAD_IN_4BIT = (process.env.LOAD_IN_4BIT ?? "1") !== "0";
const PRODUCTION_ADAPTER_FILE = path.join(
  PROJECT_ROOT,
  "outputs",
  "checkpoints",
  "PRODUCTION_ADAPTER.txt"
);
const FALLBACK_ADAPTER = path.join(
  PROJECT_ROOT,
  "outputs",
  "checkpoints",
  "qwen25math7b_BEST"
);

const JSON_START = "###JSON_START###";
const JSON_END = "###JSON_END###";

function readProductionAdapterFromFile() {
  try {
    const raw = fs.readFileSync(PRODUCTION_ADAPTER_FILE, "utf-8");
    const firstLine = String(raw)
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line.length > 0);
    if (firstLine) {
      return firstLine;
    }
  } catch (_) {
    // ignored on purpose; fallback is applied below.
  }
  return "";
}

function resolveDefaultAdapter() {
  if (process.env.ADAPTER_PATH && String(process.env.ADAPTER_PATH).trim()) {
    return String(process.env.ADAPTER_PATH).trim();
  }
  const fromFile = readProductionAdapterFromFile();
  if (fromFile) return fromFile;
  return FALLBACK_ADAPTER;
}

const DEFAULT_ADAPTER = resolveDefaultAdapter();

function extractPayload(stdoutText) {
  const start = stdoutText.indexOf(JSON_START);
  const end = stdoutText.lastIndexOf(JSON_END);
  if (start < 0 || end < 0 || end <= start) {
    throw new Error("Could not find solver JSON markers in stdout.");
  }
  const jsonText = stdoutText.slice(start + JSON_START.length, end).trim();
  return JSON.parse(jsonText);
}

function toArg(name, value, fallback) {
  const finalValue = value ?? fallback;
  if (finalValue === undefined || finalValue === null || finalValue === "") {
    return [];
  }
  return [name, String(finalValue)];
}

export function getSolverDefaults() {
  return {
    projectRoot: PROJECT_ROOT,
    pythonExe: PYTHON_EXE,
    solveScript: SOLVE_SCRIPT,
    backend: DEFAULT_BACKEND,
    modelName: DEFAULT_MODEL,
    adapterPath: DEFAULT_ADAPTER,
    timeoutMs: SOLVER_TIMEOUT_MS
  };
}

export function solveProblem(problemText, options = {}) {
  return new Promise((resolve, reject) => {
    const args = [
      SOLVE_SCRIPT,
      "--project-root",
      PROJECT_ROOT,
      "--problem-text",
      problemText,
      ...toArg("--backend", options.backend, DEFAULT_BACKEND),
      ...toArg("--model-name", options.modelName, DEFAULT_MODEL),
      ...toArg("--adapter-path", options.adapterPath, DEFAULT_ADAPTER),
      ...toArg("--max-new-tokens", options.maxNewTokens, 320),
      ...toArg("--temperature", options.temperature, 0.2),
      ...toArg("--n-plans", options.nPlans, 4),
      ...toArg("--m-drafts", options.mDrafts, 2),
      ...toArg("--refine-rounds", options.refineRounds, 1),
      ...toArg("--refine-top-k", options.refineTopK, 2),
      ...(options.loadIn4bit ?? LOAD_IN_4BIT ? ["--load-in-4bit"] : [])
    ];

    const child = spawn(PYTHON_EXE, args, {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PROJECT_ROOT,
        PYTHONUTF8: process.env.PYTHONUTF8 ?? "1",
        STRICT_NO_HALLUCINATION: process.env.STRICT_NO_HALLUCINATION ?? "1",
        HF_HUB_DISABLE_PROGRESS_BARS: process.env.HF_HUB_DISABLE_PROGRESS_BARS ?? "1",
        TQDM_DISABLE: process.env.TQDM_DISABLE ?? "1"
      },
      windowsHide: true
    });

    let stdout = "";
    let stderr = "";
    let finished = false;

    const timeout = setTimeout(() => {
      if (finished) return;
      finished = true;
      child.kill("SIGTERM");
      reject(
        new Error(
          `Solver process timed out after ${SOLVER_TIMEOUT_MS} ms.`
        )
      );
    }, SOLVER_TIMEOUT_MS);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (err) => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      reject(err);
    });

    child.on("close", (code) => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      if (code !== 0) {
        reject(
          new Error(
            `Solver process failed (code=${code}).\nSTDERR:\n${stderr}\nSTDOUT:\n${stdout}`
          )
        );
        return;
      }
      try {
        const payload = extractPayload(stdout);
        resolve(payload);
      } catch (err) {
        reject(
          new Error(
            `Solver returned non-JSON output.\nError: ${err.message}\nSTDERR:\n${stderr}\nSTDOUT:\n${stdout}`
          )
        );
      }
    });
  });
}
