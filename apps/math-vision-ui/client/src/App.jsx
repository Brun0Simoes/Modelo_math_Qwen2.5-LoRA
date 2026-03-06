import { useEffect, useState, useRef, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

/* ── Typewriter effect for reasoning output ── */
function TypewriterText({ text }) {
  const [displayed, setDisplayed] = useState("");
  const idx = useRef(0);
  useEffect(() => {
    setDisplayed("");
    idx.current = 0;
    if (!text) return;
    const interval = setInterval(() => {
      idx.current++;
      setDisplayed(text.slice(0, idx.current));
      if (idx.current >= text.length) clearInterval(interval);
    }, 8);
    return () => clearInterval(interval);
  }, [text]);
  return <>{displayed}</>;
}

/* ── Loading Modal ── */
function LoadingModal({ modelStatus }) {
  if (!modelStatus || modelStatus.status === "ready") return null;

  const isError = modelStatus.status === "error";
  const progress = modelStatus.progress || 0;

  return (
    <div className="modal-overlay">
      <div className="modal-card glass">
        <div className="modal-icon">
          {isError ? "❌" : "🧠"}
        </div>
        <h2 className="modal-title">
          {isError ? "Erro ao carregar modelo" : "Carregando modelo..."}
        </h2>
        {!isError && (
          <>
            <div className="modal-progress-bar">
              <div
                className="modal-progress-fill"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="modal-percent">{progress}%</p>
            <p className="modal-step">{modelStatus.step || "Inicializando..."}</p>
            <div className="modal-details">
              <span>Qwen2.5-Math-7B</span>
              <span className="modal-dot">•</span>
              <span>QLoRA 4-bit NF4</span>
              <span className="modal-dot">•</span>
              <span>7.7B parâmetros</span>
            </div>
          </>
        )}
        {isError && (
          <p className="modal-error">{modelStatus.error}</p>
        )}
      </div>
    </div>
  );
}

/* ── Main App ── */
export default function App() {
  const [file, setFile] = useState(null);
  const [problemText, setProblemText] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [modelStatus, setModelStatus] = useState({ status: "loading", progress: 0, step: "Conectando..." });
  const timerRef = useRef(null);

  /* Poll /api/status for model loading progress */
  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setModelStatus(data);
        return data.status;
      }
    } catch {
      // Server not ready yet
    }
    return "loading";
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        const st = await pollStatus();
        if (st === "ready" || st === "error") break;
        await new Promise(r => setTimeout(r, 2000));
      }
    };
    poll();
    return () => { cancelled = true; };
  }, [pollStatus]);

  useEffect(() => {
    if (!file) { setPreviewUrl(""); return; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (status === "loading") {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [status]);

  async function onSubmit(event) {
    event.preventDefault();
    setError("");
    setResponse(null);
    if (!file && !problemText.trim()) {
      setError("Envie uma imagem ou digite o problema.");
      return;
    }
    if (modelStatus.status !== "ready") {
      setError(`Modelo ainda carregando (${modelStatus.progress}%). Aguarde.`);
      return;
    }
    setStatus("loading");
    try {
      let res;
      if (file) {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("problemText", problemText);
        res = await fetch(`${API_BASE}/api/solve-image`, {
          method: "POST",
          body: formData
        });
      } else {
        res = await fetch(`${API_BASE}/api/solve-image`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ problemText })
        });
      }
      const text = await res.text();
      let payload;
      try {
        payload = JSON.parse(text);
      } catch {
        throw new Error(text || `Servidor retornou resposta vazia (HTTP ${res.status}).`);
      }
      if (!res.ok || !payload.ok) throw new Error(payload.error || `HTTP ${res.status}`);
      if (!problemText.trim() && payload.ocrText) setProblemText(payload.ocrText);
      setResponse(payload);
      setStatus("done");
    } catch (err) {
      setError(err.message || String(err));
      setStatus("error");
    }
  }

  const best = response?.run?.best_candidate;
  const isModelReady = modelStatus.status === "ready";

  return (
    <div className="app-container">
      {/* Background glows */}
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />

      {/* Model Loading Modal */}
      <LoadingModal modelStatus={modelStatus} />

      <main className="page">
        {/* Header */}
        <header className="hero">
          <div className="hero-badge">
            <span className="badge-icon">🧮</span>
            <span className="badge-text">QLoRA 4-bit • 97% accuracy</span>
          </div>
          <h1>MathSolver<span className="accent">AI</span></h1>
          <p className="subtitle">
            Resolva problemas de matemática de nível <strong>ITA</strong>, <strong>OBMEP</strong> e <strong>Unicamp</strong> com raciocínio passo a passo.
          </p>
          <div className="model-info">
            <span className="chip">Qwen2.5-Math-7B</span>
            <span className="chip">5.5GB VRAM</span>
            <span className="chip chip-green">29/30 benchmark</span>
            <span className={`chip ${isModelReady ? 'chip-green' : 'chip-yellow'}`}>
              {isModelReady ? '● Online' : `◌ Carregando ${modelStatus.progress}%`}
            </span>
          </div>
        </header>

        {/* Form */}
        <form className="card glass" onSubmit={onSubmit}>
          <h2 className="card-title">
            <span className="card-icon">📝</span> Problema
          </h2>

          <div className="upload-zone" onClick={() => document.getElementById('file-input').click()}>
            {previewUrl ? (
              <img className="preview" src={previewUrl} alt="preview" />
            ) : (
              <div className="upload-placeholder">
                <span className="upload-icon">📷</span>
                <span>Clique para enviar imagem do problema</span>
                <span className="upload-hint">PNG, JPG até 8MB</span>
              </div>
            )}
            <input
              id="file-input"
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <div className="field">
            <span className="label">Texto do problema</span>
            <textarea
              rows={5}
              value={problemText}
              onChange={(e) => setProblemText(e.target.value)}
              placeholder="Se x + y = 10 e xy = 21, determine x³ + y³..."
            />
            <span className="hint">Cole ou digite o problema aqui</span>
          </div>

          <div className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
            <span>{showAdvanced ? '▼' : '►'} Configurações avançadas</span>
          </div>

          {showAdvanced && (
            <div className="advanced-panel">
              <div className="field">
                <span className="label">Adapter checkpoint</span>
                <input
                  defaultValue="E:\\IA_matematica\\outputs\\checkpoints\\qwen25math7b_BEST"
                  readOnly
                />
              </div>
            </div>
          )}

          <button
            type="submit"
            className="solve-btn"
            disabled={status === "loading" || !isModelReady}
          >
            {status === "loading" ? (
              <span className="btn-loading">
                <span className="spinner" />
                Resolvendo... ({elapsed}s)
              </span>
            ) : !isModelReady ? (
              <span>⏳ Modelo carregando ({modelStatus.progress}%)</span>
            ) : (
              <span>🚀 Resolver Problema</span>
            )}
          </button>
        </form>

        {/* Error */}
        {error && (
          <section className="card glass error-card">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
          </section>
        )}

        {/* Loading animation */}
        {status === "loading" && (
          <section className="card glass loading-card">
            <div className="loading-brain">
              <div className="pulse-ring" />
              <span className="brain-emoji">🧠</span>
            </div>
            <div className="loading-text">
              <h3>Modelo pensando...</h3>
              <p>O Qwen2.5-Math-7B está analisando o problema passo a passo</p>
              <div className="loading-bar"><div className="loading-bar-fill" /></div>
              <span className="loading-timer">{elapsed}s</span>
            </div>
          </section>
        )}

        {/* Results */}
        {response && (
          <section className="card glass results-card">
            <h2 className="card-title">
              <span className="card-icon">✅</span> Resolução
              <span className="result-time">{response.elapsed ?? elapsed}s • {response.tokens ?? '?'} tokens</span>
            </h2>

            <div className="result-block">
              <h3>📖 Problema analisado</h3>
              <pre>{response.usedText}</pre>
            </div>

            {best && (
              <>
                <div className="answer-card">
                  <div className="answer-label">Resposta Final</div>
                  <div className="answer-value">
                    {best.final_answer || "(não encontrada)"}
                  </div>
                </div>

                <div className="result-block">
                  <h3>📝 Raciocínio completo</h3>
                  <pre className="draft-text">
                    <TypewriterText text={best.draft} />
                  </pre>
                </div>

                {best.verifier?.issues?.length > 0 && (
                  <div className="result-block">
                    <h3>⚠️ Observações</h3>
                    <pre>{best.verifier.issues.join("\n")}</pre>
                  </div>
                )}
              </>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
