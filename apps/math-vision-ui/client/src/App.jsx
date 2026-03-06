import { useEffect, useState, useRef } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const DEFAULT_ADAPTER =
  "E:\\\\IA_matematica\\\\outputs\\\\checkpoints\\\\qwen25math7b_BEST";

function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      {children}
      {hint ? <span className="hint">{hint}</span> : null}
    </label>
  );
}

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

export default function App() {
  const [file, setFile] = useState(null);
  const [problemText, setProblemText] = useState("");
  const [adapterPath, setAdapterPath] = useState(DEFAULT_ADAPTER);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [backend, setBackend] = useState("transformers");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

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
    setStatus("loading");
    try {
      const formData = new FormData();
      if (file) formData.append("image", file);
      formData.append("problemText", problemText);
      formData.append("backend", backend);
      formData.append("adapterPath", adapterPath);
      const res = await fetch(`${API_BASE}/api/solve-image`, {
        method: "POST",
        body: formData
      });
      const payload = await res.json();
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

  return (
    <div className="app-container">
      {/* Background decoration */}
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />

      <main className="page">
        {/* Header */}
        <header className="hero">
          <div className="hero-badge">
            <span className="badge-icon">🧮</span>
            <span className="badge-text">QLoRA 4-bit • 97% accuracy</span>
          </div>
          <h1>
            MathSolver<span className="accent">AI</span>
          </h1>
          <p className="subtitle">
            Resolva problemas de matemática de nível <strong>ITA</strong>, <strong>OBMEP</strong> e <strong>Unicamp</strong> com raciocínio passo a passo.
          </p>
          <div className="model-info">
            <span className="chip">Qwen2.5-Math-7B</span>
            <span className="chip">5.5GB VRAM</span>
            <span className="chip chip-green">29/30 benchmark</span>
          </div>
        </header>

        {/* Input form */}
        <form className="card glass" onSubmit={onSubmit}>
          <h2 className="card-title">
            <span className="card-icon">📝</span> Problema
          </h2>

          <div className="upload-zone" onClick={() => document.getElementById('file-input').click()}>
            {previewUrl ? (
              <img className="preview" src={previewUrl} alt="problem preview" />
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

          <Field label="Texto do problema" hint="Cole ou digite o problema aqui">
            <textarea
              rows={5}
              value={problemText}
              onChange={(e) => setProblemText(e.target.value)}
              placeholder="Se x + y = 10 e xy = 21, determine x³ + y³..."
            />
          </Field>

          <div className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
            <span>{showAdvanced ? '▼' : '►'} Configurações avançadas</span>
          </div>

          {showAdvanced && (
            <div className="advanced-panel">
              <div className="grid">
                <Field label="Backend">
                  <select value={backend} onChange={(e) => setBackend(e.target.value)}>
                    <option value="transformers">transformers</option>
                    <option value="heuristic">heuristic</option>
                  </select>
                </Field>
                <Field label="Adapter path">
                  <input value={adapterPath} onChange={(e) => setAdapterPath(e.target.value)} />
                </Field>
              </div>
            </div>
          )}

          <button type="submit" className="solve-btn" disabled={status === "loading"}>
            {status === "loading" ? (
              <span className="btn-loading">
                <span className="spinner" />
                Resolvendo... ({elapsed}s)
              </span>
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
              <div className="loading-bar">
                <div className="loading-bar-fill" />
              </div>
              <span className="loading-timer">{elapsed}s</span>
            </div>
          </section>
        )}

        {/* Results */}
        {response && (
          <section className="card glass results-card">
            <h2 className="card-title">
              <span className="card-icon">✅</span> Resolução
              <span className="result-time">{elapsed}s</span>
            </h2>

            {response.ocrText && (
              <div className="result-block">
                <h3>📷 Texto detectado (OCR)</h3>
                <pre>{response.ocrText}</pre>
              </div>
            )}

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
                  <div className="answer-meta">
                    Score: {best.score?.toFixed?.(3) ?? best.score} •
                    Verificado: {String(best.verifier?.passed)} •
                    Plano: {best.plan_name}
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
                    <h3>⚠️ Observações do verificador</h3>
                    <pre>{best.verifier.issues.join("\n")}</pre>
                  </div>
                )}
              </>
            )}
          </section>
        )}

        {/* Footer */}
        <footer className="footer">
          <p>
            <strong>MathSolver AI</strong> — Qwen2.5-Math-7B + QLoRA 4-bit
          </p>
          <p>
            <a href="https://github.com/Brun0Simoes/Modelo_math_Qwen2.5-LoRA" target="_blank" rel="noreferrer">
              GitHub
            </a>
            {" • "}
            97% benchmark accuracy • RTX 3070 8GB
          </p>
        </footer>
      </main>
    </div>
  );
}
