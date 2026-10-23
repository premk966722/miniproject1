import { ExternalLink, KeyRound, Server, ShieldCheck } from 'lucide-react';

export default function SettingsPanel() {
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
  return <section className="library-view settings-view" aria-labelledby="settings-title">
    <div className="library-heading"><div><span className="eyebrow"><i className="eyebrow-line" /> CONFIGURATION</span>
      <h1 id="settings-title">Workspace settings</h1><p>Frontend connection details for this local TAX-SHIELD session.</p></div></div>
    <article className="settings-card"><div className="settings-row"><div className="settings-icon"><Server size={17} /></div><div><b>Backend API</b><span>Requests are sent to the configured FastAPI service.</span></div></div>
      <code className="settings-value">{apiBase}</code>
      <a className="document-source-link" href={`${apiBase.replace(/\/$/, '')}/docs`} target="_blank" rel="noopener noreferrer">Open API documentation <ExternalLink size={13} /></a>
    </article>
    <article className="settings-card"><div className="settings-row"><div className="settings-icon"><KeyRound size={17} /></div><div><b>Provider credentials</b><span>LLM credentials are configured on the backend only.</span></div></div>
      <p className="settings-safe-note"><ShieldCheck size={14} /> No API keys or secrets are stored in the browser.</p>
    </article>
  </section>;
}
