import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowUpRight, BadgeCheck, BookOpenCheck, CircleHelp, FileSearch2, Landmark,
  MessageSquareText, PanelRightClose, Plus, Scale, Settings2, Shield, ShieldCheck } from 'lucide-react';
import { askTax, getDocuments, getHealth } from './services/api';
import type { ConversationTurn, DocumentRecord, Health } from './types';
import ChatMessage from './components/ChatMessage';
import DocumentLibrary from './components/DocumentLibrary';
import DocumentUploader from './components/DocumentUploader';
import EvidencePanel from './components/EvidencePanel';
import QuestionInput from './components/QuestionInput';
import SettingsPanel from './components/SettingsPanel';

const prompts = [
  { icon: Scale, title: 'Which Act applies?', text: 'Which income tax Act applies to AY 2026-27?' },
  { icon: BookOpenCheck, title: 'Compare tax regimes', text: 'What evidence explains the old and new tax regimes?' },
  { icon: FileSearch2, title: 'Find filing guidance', text: 'What does the indexed guidance say about filing an income tax return?' },
];

const TURNS_STORAGE_KEY = 'taxshield.session.turns';
const TAX_PERIOD_STORAGE_KEY = 'taxshield.session.taxPeriod';

function loadStoredTurns(): ConversationTurn[] {
  try {
    const raw = window.sessionStorage.getItem(TURNS_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is ConversationTurn => Boolean(item) && typeof item === 'object'
      && typeof (item as ConversationTurn).id === 'string'
      && typeof (item as ConversationTurn).question === 'string'
      && (Boolean((item as ConversationTurn).response) || typeof (item as ConversationTurn).error === 'string'));
  } catch { return []; }
}

function loadStoredTaxPeriod(): string {
  try { return window.sessionStorage.getItem(TAX_PERIOD_STORAGE_KEY) || ''; } catch { return ''; }
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [taxPeriod, setTaxPeriod] = useState(loadStoredTaxPeriod);
  const [health, setHealth] = useState<Health>({ status: 'checking', documents: 0 });
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [turns, setTurns] = useState<ConversationTurn[]>(loadStoredTurns);
  const [loading, setLoading] = useState(false);
  const [screen, setScreen] = useState<'chat' | 'library' | 'settings'>('chat');
  const [showEvidence, setShowEvidence] = useState(true);
  const [documentLoading, setDocumentLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshDocuments = useCallback(async () => {
    setDocumentLoading(true);
    try { setDocuments(await getDocuments()); } finally { setDocumentLoading(false); }
  }, []);

  useEffect(() => {
    let active = true;
    void getHealth().then(value => { if (active) setHealth(value); });
    void getDocuments().then(value => { if (active) setDocuments(value); }).catch(() => {});
    const timer = window.setInterval(() => void getHealth().then(value => { if (active) setHealth(value); }), 30_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' }); }, [turns, loading]);

  useEffect(() => {
    try {
      window.sessionStorage.setItem(TURNS_STORAGE_KEY, JSON.stringify(turns.filter(turn => turn.response || turn.error)));
    } catch { /* storage may be unavailable (private mode, quota) */ }
  }, [turns]);

  useEffect(() => {
    try { window.sessionStorage.setItem(TAX_PERIOD_STORAGE_KEY, taxPeriod); } catch { /* storage may be unavailable */ }
  }, [taxPeriod]);

  async function submit(value = question) {
    const text = value.trim();
    if (text.length < 2 || loading) return;
    setQuestion(''); setLoading(true); setScreen('chat');
    const turn: ConversationTurn = { id: crypto.randomUUID(), question: text };
    setTurns(previous => [...previous, turn]);
    try {
      const response = await askTax(text, taxPeriod);
      setTurns(previous => previous.map(item => item.id === turn.id ? { ...item, response } : item));
    } catch (error) {
      setTurns(previous => previous.map(item => item.id === turn.id
        ? { ...item, error: error instanceof Error ? error.message : 'The service could not answer right now.' } : item));
    } finally { setLoading(false); }
  }

  const currentTurn = turns.at(-1);
  return <div className="app-shell">
    <aside className="sidebar" aria-label="Main navigation">
      <a className="brand" href="#top" onClick={event => { event.preventDefault(); setScreen('chat'); }}>
        <div className="brand-mark"><Shield size={19} strokeWidth={2.4} /></div><div><b>TAX-SHIELD</b><span>LEGAL RESEARCH ASSISTANT</span></div>
      </a>
      <button className="new-chat" onClick={() => { setTurns([]); setQuestion(''); setScreen('chat'); }}><Plus size={16} /> New conversation</button>
      <div className="nav-label">WORKSPACE</div>
      <button className={`nav-item ${screen === 'chat' ? 'active' : ''}`} onClick={() => setScreen('chat')}><MessageSquareText size={16} /> Research chat</button>
      <button className={`nav-item ${screen === 'library' ? 'active' : ''}`} onClick={() => setScreen('library')}><Landmark size={16} /> Document library <span className="nav-count">{documents.length}</span></button>
      <button className={`nav-item ${screen === 'settings' ? 'active' : ''}`} onClick={() => setScreen('settings')}><Settings2 size={16} /> API configuration</button>
      <div className="nav-label">THIS SESSION</div>
      <div className="session-history" aria-label="Conversation history">{turns.map((turn, index) => <button key={turn.id} onClick={() => {
        document.getElementById(`turn-${turn.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); setScreen('chat');
      }} className="history-item" title={turn.question}><span>{String(index + 1).padStart(2, '0')}</span>{turn.question}</button>)}</div>
      <div className="sidebar-spacer" />
      <div className="corpus-panel"><div className="corpus-top"><span className={`pulse ${health.status !== 'healthy' ? 'offline' : ''}`} /> API {health.status === 'healthy' ? 'CONNECTED' : health.status === 'checking' ? 'CONNECTING' : 'OFFLINE'}</div>
        <p>{health.uploaded_documents ?? documents.length} PDFs ingested<br />{health.uploaded_chunks ?? 0} evidence passages indexed</p>
        <div className="corpus-date"><BadgeCheck size={13} /> Sources retain provenance labels</div></div>
      <div className="sidebar-foot"><div className="avatar">TS</div><div><b>Local workspace</b><span>History stays in this tab</span></div></div>
    </aside>

    <main className="main-area" id="top">
      <header className="topbar"><div className="breadcrumb">TAX-SHIELD <span>/</span><b>{screen === 'chat' ? 'Research' : screen === 'library' ? 'Document library' : 'Configuration'}</b></div>
        <div className="top-actions"><button className="mobile-nav-button" onClick={() => setScreen(screen === 'library' ? 'chat' : 'library')} aria-label={screen === 'library' ? 'Open research chat' : 'Open document library'}><Landmark size={15} /><span>{screen === 'library' ? 'Research' : 'Library'}</span></button>
          <button className="mobile-nav-button" onClick={() => setScreen('settings')} aria-label="Open API configuration"><Settings2 size={15} /></button>
          <div className="status-pill" role="status"><span className={`status-dot ${health.status !== 'healthy' ? 'offline' : ''}`} />{health.status === 'healthy' ? 'API CONNECTED' : health.status === 'checking' ? 'CONNECTING' : 'API OFFLINE'}</div>
          <button className="icon-button" title="View API documentation" aria-label="Open API documentation" onClick={() => window.open(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'}/docs`, '_blank', 'noopener,noreferrer')}><CircleHelp size={17} /></button></div>
      </header>
      {screen === 'settings' ? <SettingsPanel /> : screen === 'library' ? <DocumentLibrary documents={documents} loading={documentLoading} onRefresh={() => void refreshDocuments()}
        onUploaded={async () => { await refreshDocuments(); setHealth(await getHealth()); }} /> :
      <div className={`content-grid ${!showEvidence ? 'wide' : ''}`}>
        <section className="conversation" aria-label="Tax research conversation">
          {!turns.length ? <div className="welcome">
            <div className="eyebrow"><span className="eyebrow-line" /> EVIDENCE-FIRST TAX RESEARCH</div>
            <h1>Research with<br /><em>the evidence in view.</em></h1>
            <p className="welcome-sub">Ask about Indian income tax and inspect the passages behind each response. Answers depend on documents currently indexed in this workspace.</p>
            <div className="welcome-note"><ShieldCheck size={16} /><span>Hybrid retrieval <i>·</i> Citation validation <i>·</i> Source provenance labels</span></div>
            <div className="prompt-label">SUGGESTED QUESTIONS</div>
            <div className="prompt-grid">{prompts.map(({ icon: Icon, title, text }) => <button className="prompt-card" key={title} onClick={() => { setQuestion(text); void submit(text); }}>
              <div className="prompt-icon"><Icon size={17} /></div><span><b>{title}</b><small>{text}</small></span><ArrowUpRight size={15} className="prompt-arrow" /></button>)}</div>
            <DocumentUploader onUploaded={async () => { await refreshDocuments(); setHealth(await getHealth()); }} />
            {documents.length > 0 && <div className="uploaded-list"><span className="upload-list-title">RECENT UPLOADS</span>{documents.slice(0, 3).map(document => <div className="uploaded-item" key={document.id}>
              <FileSearch2 size={13} /><span>{document.title || document.filename}</span><small>{document.page_count} pages · user-provided · not verified</small></div>)}</div>}
            <div className="corpus-nudge"><div className="nudge-icon"><Landmark size={16} /></div><div><b>Check the source metadata</b><span>Uploaded PDFs are not treated as government-verified unless their origin has been independently confirmed.</span></div><ArrowDown size={15} /></div>
          </div> : <div className="result-wrap">{turns.map(turn => <div id={`turn-${turn.id}`} key={turn.id}>
            <ChatMessage question={turn.question} response={turn.response} error={turn.error} pending={loading && turn.id === currentTurn?.id} />
          </div>)}<div ref={bottomRef} />
            {currentTurn?.error && <button className="follow-up" onClick={() => { setQuestion(currentTurn.question); void submit(currentTurn.question); }}><Plus size={15} /> Retry this question</button>}
          </div>}
          <QuestionInput value={question} taxPeriod={taxPeriod} loading={loading} onChange={setQuestion} onTaxPeriodChange={setTaxPeriod} onSubmit={() => void submit()} />
        </section>
        <EvidencePanel citations={currentTurn?.response?.citations || []} query={currentTurn?.question || ''} open={showEvidence} onClose={() => setShowEvidence(false)} />
        {!showEvidence && <button className="reopen-evidence" onClick={() => setShowEvidence(true)}><PanelRightClose size={15} /> Evidence panel</button>}
      </div>}
    </main>
  </div>;
}
