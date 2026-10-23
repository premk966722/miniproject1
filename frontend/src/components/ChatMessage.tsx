import { AlertCircle, LoaderCircle, UserRound } from 'lucide-react';
import type { ChatResponse } from '../types';
import AnswerPanel from './AnswerPanel';
import EvidenceCard from './EvidenceCard';

type Props = { question: string; response?: ChatResponse; pending?: boolean; error?: string };

export default function ChatMessage({ question, response, pending, error }: Props) {
  return <article className="chat-turn">
    <div className="thread-question"><div className="avatar user-avatar"><UserRound size={14} /></div><div>
      <span className="thread-kicker">YOUR QUESTION</span><h2>{question}</h2></div></div>
    {pending && <div className="loading-answer" role="status" aria-live="polite"><div className="loading-icon"><LoaderCircle size={20} /></div>
      <div><b>Searching indexed evidence</b><span>Combining relevant passages and preparing the answer…</span><div className="loading-track"><i /></div></div></div>}
    {error && <div className="error-box" role="alert"><AlertCircle size={16} />{error}</div>}
    {response && <><AnswerPanel response={response} />
      <div className="mobile-source-title">EVIDENCE <span>{response.citations.length}</span></div>
      <div className="source-list">{response.citations.map((citation, index) => <EvidenceCard key={citation.citation_id || citation.chunk_id || citation.id || index}
        citation={citation} index={index + 1} query={question} />)}</div>
    </>}
  </article>;
}
