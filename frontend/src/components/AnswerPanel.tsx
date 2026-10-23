import { AlertTriangle, BadgeCheck, ShieldCheck, Sparkles } from 'lucide-react';
import type { ChatResponse } from '../types';

type Props = { response: ChatResponse };

export default function AnswerPanel({ response }: Props) {
  const statusText = response.insufficient_evidence ? 'Insufficient evidence'
    : response.status === 'needs_clarification' ? 'More detail needed'
      : response.status === 'citation_warning' ? 'Citation needs review'
        : response.citations.length ? 'Evidence-grounded response' : 'TAX-SHIELD assistant';
  const supporting = response.citations.length
    ? `${response.citations.length} supporting ${response.citations.length === 1 ? 'passage' : 'passages'}`
    : response.insufficient_evidence ? 'No matching source passages' : 'Conversational reply';
  const Icon = response.insufficient_evidence ? AlertTriangle : response.grounded ? BadgeCheck : Sparkles;
  return <section className={`answer-card answer-status-${response.status}`} aria-labelledby="answer-heading">
    <div className="answer-head"><div className="answer-icon"><Icon size={16} /></div>
      <div><b id="answer-heading">{statusText}</b><span>{supporting}</span></div>
      <div className={`confidence ${response.insufficient_evidence ? 'low' : ''}`}><span />{response.retrieval_metadata.retrieval_method || 'hybrid retrieval'}</div>
    </div>
    {response.insufficient_evidence && <div className="evidence-alert" role="status"><AlertTriangle size={15} /> The indexed documents did not provide enough evidence to answer this question reliably.</div>}
    <div className="answer-text">{response.answer.split('\n').map((line, index) => <p key={index}>{line || <br />}</p>)}</div>
    {response.citation_warning && <div className="citation-warning" role="alert"><AlertTriangle size={14} />{response.citation_warning}</div>}
    {response.notice && <div className="answer-notice"><ShieldCheck size={14} /><span>{response.notice}</span></div>}
  </section>;
}
