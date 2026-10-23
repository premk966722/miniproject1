import { BookOpenCheck, FileSearch2, Landmark, PanelRightClose, Scale, ShieldCheck } from 'lucide-react';
import type { Citation } from '../types';
import EvidenceCard from './EvidenceCard';

type Props = { citations: Citation[]; query: string; open: boolean; onClose: () => void };

export default function EvidencePanel({ citations, query, open, onClose }: Props) {
  if (!open) return null;
  return <aside className="evidence-rail" aria-label="Evidence trail">
    <div className="rail-heading"><div><span className="rail-eyebrow">TRANSPARENCY LAYER</span><h2>Evidence trail</h2></div>
      <button className="icon-button rail-close" onClick={onClose} title="Close evidence panel" aria-label="Close evidence panel"><PanelRightClose size={16} /></button>
    </div>
    {citations.length ? <><div className="rail-intro">Review the exact passages used to prepare the response. Provenance labels reflect stored source metadata.</div>
      <div className="rail-list">{citations.map((citation, index) => <EvidenceCard citation={citation} index={index + 1} query={query} key={citation.citation_id || citation.chunk_id || citation.id || index} />)}</div>
    </> : <><div className="rail-intro">TAX-SHIELD retrieves evidence before generating a response. No citation is shown when no passage was retrieved.</div>
      <div className="pipeline"><div className="pipeline-step"><div className="step-icon"><BookOpenCheck size={15} /></div><div><b>Question</b><span>Scoped by the selected tax period.</span></div></div>
        <div className="pipeline-connector" /><div className="pipeline-step"><div className="step-icon"><FileSearch2 size={15} /></div><div><b>Hybrid retrieval</b><span>BM25 and semantic results are combined using reciprocal rank fusion.</span></div></div>
        <div className="pipeline-connector" /><div className="pipeline-step"><div className="step-icon"><Scale size={15} /></div><div><b>Evidence validation</b><span>Citations are checked against retrieved chunk identifiers.</span></div></div>
        <div className="pipeline-connector" /><div className="pipeline-step"><div className="step-icon"><ShieldCheck size={15} /></div><div><b>Grounded response</b><span>Unsupported claims are treated as insufficient evidence.</span></div></div>
      </div><div className="rail-foot"><Landmark size={14} /><span>Uploaded PDFs are user-provided and are not automatically government-verified.</span></div>
    </>}
  </aside>;
}
