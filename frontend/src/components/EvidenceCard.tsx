import { useState } from 'react';
import { ArrowUpRight, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';
import type { Citation } from '../types';
import { safeExternalUrl } from '../services/api';
import HighlightedText from './HighlightedText';
import SourceBadge from './SourceBadge';

type Props = { citation: Citation; index: number; query: string };

export default function EvidenceCard({ citation, index, query }: Props) {
  const [expanded, setExpanded] = useState(false);
  const title = citation.document_title || citation.title || 'Untitled source';
  const excerpt = citation.evidence_excerpt || citation.excerpt || '';
  const url = safeExternalUrl(citation.source_url || citation.url);
  const page = citation.page_number ?? citation.page_start;
  const section = citation.section_reference || citation.section || citation.reference;
  return <article className="source-card evidence-card">
    <div className="source-number">{String(index).padStart(2, '0')}</div>
    <div className="source-copy">
      <div className="evidence-card-top"><SourceBadge status={citation.source_status} verified={citation.official_domain_verified} />
        <code title="Citation identifier">[{citation.citation_id || citation.chunk_id || citation.id || 'citation'}]</code></div>
      <h3>{title}</h3>
      <div className="source-ref"><BookOpen size={13} />{section || 'Provision not identified'}{page ? ` · page ${page}` : ''}</div>
      <p className={`evidence-excerpt ${expanded ? 'expanded' : ''}`}><HighlightedText text={excerpt} query={query} /></p>
      <div className="evidence-actions">
        {excerpt.length > 260 && <button type="button" className="text-action" onClick={() => setExpanded(value => !value)} aria-expanded={expanded}>
          {expanded ? <>Show less <ChevronUp size={13} /></> : <>Show full passage <ChevronDown size={13} /></>}
        </button>}
        {url ? <a href={url} target="_blank" rel="noopener noreferrer">Open source <ArrowUpRight size={13} /></a> : <span className="missing-source">No source URL supplied</span>}
      </div>
    </div>
  </article>;
}
