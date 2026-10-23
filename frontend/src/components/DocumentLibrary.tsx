import { ArrowUpRight, FileSearch2, LoaderCircle, RefreshCw, Trash2 } from 'lucide-react';
import type { DocumentRecord } from '../types';
import SourceBadge from './SourceBadge';
import { safeExternalUrl } from '../services/api';
import DocumentUploader from './DocumentUploader';

type Props = { documents: DocumentRecord[]; loading: boolean; onRefresh: () => void;
  onUploaded: (document: DocumentRecord) => Promise<void> | void };

export default function DocumentLibrary({ documents, loading, onRefresh, onUploaded }: Props) {
  return <section className="library-view" aria-labelledby="library-title">
    <div className="library-heading"><div><span className="eyebrow"><i className="eyebrow-line" /> DOCUMENT COLLECTION</span>
      <h1 id="library-title">Your research library</h1><p>PDFs available to the evidence retrieval pipeline.</p></div>
      <button className="secondary-button" onClick={onRefresh} disabled={loading}><RefreshCw size={15} className={loading ? 'spin' : ''} /> Refresh</button>
    </div>
    <DocumentUploader onUploaded={onUploaded} />
    {loading ? <div className="library-empty"><LoaderCircle className="spin" /> Loading documents…</div>
      : documents.length ? <div className="document-grid">{documents.map(document => <article className="document-card" key={document.id}>
        <div className="document-card-icon"><FileSearch2 size={18} /></div><div className="document-card-title"><b>{document.title || document.filename}</b><span>{document.filename}</span></div>
        <SourceBadge status={document.source_type === 'user_upload' ? 'user_uploaded_unverified' : undefined} verified={document.official_domain_verified} />
        <dl><div><dt>Pages</dt><dd>{document.page_count}</dd></div><div><dt>Chunks</dt><dd>{document.chunk_count}</dd></div><div><dt>Status</dt><dd>{document.ingestion_status}</dd></div>
          {document.tax_period && <div><dt>Tax period</dt><dd>{document.tax_period}</dd></div>}</dl>
        {safeExternalUrl(document.source_url) && <a className="document-source-link" href={safeExternalUrl(document.source_url)!} target="_blank" rel="noopener noreferrer">Source URL <ArrowUpRight size={12} /></a>}
        <small className="document-id">ID {document.id}</small>
      </article>)}</div> : <div className="library-empty"><FileSearch2 size={22} /><b>No ingested PDFs yet</b><span>Upload a searchable tax document to add evidence to the collection.</span></div>}
    <p className="library-note"><Trash2 size={13} /> Uploaded files remain local to this TAX-SHIELD backend and are marked as unverified user uploads.</p>
  </section>;
}
