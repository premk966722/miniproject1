import { useRef, useState } from 'react';
import { AlertCircle, LoaderCircle, UploadCloud } from 'lucide-react';
import { uploadDocument } from '../services/api';
import type { DocumentRecord } from '../types';

type Props = { onUploaded: (document: DocumentRecord) => Promise<void> | void };

export default function DocumentUploader({ onUploaded }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);

  async function handleFile(file?: File) {
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) { setMessage('PDF uploads are limited to 20 MB.'); setIsError(true); return; }
    setBusy(true); setMessage(`Processing ${file.name}…`); setIsError(false);
    try {
      const result = await uploadDocument(file);
      await onUploaded(result);
      setMessage(`Indexed ${result.chunks_created} chunks from ${result.page_count} pages. Document ID: ${result.id}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'The PDF could not be uploaded.'); setIsError(true);
    } finally { setBusy(false); }
  }

  return <div className="upload-panel">
    <div className="upload-panel-icon"><UploadCloud size={17} /></div>
    <div className="upload-panel-copy"><b>Add a tax document to your research set</b>
      <span>Text-searchable PDF, up to 20 MB. Uploading does not verify government provenance.</span>
      {message && <small role={isError ? 'alert' : 'status'} className={isError ? 'upload-error' : 'upload-success'}>
        {busy && <LoaderCircle size={12} className="spin" />}{isError && <AlertCircle size={12} />}{message}</small>}
    </div>
    <button type="button" className="upload-action" onClick={() => input.current?.click()} disabled={busy}>
      {busy ? 'PROCESSING' : 'CHOOSE PDF'}
    </button>
    <input ref={input} type="file" accept="application/pdf,.pdf" className="visually-hidden" aria-label="Choose PDF document"
      onChange={event => { void handleFile(event.target.files?.[0]); event.currentTarget.value = ''; }} disabled={busy} />
  </div>;
}
