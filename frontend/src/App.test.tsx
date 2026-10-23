import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import App from './App';
import * as api from './services/api';

vi.mock('./services/api', () => ({
  askTax: vi.fn(), getDocuments: vi.fn(), getHealth: vi.fn(), uploadDocument: vi.fn(),
  safeExternalUrl: (value?: string | null) => value || null,
}));

const emptyAnswer = { question: 'What is Section 80C?', answer: 'Not enough evidence.', status: 'insufficient_evidence' as const,
  insufficient_evidence: true, citations: [], retrieval_metadata: { retrieval_method: 'hybrid_rrf', retrieved_count: 0 }, grounded: false };

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  vi.mocked(api.getHealth).mockResolvedValue({ status: 'healthy', documents: 0, uploaded_documents: 0, uploaded_chunks: 0 });
  vi.mocked(api.getDocuments).mockResolvedValue([]);
});
afterEach(() => cleanup());

describe('TAX-SHIELD chat and library', () => {
  it('shows loading, then the actual answer and insufficient-evidence state', async () => {
    let resolveAnswer!: (value: typeof emptyAnswer) => void;
    vi.mocked(api.askTax).mockReturnValue(new Promise(resolve => { resolveAnswer = resolve; }));
    render(<App />);
    fireEvent.change(screen.getByLabelText('Ask an Indian income tax question'), { target: { value: 'What is Section 80C?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send question' }));
    expect(screen.getByText('Searching indexed evidence')).toBeInTheDocument();
    resolveAnswer(emptyAnswer);
    await waitFor(() => expect(screen.getByText('Not enough evidence.')).toBeInTheDocument());
    expect(screen.getByText('Insufficient evidence')).toBeInTheDocument();
    expect(screen.getByText(/did not provide enough evidence/i)).toBeInTheDocument();
  });

  it('renders a real citation returned by the chat API', async () => {
    vi.mocked(api.askTax).mockResolvedValue({ ...emptyAnswer, answer: 'Section 80C is described here [c_real].',
      status: 'answered', insufficient_evidence: false, grounded: true, citations: [{ citation_id: 'c_real',
        chunk_id: 'c_real', document_id: 'doc-real', document_title: 'Income Tax Act', page_number: 42,
        section_reference: 'Section 80C', source_url: 'https://incometax.gov.in/act.pdf', source_status: 'government_origin_verified',
        evidence_excerpt: 'Eligible deduction under Section 80C.' }] });
    render(<App />);
    fireEvent.change(screen.getByLabelText('Ask an Indian income tax question'), { target: { value: 'What is Section 80C?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send question' }));
    expect(await screen.findByText('Section 80C is described here [c_real].')).toBeInTheDocument();
    expect(screen.getAllByText('Government origin verified')).toHaveLength(2);
    expect(screen.getAllByText((_content, element) => element?.tagName === 'P' && element.textContent?.startsWith('Eligible deduction under Section 80C.') === true)).toHaveLength(2);
  });

  it('keeps conversation history for the tab after a reload', async () => {
    vi.mocked(api.askTax).mockResolvedValue({ ...emptyAnswer, answer: 'Section 80C is described here [c_real].',
      status: 'answered', insufficient_evidence: false, grounded: true, citations: [{ citation_id: 'c_real',
        chunk_id: 'c_real', document_id: 'doc-real', document_title: 'Income Tax Act', page_number: 42,
        section_reference: 'Section 80C', source_url: 'https://incometax.gov.in/act.pdf', source_status: 'government_origin_verified',
        evidence_excerpt: 'Eligible deduction under Section 80C.' }] });
    const first = render(<App />);
    fireEvent.change(screen.getByLabelText('Ask an Indian income tax question'), { target: { value: 'What is Section 80C?' } });
    fireEvent.change(screen.getByLabelText(/tax period/i), { target: { value: 'FY 2025-26' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send question' }));
    expect(await screen.findByText('Section 80C is described here [c_real].')).toBeInTheDocument();
    first.unmount();

    render(<App />);
    expect(await screen.findByText('Section 80C is described here [c_real].')).toBeInTheDocument();
    expect(screen.getByTitle('What is Section 80C?')).toBeInTheDocument();
    expect(screen.getByLabelText(/tax period/i)).toHaveValue('FY 2025-26');
  });

  it('clears stored history when a new conversation starts', async () => {
    vi.mocked(api.askTax).mockResolvedValue({ ...emptyAnswer, answer: 'Stored answer.' });
    const first = render(<App />);
    fireEvent.change(screen.getByLabelText('Ask an Indian income tax question'), { target: { value: 'What is Section 80C?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send question' }));
    expect(await screen.findByText('Stored answer.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /New conversation/ }));
    expect(window.sessionStorage.getItem('taxshield.session.turns')).toBe('[]');
    first.unmount();
    render(<App />);
    expect(screen.queryByText('Stored answer.')).not.toBeInTheDocument();
  });

  it('displays service failures without inventing an answer', async () => {
    vi.mocked(api.askTax).mockRejectedValue(new Error('LLM provider is unavailable.'));
    render(<App />);
    fireEvent.change(screen.getByLabelText('Ask an Indian income tax question'), { target: { value: 'What is Section 80C?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send question' }));
    await waitFor(() => expect(screen.getByText('LLM provider is unavailable.')).toBeInTheDocument());
    expect(screen.queryByText('Evidence-grounded response')).not.toBeInTheDocument();
  });

  it('uploads a document and refreshes the visible document library', async () => {
    const pdf = new File(['pdf'], 'act.pdf', { type: 'application/pdf' });
    vi.mocked(api.uploadDocument).mockResolvedValue({ id: 'doc-1', title: 'Tax Act', filename: 'act.pdf', file_size: 3,
      upload_timestamp: 'now', page_count: 3, chunk_count: 6, chunks_created: 6, ingestion_status: 'completed',
      source_url: null, source_type: 'user_upload', official_domain_verified: false, document_type: null, tax_period: null });
    vi.mocked(api.getDocuments).mockResolvedValueOnce([]).mockResolvedValue([{ id: 'doc-1', title: 'Tax Act', filename: 'act.pdf',
      file_size: 3, upload_timestamp: 'now', page_count: 3, chunk_count: 6, ingestion_status: 'completed', source_url: null,
      source_type: 'user_upload', official_domain_verified: false, document_type: null, tax_period: null }]);
    render(<App />);
    fireEvent.change(screen.getByLabelText('Choose PDF document'), { target: { files: [pdf] } });
    await waitFor(() => expect(screen.getByText(/Document ID: doc-1/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /Document library/ }));
    expect(await screen.findByText('Tax Act')).toBeInTheDocument();
    expect(screen.getByText('6')).toBeInTheDocument();
  });
});
