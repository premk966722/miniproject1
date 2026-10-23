import { afterEach, describe, expect, it, vi } from 'vitest';
import { askTax, getDocuments, safeExternalUrl, uploadDocument } from './api';

afterEach(() => vi.restoreAllMocks());

describe('API service', () => {
  it('posts a typed chat request and validates the returned payload', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      answer: 'Evidence [c_1].', citations: [], retrieval_metadata: {}, status: 'answered',
    }), { status: 200, headers: { 'content-type': 'application/json' } }));
    await expect(askTax('Explain section 80C', 'FY 2025-26', 4)).resolves.toMatchObject({ answer: 'Evidence [c_1].' });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(init?.body))).toEqual({ question: 'Explain section 80C', tax_period: 'FY 2025-26', top_k: 4 });
    expect(new Headers(init?.headers).get('content-type')).toContain('application/json');
  });

  it('surfaces backend errors and malformed document lists', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Provider unavailable.' }), {
      status: 503, headers: { 'content-type': 'application/json' },
    }));
    await expect(askTax('Explain section 80C', '')).rejects.toThrow('Provider unavailable.');
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response(JSON.stringify({ documents: null }), {
      status: 200, headers: { 'content-type': 'application/json' },
    }));
    await expect(getDocuments()).rejects.toThrow('invalid document list');
  });

  it('uploads multipart data without setting a content type header', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ id: 'doc-1' }), {
      status: 200, headers: { 'content-type': 'application/json' },
    }));
    await uploadDocument(new File(['pdf'], 'tax.pdf', { type: 'application/pdf' }));
    const [, init] = fetchMock.mock.calls[0];
    expect(init?.body).toBeInstanceOf(FormData);
    expect(new Headers(init?.headers).has('content-type')).toBe(false);
  });

  it('allows only absolute HTTP(S) source links', () => {
    expect(safeExternalUrl('javascript:alert(1)')).toBeNull();
    expect(safeExternalUrl('https://incometax.gov.in/act.pdf')).toBe('https://incometax.gov.in/act.pdf');
    expect(safeExternalUrl('/relative/path')).toBeNull();
  });
});
