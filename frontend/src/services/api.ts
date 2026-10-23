import type { ChatResponse, DocumentRecord, Health } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001').replace(/\/$/, '');

async function requestJson<T>(path: string, init?: RequestInit, timeoutMs = 30_000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
    const contentType = response.headers.get('content-type') || '';
    const payload: unknown = contentType.includes('application/json') ? await response.json() : null;
    if (!response.ok) {
      const detail = payload && typeof payload === 'object' && 'detail' in payload
        ? String((payload as { detail: unknown }).detail)
        : `The API returned HTTP ${response.status}.`;
      throw new Error(detail);
    }
    if (!payload || typeof payload !== 'object') throw new Error('The API returned an invalid response.');
    return payload as T;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('The request timed out. Please try again.');
    if (error instanceof TypeError) throw new Error('Could not connect to TAX-SHIELD. Check that the backend is running.');
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function getHealth(): Promise<Health> {
  try {
    const data = await requestJson<Omit<Health, 'documents'>>('/api/health');
    return { ...data, documents: (data.seed_sources || 0) + (data.uploaded_documents || 0) };
  } catch {
    return { status: 'offline', documents: 0 };
  }
}

export async function getDocuments(): Promise<DocumentRecord[]> {
  const data = await requestJson<{ documents: DocumentRecord[] }>('/api/documents');
  if (!Array.isArray(data.documents)) throw new Error('The API returned an invalid document list.');
  return data.documents;
}

export async function uploadDocument(file: File): Promise<DocumentRecord & { chunks_created: number }> {
  if (file.type !== 'application/pdf' || !file.name.toLowerCase().endsWith('.pdf')) {
    throw new Error('Choose a PDF file with the application/pdf file type.');
  }
  const form = new FormData();
  form.append('file', file);
  return requestJson('/api/documents/upload', { method: 'POST', body: form }, 120_000);
}

export async function askTax(question: string, taxPeriod: string, topK = 5): Promise<ChatResponse> {
  const result = await requestJson<ChatResponse>('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, tax_period: taxPeriod || null, top_k: topK }),
  }, 60_000);
  if (typeof result.answer !== 'string' || !Array.isArray(result.citations) || !result.retrieval_metadata) {
    throw new Error('The API returned an invalid answer payload.');
  }
  return result;
}

export function safeExternalUrl(value?: string | null): string | null {
  if (!value) return null;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'https:' || parsed.protocol === 'http:' ? parsed.href : null;
  } catch {
    return null;
  }
}
