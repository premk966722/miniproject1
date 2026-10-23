export type Health = {
  status: 'healthy' | 'offline' | string;
  application?: string;
  seed_sources?: number;
  uploaded_documents?: number;
  uploaded_chunks?: number;
  documents: number;
};

export type DocumentRecord = {
  id: string;
  title: string;
  filename: string;
  file_size: number;
  upload_timestamp: string;
  page_count: number;
  chunk_count: number;
  ingestion_status: string;
  source_url: string | null;
  source_type: string;
  official_domain_verified: boolean;
  document_type: string | null;
  tax_period: string | null;
};

export type Citation = {
  citation_id: string;
  document_id: string;
  chunk_id: string;
  document_title: string;
  page_number: number | null;
  section_reference: string | null;
  source_url: string | null;
  source_status: string;
  evidence_excerpt: string;
  id?: string;
  title?: string | null;
  authority?: string | null;
  url?: string | null;
  reference?: string | null;
  excerpt?: string;
  page_start?: number | null;
  section?: string | null;
  source_type?: string;
  official_domain_verified?: boolean;
};

export type ChatResponse = {
  question: string;
  answer: string;
  status: 'answered' | 'insufficient_evidence' | 'needs_clarification' | 'citation_warning';
  insufficient_evidence: boolean;
  citations: Citation[];
  retrieval_metadata: { retrieval_method?: string; retrieved_count?: number; [key: string]: unknown };
  grounded?: boolean;
  confidence?: number;
  notice?: string;
  citation_warning?: string | null;
};

export type ConversationTurn = { id: string; question: string; response?: ChatResponse; error?: string };
