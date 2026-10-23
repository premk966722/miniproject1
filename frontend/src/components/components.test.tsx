import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import '@testing-library/jest-dom/vitest';
import EvidenceCard from './EvidenceCard';
import HighlightedText from './HighlightedText';

afterEach(() => cleanup());

describe('evidence components', () => {
  it('highlights escaped terms as text without injecting markup', () => {
    const { container } = render(<HighlightedText text={'Section 80C (tax)'} query={'80C (tax)'} />);
    expect(container.querySelectorAll('mark')).toHaveLength(2);
    expect(container.querySelector('script')).toBeNull();
  });

  it('shows missing source metadata and expands evidence text accessibly', () => {
    render(<EvidenceCard index={1} query="deduction" citation={{ citation_id: 'c_123', document_id: 'doc-1',
      chunk_id: 'c_123', document_title: 'Uploaded tax note', page_number: null, section_reference: null,
      source_url: null, source_status: 'user_uploaded_unverified', evidence_excerpt: 'deduction '.repeat(40) }} />);
    expect(screen.getByText('User upload · not verified')).toBeInTheDocument();
    expect(screen.getByText('No source URL supplied')).toBeInTheDocument();
    const button = screen.getByRole('button', { name: 'Show full passage' });
    fireEvent.click(button);
    expect(screen.getByRole('button', { name: 'Show less' })).toHaveAttribute('aria-expanded', 'true');
  });
});
