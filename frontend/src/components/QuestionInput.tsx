import { ArrowUpRight, Command, LoaderCircle, Sparkles } from 'lucide-react';

type Props = { value: string; taxPeriod: string; loading: boolean; onChange: (value: string) => void;
  onTaxPeriodChange: (value: string) => void; onSubmit: () => void };

export default function QuestionInput({ value, taxPeriod, loading, onChange, onTaxPeriodChange, onSubmit }: Props) {
  return <div className="composer-wrap">
    <form className="composer" onSubmit={event => { event.preventDefault(); onSubmit(); }}>
      <div className="composer-top"><span className="composer-mode"><Sparkles size={14} /> TAX-SHIELD <i>·</i> GROUNDED MODE</span>
        <label htmlFor="tax-period">Tax period <select id="tax-period" value={taxPeriod} onChange={event => onTaxPeriodChange(event.target.value)}>
          <option value="">Not specified</option><option>FY 2025-26</option><option>AY 2026-27</option><option>FY 2024-25</option><option>AY 2025-26</option>
        </select></label>
      </div>
      <label className="sr-only" htmlFor="question-input">Ask an Indian income tax question</label>
      <textarea id="question-input" value={value} onChange={event => onChange(event.target.value)}
        onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); onSubmit(); } }}
        placeholder="Ask about a tax provision, return, or deduction…" maxLength={1200} rows={2} />
      <div className="composer-bottom"><span><Command size={12} /> Enter to send <i>·</i> Shift + Enter for a new line</span>
        <button type="submit" disabled={loading || value.trim().length < 2} aria-label="Send question">
          {loading ? <LoaderCircle size={16} className="spin" /> : <ArrowUpRight size={17} />}
        </button>
      </div>
    </form><div className="legal-line"><span aria-hidden="true">◆</span> Educational research aid. Verify cited guidance for your circumstances.</div>
  </div>;
}
