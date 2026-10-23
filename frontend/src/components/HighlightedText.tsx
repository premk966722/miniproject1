import { Fragment } from 'react';

type Props = { text: string; query: string };

export default function HighlightedText({ text, query }: Props) {
  const terms = [...new Set(query.match(/[\p{L}\p{N}][\p{L}\p{N}.-]*/gu) || [])]
    .filter(term => term.length > 1)
    .sort((a, b) => b.length - a.length);
  if (!terms.length || !text) return <>{text}</>;
  const escaped = terms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`(${escaped.join('|')})`, 'giu');
  return <>{text.split(pattern).map((part, index) => terms.some(term => term.toLocaleLowerCase() === part.toLocaleLowerCase())
    ? <mark key={`${index}-${part}`}>{part}</mark>
    : <Fragment key={index}>{part}</Fragment>)}</>;
}
