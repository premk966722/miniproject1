import { BadgeCheck, FileText, ShieldAlert } from 'lucide-react';

type Props = { status?: string; verified?: boolean };

export default function SourceBadge({ status, verified }: Props) {
  const government = status === 'government_origin_verified' || verified === true;
  const uploaded = status === 'user_uploaded_unverified';
  const label = government ? 'Government origin verified' : uploaded ? 'User upload · not verified' : 'Source not verified';
  const Icon = government ? BadgeCheck : uploaded ? FileText : ShieldAlert;
  return <span className={`source-badge ${government ? 'verified' : uploaded ? 'uploaded' : 'unverified'}`}>
    <Icon size={13} aria-hidden="true" />{label}
  </span>;
}
