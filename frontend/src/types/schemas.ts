/**
 * TypeScript types matching backend Pydantic schemas.
 * Enforces local-first evidence grounding, source separation, and finding rules.
 */

export type SourceType = 'contract' | 'legislation' | 'court_judgment';

export type FindingType = 
  | 'explicit_statement'
  | 'interpretation'
  | 'missing_clause'
  | 'potential_concern';

export type TriageStatus = 
  | 'sufficient_evidence'
  | 'insufficient_evidence'
  | 'escalation_required';

export interface EvidencePassage {
  evidence_id: string;
  document_id: string;
  source_type: SourceType;
  page_number: number;
  start_char: number;
  end_char: number;
  exact_quote: string;
  clause_number?: string;
  relevance_score?: number;
}

export interface Finding {
  finding_id: string;
  finding_type: FindingType;
  title: string;
  description: string;
  is_interpretation: boolean;
  exists_in_document: boolean;
  evidence: EvidencePassage[];
  definitive_illegality_claim: boolean;
  severity?: 'low' | 'medium' | 'high';
  reasoning?: string;
  escalation_recommended: boolean;
}

export interface OverviewClauseSummary {
  category: string;
  clause_number?: string;
  summary_text: string;
  evidence: EvidencePassage[];
  concerns: Finding[];
  is_missing: boolean;
}

export interface DocumentOverview {
  document_id: string;
  filename: string;
  page_count: number;
  key_clauses: OverviewClauseSummary[];
  potential_concerns: Finding[];
  missing_critical_clauses: string[];
  created_at: string;
}

export interface QuestionRequest {
  document_id: string;
  question: string;
}

export interface TriageResponse {
  status: TriageStatus;
  answer: string;
  evidence: EvidencePassage[];
  findings: Finding[];
  limitations: string[];
  insufficient_evidence_details?: string;
  escalation_required: boolean;
  escalation_message?: string;
}
