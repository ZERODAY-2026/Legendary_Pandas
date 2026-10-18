export type SourceType = "contract" | "legislation" | "court_judgment";

export interface Chunk {
  document_id: string;
  source_type: SourceType;
  page_start: number;
  page_end: number;
  clause_id: string | null;
  text: string;
  span_start: number;
  span_end: number;
}

export interface Evidence {
  source_id: string;
  source_type: "contract" | "legislation" | "court_judgment";
  page: number | null;
  section: string | null;
  quote: string;
}

export type FindingCategory =
  | "obligation"
  | "right"
  | "deadline"
  | "penalty"
  | "termination"
  | "ambiguity"
  | "missing_information"
  | "potential_legal_concern";

export interface Finding {
  title: string;
  category: FindingCategory;
  explicit_text: string | null;
  plain_language_explanation: string;
  interpretation_or_concern: string | null;
  evidence: Evidence[];
}

export interface DocumentOverview {
  document_id: string;
  filename: string;
  page_count: number;
  char_count: number;
  chunk_count: number;
  extracted_at: string;
}

export interface QueryResponse {
  answer: string;
  status: "answered_with_evidence" | "insufficient_evidence" | "escalation_required";
  findings: Finding[];
  evidence_used: Evidence[];
  disclaimer: string;
}
