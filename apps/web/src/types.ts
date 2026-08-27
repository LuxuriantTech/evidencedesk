export type Role = "admin" | "analyst" | "reader";
export interface User {
  id: string;
  username: string;
  role: Role;
}
export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: User;
}
export interface Dossier {
  id: string;
  name: string;
  description: string;
  is_synthetic: boolean;
}
export interface Document {
  id: string;
  filename: string;
  media_type: string;
  status: string;
  task_id: string | null;
  error_code: string | null;
}
export interface Citation {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page: number;
  section: string | null;
  excerpt: string;
}
export interface PassageCandidateAssessment {
  answerable: boolean;
  answer: string | null;
  confidence: number;
  supporting_document: string | null;
  supporting_page: number | null;
  supporting_excerpt: string | null;
  ambiguity_reason: string | null;
  extracted_fields: Record<string, string | string[]>;
  supporting_chunk_id: string;
}
export interface Answer {
  status: "answered" | "partially_supported" | "abstained" | "ambiguous";
  answerable: boolean;
  answer: string;
  confidence: number;
  mode: string;
  supporting_document: string | null;
  supporting_page: number | null;
  supporting_excerpt: string | null;
  ambiguity_reason: string | null;
  extracted_fields: Record<string, string | string[]>;
  candidate_assessments: PassageCandidateAssessment[];
  citations: Citation[];
  correlation_id: string;
}
export interface Chunk {
  id: string;
  page: number;
  section: string | null;
  ordinal: number;
  text: string;
}
export interface Extraction {
  document_id: string;
  document_name: string;
  schema_version: string;
  payload: Record<string, unknown>;
}
export interface AuditEvent {
  id: string;
  actor_id: string | null;
  action: string;
  document_id: string | null;
  result: string;
  correlation_id: string;
  created_at: string;
}
export interface SystemStatus {
  mode: string;
  public_demo_mode: boolean;
  documents: Record<string, number>;
  failed_tasks: number;
}
export interface EvaluationRun {
  id: string;
  dataset_version: string;
  parameters_version: string;
  split: string;
  mode: string;
  metrics: Record<string, unknown>;
  verdict: string;
  created_at: string;
}
