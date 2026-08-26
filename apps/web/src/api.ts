import type {
  Answer,
  AuditEvent,
  Chunk,
  Document,
  Dossier,
  EvaluationRun,
  Extraction,
  SystemStatus,
  TokenResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
  }
}
async function request<T>(
  path: string,
  token?: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: { code?: string };
    } | null;
    throw new ApiError(body?.detail?.code ?? "request_failed", response.status);
  }
  return response.json() as Promise<T>;
}
export const api = {
  login: (username: string, password: string) =>
    request<TokenResponse>("/api/v1/auth/token", undefined, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username, password }),
    }),
  dossiers: (token: string) => request<Dossier[]>("/api/v1/dossiers", token),
  documents: (dossierId: string, token: string) =>
    request<Document[]>(`/api/v1/dossiers/${dossierId}/documents`, token),
  ask: (dossierId: string, question: string, token: string) =>
    request<Answer>(`/api/v1/dossiers/${dossierId}/ask`, token, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  content: (documentId: string, token: string) =>
    request<Chunk[]>(`/api/v1/documents/${documentId}/content`, token),
  extraction: (dossierId: string, token: string) =>
    request<Extraction[]>(`/api/v1/dossiers/${dossierId}/extraction`, token),
  audit: (token: string) =>
    request<AuditEvent[]>("/api/v1/audit-events", token),
  status: (token: string) => request<SystemStatus>("/api/v1/status", token),
  evaluations: (token: string) =>
    request<EvaluationRun[]>("/api/v1/evaluations", token),
  upload: (dossierId: string, file: File, token: string) => {
    const body = new FormData();
    body.append("file", file);
    body.append("is_synthetic", "true");
    return request<Document>(`/api/v1/dossiers/${dossierId}/documents`, token, {
      method: "POST",
      body,
    });
  },
};
