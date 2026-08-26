import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "./api";
import type {
  Answer,
  AuditEvent,
  Chunk,
  Citation,
  Document,
  Dossier,
  EvaluationRun,
  Extraction,
  SystemStatus,
  TokenResponse,
} from "./types";

const FIELD_LABELS: Record<string, string> = {
  organization_name: "Organisation",
  document_type: "Type de document",
  effective_date: "Date d’effet",
  renewal_date: "Date de renouvellement",
  important_amounts: "Montants importants",
  obligations: "Obligations",
  responsible_people: "Responsables mentionnés",
  risks: "Risques et incohérences",
};

const STATUS_LABELS: Record<string, string> = {
  queued: "En attente",
  processing: "Traitement",
  completed: "Terminé",
  failed: "Échec",
  deleted: "Supprimé",
};

interface EvidenceField {
  value: unknown;
  citations: Citation[];
}

function apiMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return "Une erreur inattendue est survenue.";
  if (error.status === 401) return "Identifiant ou mot de passe invalide.";
  const messages: Record<string, string> = {
    synthetic_attestation_required: "Confirmez que le document est entièrement synthétique.",
    public_demo_file_not_approved:
      "Démo publique : utilisez l'un des fichiers synthétiques versionnés fournis.",
    empty_file: "Le fichier est vide.",
    file_too_large: "Le fichier dépasse la limite de 10 Mio.",
    request_too_large: "La requête dépasse la limite de 11 Mio.",
    unsupported_type: "Format refusé. Utilisez un PDF texte, TXT ou Markdown.",
    invalid_signature: "Le contenu du fichier ne correspond pas à son extension.",
    invalid_text: "Le fichier texte est invalide ou illisible en UTF-8.",
    queue_unavailable: "La file de traitement est momentanément indisponible.",
    forbidden: "Votre rôle ne permet pas cette action.",
  };
  return messages[error.message] ?? "Le service est indisponible. Réessayez dans un instant.";
}

function isEvidenceField(value: unknown): value is EvidenceField {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return "value" in candidate && Array.isArray(candidate.citations);
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Non renseigné";
  if (Array.isArray(value)) return value.length ? value.join(" · ") : "Aucun élément";
  return String(value);
}

function percentage(value: unknown): string {
  if (typeof value !== "number") return "Non mesuré";
  return `${(value * 100).toFixed(1).replace(".", ",")} %`;
}

function numberMetric(value: unknown, suffix: string): string {
  return typeof value === "number" ? `${value.toFixed(3)} ${suffix}` : "Non mesuré";
}

function CitationButton({
  citation,
  onOpen,
}: {
  citation: Citation;
  onOpen: (citation: Citation) => void;
}) {
  return (
    <button
      className="citation"
      onClick={() => onOpen(citation)}
      aria-label={`Voir la source ${citation.document_name}, page ${citation.page}`}
      type="button"
    >
      {citation.document_name} · page {citation.page}
    </button>
  );
}

function Login({ onLogin }: { onLogin: (session: TokenResponse) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      onLogin(await api.login(username, password));
    } catch (caught) {
      setError(apiMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  const fillDemoAccount = () => {
    setUsername("demo.admin");
    setPassword(import.meta.env.VITE_DEMO_ADMIN_PASSWORD ?? "");
  };

  return (
    <main className="login-shell">
      <section className="login-card" aria-labelledby="login-title">
        <p className="eyebrow">Analyse documentaire sourcée</p>
        <h1 id="login-title">EvidenceDesk</h1>
        <p className="lede">
          Vérifiez un dossier fournisseur synthétique sans confondre réponse et preuve.
        </p>
        <form onSubmit={submit}>
          <label>
            Identifiant
            <input
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </label>
          <label>
            Mot de passe
            <input
              autoComplete="current-password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </label>
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
          <button className="primary" disabled={loading}>
            {loading ? "Connexion…" : "Se connecter"}
          </button>
        </form>
        <button className="secondary" onClick={fillDemoAccount} type="button">
          Préremplir le compte de démonstration
        </button>
        <p className="notice">
          Les identifiants locaux sont documentés dans le README. Le jeton reste uniquement en
          mémoire et disparaît à la fermeture de l’onglet.
        </p>
      </section>
    </main>
  );
}

function EvaluationCard({ run }: { run: EvaluationRun }) {
  const metrics = run.metrics;
  return (
    <article className="evaluation-card">
      <div className="panel-heading">
        <div>
          <strong>{run.split === "holdout" ? "Holdout indépendant" : "Développement"}</strong>
          <p>{run.dataset_version}</p>
        </div>
        <span className={`verdict ${run.verdict.toLowerCase()}`}>{run.verdict}</span>
      </div>
      <dl className="metric-grid">
        <div>
          <dt>Précision citations</dt>
          <dd>{percentage(metrics.citation_precision)}</dd>
        </div>
        <div>
          <dt>Exactitude cas sourcés</dt>
          <dd>{percentage(metrics.citation_case_accuracy)}</dd>
        </div>
        <div>
          <dt>F1 extraction</dt>
          <dd>{percentage(metrics.extraction_f1)}</dd>
        </div>
        <div>
          <dt>Exactitude abstention</dt>
          <dd>{percentage(metrics.abstention_accuracy)}</dd>
        </div>
        <div>
          <dt>Recall récupération @5</dt>
          <dd>{percentage(metrics.retrieval_recall_at_5)}</dd>
        </div>
        <div>
          <dt>Latence médiane</dt>
          <dd>{numberMetric(metrics.latency_median_ms, "ms")}</dd>
        </div>
        <div>
          <dt>Latence p95</dt>
          <dd>{numberMetric(metrics.latency_p95_ms, "ms")}</dd>
        </div>
        <div>
          <dt>Taux d’erreur</dt>
          <dd>{percentage(metrics.error_rate)}</dd>
        </div>
        <div>
          <dt>Coût estimé</dt>
          <dd>
            {typeof metrics.estimated_cost_usd === "number"
              ? `${metrics.estimated_cost_usd.toFixed(4)} USD`
              : "Non mesuré"}
          </dd>
        </div>
      </dl>
      <p className="run-meta">
        {run.mode} · paramètres {run.parameters_version}
      </p>
    </article>
  );
}

export function App() {
  const [session, setSession] = useState<TokenResponse | null>(null);
  const [dossiers, setDossiers] = useState<Dossier[]>([]);
  const [selected, setSelected] = useState<Dossier | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [extractions, setExtractions] = useState<Extraction[]>([]);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([]);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [question, setQuestion] = useState("");
  const [syntheticConfirmed, setSyntheticConfirmed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const token = session?.access_token;

  const loadDocuments = useCallback(
    async (dossierId: string, activeToken: string) => {
      const items = await api.documents(dossierId, activeToken);
      setDocuments(items);
      setSelectedDocumentId((current) => {
        if (current && items.some((item) => item.id === current)) return current;
        return items[0]?.id ?? null;
      });
      return items;
    },
    [],
  );

  const loadAdminViews = useCallback(async (activeToken: string) => {
    const [events, runs, current] = await Promise.all([
      api.audit(activeToken),
      api.evaluations(activeToken),
      api.status(activeToken),
    ]);
    setAudit(events);
    setEvaluations(runs);
    setSystem(current);
  }, []);

  useEffect(() => {
    if (!token) return;
    setLoading(true); // eslint-disable-line react-hooks/set-state-in-effect
    api
      .dossiers(token)
      .then((items) => {
        setDossiers(items);
        setSelected(items[0] ?? null);
      })
      .catch((caught) => setError(apiMessage(caught)))
      .finally(() => setLoading(false));
  }, [token]);

  useEffect(() => {
    if (!selected || !token) return;
    setLoading(true); // eslint-disable-line react-hooks/set-state-in-effect
    Promise.all([loadDocuments(selected.id, token), api.extraction(selected.id, token)])
      .then(([, values]) => setExtractions(values))
      .catch((caught) => setError(apiMessage(caught)))
      .finally(() => setLoading(false));
  }, [loadDocuments, selected, token]);

  useEffect(() => {
    if (!selectedDocumentId || !token) {
      setChunks([]); // eslint-disable-line react-hooks/set-state-in-effect
      return;
    }
    api
      .content(selectedDocumentId, token)
      .then(setChunks)
      .catch((caught) => setError(apiMessage(caught)));
  }, [selectedDocumentId, token]);

  useEffect(() => {
    if (!token || session?.user.role !== "admin") return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadAdminViews(token).catch((caught) => setError(apiMessage(caught)));
  }, [loadAdminViews, session?.user.role, token]);

  const pendingDocuments = useMemo(
    () => documents.some((item) => item.status === "queued" || item.status === "processing"),
    [documents],
  );

  useEffect(() => {
    if (!pendingDocuments || !selected || !token) return;
    const poll = window.setInterval(() => {
      loadDocuments(selected.id, token)
        .then(async (items) => {
          if (!items.some((item) => item.status === "queued" || item.status === "processing")) {
            setExtractions(await api.extraction(selected.id, token));
            if (session?.user.role === "admin") await loadAdminViews(token);
          }
        })
        .catch((caught) => setError(apiMessage(caught)));
    }, 1_000);
    return () => window.clearInterval(poll);
  }, [loadAdminViews, loadDocuments, pendingDocuments, selected, session?.user.role, token]);

  if (!session) return <Login onLogin={setSession} />;

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    if (!selected || !token || !question.trim()) return;
    setLoading(true);
    setError("");
    try {
      setAnswer(await api.ask(selected.id, question, token));
      if (session.user.role === "admin") await loadAdminViews(token);
    } catch (caught) {
      setError(apiMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  const openCitation = async (citation: Citation) => {
    if (!token) return;
    setSelectedDocumentId(citation.document_id);
    try {
      const sourceChunks = await api.content(citation.document_id, token);
      setChunks(sourceChunks);
      window.setTimeout(() => {
        document.getElementById(`chunk-${citation.chunk_id}`)?.focus();
      }, 0);
    } catch (caught) {
      setError(apiMessage(caught));
    }
  };

  const upload = async (file: File) => {
    if (!selected || !token || !syntheticConfirmed) return;
    setLoading(true);
    setError("");
    try {
      const created = await api.upload(selected.id, file, token);
      setDocuments((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      setSelectedDocumentId(created.id);
      setSyntheticConfirmed(false);
    } catch (caught) {
      setError(apiMessage(caught));
    } finally {
      setLoading(false);
    }
  };

  const selectedDocument = documents.find((item) => item.id === selectedDocumentId);

  return (
    <div className="app-shell">
      <a className="skip" href="#workspace">
        Aller au contenu
      </a>
      <header>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            E
          </span>
          <strong>EvidenceDesk</strong>
        </div>
        <nav aria-label="Navigation principale">
          <a href="#workspace">Dossier</a>
          <a href="#import">Import</a>
          <a href="#extraction">Extraction</a>
          <a href="#audit">Audit</a>
          <a href="#evaluation">Évaluation</a>
        </nav>
        <button className="profile" onClick={() => setSession(null)} type="button">
          {session.user.username} <span>{session.user.role}</span>
        </button>
      </header>

      <main id="workspace">
        {error && (
          <p role="alert" className="warning">
            {error}
          </p>
        )}
        {loading && (
          <p className="loading" aria-live="polite">
            Mise à jour des preuves…
          </p>
        )}
        {!loading && dossiers.length === 0 && (
          <section className="panel empty-state">
            <h1>Aucun dossier disponible</h1>
            <p>Votre rôle ne donne accès à aucun dossier.</p>
          </section>
        )}

        {selected && (
          <>
            <section className="hero">
              <div>
                <p className="eyebrow">
                  {selected.is_synthetic ? "Dossier synthétique" : "Dossier"}
                </p>
                <h1>{selected.name}</h1>
                <p>{selected.description}</p>
              </div>
              <div className="hero-status" aria-label="État des documents">
                <span className="status completed">
                  {documents.filter((item) => item.status === "completed").length} terminés
                </span>
                {pendingDocuments && <span className="status processing">Traitement actif</span>}
              </div>
            </section>
            <p className="data-boundary">
              Démonstration locale : importez uniquement des documents entièrement synthétiques.
            </p>

            <div className="workspace-grid">
              <aside id="import" className="panel documents">
                <div className="panel-heading">
                  <h2>Documents</h2>
                  <span>{documents.length}</span>
                </div>
                {documents.length ? (
                  <ul>
                    {documents.map((item) => (
                      <li key={item.id}>
                        <button
                          className={`document ${item.id === selectedDocumentId ? "selected" : ""}`}
                          onClick={() => setSelectedDocumentId(item.id)}
                          type="button"
                        >
                          <span>{item.filename}</span>
                          <small>
                            <span className={`status-dot ${item.status}`} aria-hidden="true" />
                            {STATUS_LABELS[item.status] ?? item.status}
                            {item.error_code ? ` · ${item.error_code}` : ""}
                          </small>
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">Aucun document importé.</p>
                )}
                {(session.user.role === "admin" || session.user.role === "analyst") && (
                  <div className="upload-box">
                    <label className="check-row">
                      <input
                        type="checkbox"
                        checked={syntheticConfirmed}
                        onChange={(event) => setSyntheticConfirmed(event.target.checked)}
                      />
                      Je confirme que ce fichier est synthétique.
                    </label>
                    <label className={`file-button ${syntheticConfirmed ? "" : "disabled"}`}>
                      Choisir un PDF, TXT ou Markdown
                      <input
                        type="file"
                        accept=".pdf,.txt,.md,text/plain,text/markdown,application/pdf"
                        disabled={!syntheticConfirmed || loading}
                        onChange={(event) => {
                          const file = event.target.files?.[0];
                          if (file) void upload(file);
                          event.target.value = "";
                        }}
                      />
                    </label>
                    <small>10 Mio maximum · traitement par le worker Redis/ARQ</small>
                  </div>
                )}
              </aside>

              <section className="panel question-panel">
                <div className="panel-heading">
                  <h2>Question avec preuves</h2>
                  <span>Mode extractif local</span>
                </div>
                <form onSubmit={ask}>
                  <label htmlFor="question">Question sur ce dossier</label>
                  <textarea
                    id="question"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="Ex. Quel délai de notification impose le contrat ?"
                    required
                    maxLength={1_000}
                  />
                  <button className="primary" disabled={loading}>
                    Rechercher des preuves
                  </button>
                </form>
                {answer ? (
                  <article className={`answer ${answer.status}`} aria-live="polite">
                    <div className="answer-meta">
                      <strong>
                        {answer.status === "answered"
                          ? "Réponse sourcée"
                          : answer.status === "ambiguous"
                            ? "Preuves contradictoires"
                            : "Abstention explicite"}
                      </strong>
                      <span>Confiance {Math.round(answer.confidence * 100)} %</span>
                    </div>
                    <p>{answer.answer}</p>
                    <div className="citation-row">
                      {answer.citations.map((citation) => (
                        <CitationButton
                          citation={citation}
                          onOpen={(item) => void openCitation(item)}
                          key={`${citation.chunk_id}-${citation.excerpt}`}
                        />
                      ))}
                    </div>
                  </article>
                ) : (
                  <p className="empty-state answer-placeholder">
                    La réponse apparaîtra ici. Sans preuve suffisante, EvidenceDesk refusera de
                    conclure.
                  </p>
                )}
              </section>

              <section className="panel source-panel" aria-live="polite">
                <div className="panel-heading">
                  <h2>Passages source</h2>
                  <span>{selectedDocument?.filename ?? "Aucun document"}</span>
                </div>
                <div className="source-scroll" tabIndex={0} aria-label="Passages du document">
                  {chunks.length ? (
                    chunks.map((chunk) => (
                      <article
                        id={`chunk-${chunk.id}`}
                        tabIndex={-1}
                        className="document-page"
                        key={chunk.id}
                      >
                        <p className="page-label">
                          Page {chunk.page}
                          {chunk.section ? ` · ${chunk.section}` : ""}
                        </p>
                        <p>{chunk.text}</p>
                      </article>
                    ))
                  ) : (
                    <p className="empty-state">
                      {selectedDocument?.status === "completed"
                        ? "Aucun passage disponible."
                        : "Le contenu apparaîtra après traitement."}
                    </p>
                  )}
                </div>
              </section>
            </div>

            <section id="extraction" className="panel extraction">
              <div className="panel-heading">
                <h2>Extraction structurée</h2>
                <span>Chaque valeur est reliée à sa preuve</span>
              </div>
              {extractions.length ? (
                <div className="extraction-documents">
                  {extractions.map((item, index) => (
                    <details key={item.document_id} open={index === 0}>
                      <summary>
                        {item.document_name} <span>{item.schema_version}</span>
                      </summary>
                      <dl className="field-grid">
                        {Object.entries(item.payload).map(([fieldName, rawField]) => {
                          if (!isEvidenceField(rawField)) return null;
                          return (
                            <div key={fieldName}>
                              <dt>{FIELD_LABELS[fieldName] ?? fieldName}</dt>
                              <dd>
                                <span>{displayValue(rawField.value)}</span>
                                <span className="citation-row">
                                  {rawField.citations.map((citation) => (
                                    <CitationButton
                                      citation={citation}
                                      onOpen={(source) => void openCitation(source)}
                                      key={`${fieldName}-${citation.chunk_id}-${citation.excerpt}`}
                                    />
                                  ))}
                                </span>
                              </dd>
                            </div>
                          );
                        })}
                      </dl>
                    </details>
                  ))}
                </div>
              ) : (
                <p className="empty-state">Aucune extraction calculée.</p>
              )}
            </section>
          </>
        )}

        <div className="lower-grid">
          <section id="audit" className="panel audit-panel">
            <div className="panel-heading">
              <h2>Journal d’audit</h2>
              <span>{session.user.role === "admin" ? `${audit.length} événements` : "Admin"}</span>
            </div>
            {session.user.role !== "admin" ? (
              <p className="empty-state">Accès administrateur requis.</p>
            ) : audit.length ? (
              <div className="table-scroll" tabIndex={0} aria-label="Événements du journal d’audit">
                <table>
                  <thead>
                    <tr>
                      <th>UTC</th>
                      <th>Utilisateur</th>
                      <th>Action</th>
                      <th>Résultat</th>
                      <th>Corrélation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {audit.map((item) => (
                      <tr key={item.id}>
                        <td>{new Date(item.created_at).toISOString()}</td>
                        <td>{item.actor_id ? item.actor_id.slice(0, 8) : "système"}</td>
                        <td>{item.action}</td>
                        <td>{item.result}</td>
                        <td title={item.correlation_id}>{item.correlation_id.slice(0, 8)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty-state">Aucun événement d’audit.</p>
            )}
          </section>

          <section id="evaluation" className="panel evaluation-panel">
            <div className="panel-heading">
              <h2>Évaluation reproductible</h2>
              <span>Résultats calculés</span>
            </div>
            {evaluations.length ? (
              evaluations.map((item) => <EvaluationCard run={item} key={item.id} />)
            ) : (
              <p className="empty-state">
                Aucun résultat enregistré. La page ne remplace jamais une absence de mesure par
                une valeur fictive.
              </p>
            )}
          </section>

          <section id="status" className="panel status-panel">
            <div className="panel-heading">
              <h2>État opérationnel</h2>
              <span>Vue administrateur</span>
            </div>
            {system ? (
              <dl className="status-grid">
                <div>
                  <dt>Mode</dt>
                  <dd>{system.mode}</dd>
                </div>
                <div>
                  <dt>Tâches en échec</dt>
                  <dd>{system.failed_tasks}</dd>
                </div>
                {Object.entries(system.documents).map(([statusName, count]) => (
                  <div key={statusName}>
                    <dt>{STATUS_LABELS[statusName] ?? statusName}</dt>
                    <dd>{count}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="empty-state">Aucun état disponible pour ce rôle.</p>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
