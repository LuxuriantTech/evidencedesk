import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { App } from "./App";

const json = (value: unknown) =>
  new Response(JSON.stringify(value), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
describe("EvidenceDesk REST flow", () => {
  it("shows the research warning and never prefills a browser password", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(
      screen.getByText(
        "Research prototype using synthetic data only. Not validated for production, legal, medical, financial or compliance decisions.",
      ),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Préremplir l’identifiant analyste" }));
    expect(screen.getByLabelText("Identifiant")).toHaveValue("demo.analyst");
    expect(screen.getByLabelText("Mot de passe")).toHaveValue("");
  });

  it("authenticates then renders API-provided dossier data", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        json({
          access_token: "memory-token",
          token_type: "bearer",
          expires_in: 1800,
          user: { id: "u1", username: "demo.admin", role: "admin" },
        }),
      )
      .mockResolvedValueOnce(
        json([
          {
            id: "d1",
            name: "Dossier Atlas",
            description: "Corpus synthétique",
            is_synthetic: true,
          },
        ]),
      )
      .mockResolvedValueOnce(json([]))
      .mockResolvedValueOnce(json([]))
      .mockResolvedValueOnce(json({ mode: "extractive-local", public_demo_mode: true, documents: {}, failed_tasks: 0 }))
      .mockResolvedValueOnce(
        json([
          {
            id: "doc1",
            filename: "accord.pdf",
            media_type: "application/pdf",
            status: "completed",
            task_id: null,
            error_code: null,
          },
        ]),
      )
      .mockResolvedValueOnce(json([]))
      .mockResolvedValueOnce(
        json([
          {
            id: "c1",
            page: 2,
            section: "Durée",
            ordinal: 1,
            text: "Renouvellement le 30 septembre 2027.",
          },
        ]),
      );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Identifiant"), "demo.admin");
    await user.type(
      screen.getByLabelText("Mot de passe"),
      "EvidenceDemo-Admin-2026!",
    );
    await user.click(screen.getByRole("button", { name: "Se connecter" }));
    expect(
      await screen.findByRole("heading", { name: "Dossier Atlas" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByText("Renouvellement le 30 septembre 2027."),
    ).toBeInTheDocument();
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/v1/auth/token",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/v1/dossiers",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer memory-token",
        }),
      }),
    );
    fetchMock.mockRestore();
  });
  it("shows a safe error when authentication is denied", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({ detail: { code: "invalid_credentials" } }),
          { status: 401 },
        ),
      );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Identifiant"), "bad");
    await user.type(screen.getByLabelText("Mot de passe"), "bad");
    await user.click(screen.getByRole("button", { name: "Se connecter" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Identifiant ou mot de passe invalide.",
    );
    fetchMock.mockRestore();
  });
  it("renders the administrator audit and operational state from REST", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        json({
          access_token: "t",
          token_type: "bearer",
          expires_in: 1800,
          user: { id: "u", username: "demo.admin", role: "admin" },
        }),
      )
      .mockResolvedValueOnce(json([]));
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Identifiant"), "demo.admin");
    await user.type(screen.getByLabelText("Mot de passe"), "x");
    await user.click(screen.getByRole("button", { name: "Se connecter" }));
    expect(await screen.findByText("État opérationnel")).toBeInTheDocument();
    fetchMock.mockRestore();
  });

  it("renders measured evaluation metrics and field-level extraction evidence", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input) => {
        const path = String(input);
        if (path === "/api/v1/auth/token") {
          return json({
            access_token: "t",
            token_type: "bearer",
            expires_in: 1800,
            user: { id: "u", username: "demo.admin", role: "admin" },
          });
        }
        if (path === "/api/v1/dossiers") {
          return json([
            {
              id: "d1",
              name: "Dossier Atlas",
              description: "Synthétique",
              is_synthetic: true,
            },
          ]);
        }
        if (path === "/api/v1/dossiers/d1/documents") {
          return json([
            {
              id: "doc1",
              filename: "accord.pdf",
              media_type: "application/pdf",
              status: "completed",
              task_id: "task1",
              error_code: null,
            },
          ]);
        }
        if (path === "/api/v1/dossiers/d1/extraction") {
          return json([
            {
              document_id: "doc1",
              document_name: "accord.pdf",
              schema_version: "supplier-v1",
              payload: {
                effective_date: {
                  value: "2026-01-15",
                  citations: [
                    {
                      chunk_id: "c1",
                      document_id: "doc1",
                      document_name: "accord.pdf",
                      page: 1,
                      section: null,
                      excerpt: "Effective date: 2026-01-15",
                    },
                  ],
                },
              },
            },
          ]);
        }
        if (path === "/api/v1/documents/doc1/content") {
          return json([
            {
              id: "c1",
              page: 1,
              section: null,
              ordinal: 1,
              text: "Effective date: 2026-01-15",
            },
          ]);
        }
        if (path === "/api/v1/audit-events") return json([]);
        if (path === "/api/v1/status") {
          return json({
            mode: "extractive-local",
            public_demo_mode: true,
            documents: { completed: 1 },
            failed_tasks: 0,
          });
        }
        if (path === "/api/v1/evaluations") {
          return json([
            {
              id: "eval1",
              dataset_version: "2026.08.26.2-blind",
              parameters_version: "extractive-local-v1.1-frozen",
              split: "holdout",
              mode: "extractive-local",
              metrics: {
                citation_precision: 0.7,
                citation_case_accuracy: 0.777778,
                extraction_f1: 0.941176,
                abstention_accuracy: 0.8,
                latency_median_ms: 1.448,
                latency_p95_ms: 1.976,
                retrieval_recall_at_5: 0.9,
                error_rate: 0,
                estimated_cost_usd: 0,
              },
              verdict: "FAIL",
              created_at: "2026-08-26T18:34:34Z",
            },
          ]);
        }
        throw new Error(`Unexpected request: ${path}`);
      },
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Identifiant"), "demo.admin");
    await user.type(screen.getByLabelText("Mot de passe"), "x");
    await user.click(screen.getByRole("button", { name: "Se connecter" }));

    expect(await screen.findByText("Date d’effet")).toBeInTheDocument();
    expect(screen.getByText("2026-01-15")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /source accord.pdf, page 1/i }),
    ).toBeInTheDocument();
    expect(await screen.findByText("70,0 %")).toBeInTheDocument();
    expect(screen.getByText("77,8 %")).toBeInTheDocument();
    expect(screen.getByText("94,1 %")).toBeInTheDocument();
    expect(screen.getByText("1.448 ms")).toBeInTheDocument();
    expect(screen.getByText("90,0 %")).toBeInTheDocument();
    fetchMock.mockRestore();
  });

  it("renders a partially supported answer with its explicit source and reason", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = String(input);
      if (path === "/api/v1/auth/token") {
        return json({
          access_token: "t",
          token_type: "bearer",
          expires_in: 1800,
          user: { id: "u", username: "demo.analyst", role: "analyst" },
        });
      }
      if (path === "/api/v1/dossiers") {
        return json([{ id: "d1", name: "Dossier Atlas", description: "Synthétique", is_synthetic: true }]);
      }
      if (path === "/api/v1/dossiers/d1/documents") {
        return json([{ id: "doc1", filename: "accord.pdf", media_type: "application/pdf", status: "completed", task_id: null, error_code: null }]);
      }
      if (path === "/api/v1/dossiers/d1/extraction") return json([]);
      if (path === "/api/v1/documents/doc1/content") return json([]);
      if (path === "/api/v1/dossiers/d1/ask") {
        return json({
          status: "partially_supported",
          answerable: false,
          answer: "La preuve ne précise pas la valeur demandée.",
          confidence: 0.61,
          mode: "deterministic-v3-a",
          supporting_document: "doc1",
          supporting_page: 2,
          supporting_excerpt: "Le suivi confirme l'événement, sans date définitive.",
          ambiguity_reason: "L'événement est établi, mais la date demandée est absente.",
          extracted_fields: {},
          citations: [{ chunk_id: "c2", document_id: "doc1", document_name: "accord.pdf", page: 2, section: null, excerpt: "Le suivi confirme l'événement, sans date définitive." }],
          correlation_id: "correlation-v3",
        });
      }
      throw new Error(`Unexpected request: ${path}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Identifiant"), "demo.analyst");
    await user.type(screen.getByLabelText("Mot de passe"), "x");
    await user.click(screen.getByRole("button", { name: "Se connecter" }));
    await screen.findByRole("heading", { name: "Dossier Atlas" });
    await user.type(screen.getByLabelText("Question sur ce dossier"), "Quelle date est confirmée ?");
    await user.click(screen.getByRole("button", { name: "Rechercher des preuves" }));

    expect(await screen.findByText("Preuve partielle")).toBeInTheDocument();
    expect(screen.getByText(/L'événement est établi, mais la date demandée est absente\./)).toBeInTheDocument();
    expect(screen.getByText("Extrait utilisé : Le suivi confirme l'événement, sans date définitive.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /source accord.pdf, page 2/i })).toBeInTheDocument();
    fetchMock.mockRestore();
  });
});
