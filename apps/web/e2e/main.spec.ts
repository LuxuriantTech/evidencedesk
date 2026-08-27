import { test, expect, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function deleteTemporaryDocuments(request: APIRequestContext): Promise<void> {
  if (process.env.E2E_REAL_API !== "1") return;
  const password = process.env.E2E_DEMO_ADMIN_PASSWORD ?? "";
  const login = await request.post("/api/v1/auth/token", {
    form: { username: "demo.admin", password },
  });
  if (!login.ok()) return;
  const token = ((await login.json()) as { access_token: string }).access_token;
  const dossiers = await request.get("/api/v1/dossiers", {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!dossiers.ok()) return;
  const dossierId = ((await dossiers.json()) as Array<{ id: string }>)[0]?.id;
  if (!dossierId) return;
  const documents = await request.get(`/api/v1/dossiers/${dossierId}/documents`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!documents.ok()) return;
  const temporaryFilenames = new Set([
    "demo-supplier-note.md",
    "e2e-synthetic-supplier.txt",
  ]);
  const temporary = ((await documents.json()) as Array<{ id: string; filename: string }>).filter(
    (item) => temporaryFilenames.has(item.filename),
  );
  for (const document of temporary) {
    await request.delete(`/api/v1/documents/${document.id}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
  }
}

test.beforeEach(async ({ request }) => {
  await deleteTemporaryDocuments(request);
});

test.afterEach(async ({ request }) => {
  await deleteTemporaryDocuments(request);
});

test("loads an API-backed sourced dossier", async ({ page }) => {
  test.skip(process.env.E2E_REAL_API === "1", "La pile réelle fournit ses propres données et URL.");
  await page.route("**/api/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const data: Record<string, unknown> = {
      "/api/v1/auth/token": { access_token: "token", token_type: "bearer", expires_in: 1800, user: { id: "u1", username: "demo.admin", role: "admin" } },
      "/api/v1/dossiers": [{ id: "d1", name: "Dossier Atlas", description: "Corpus synthétique", is_synthetic: true }],
      "/api/v1/dossiers/d1/documents": [{ id: "doc1", filename: "accord.pdf", media_type: "application/pdf", status: "completed", task_id: null, error_code: null }],
      "/api/v1/dossiers/d1/extraction": [],
      "/api/v1/documents/doc1/content": [{ id: "c1", page: 2, section: "Durée", ordinal: 1, text: "Renouvellement le 30 septembre 2027." }],
      "/api/v1/audit-events": [],
      "/api/v1/evaluations": [],
      "/api/v1/status": { mode: "extractive-local", public_demo_mode: true, documents: { completed: 1 }, failed_tasks: 0 },
    };
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(data[path] ?? { status: "abstained", answer: "Aucune preuve.", confidence: 0, mode: "extractive-local", citations: [], correlation_id: "q1" }) });
  });
  await page.goto("/"); await page.getByLabel("Identifiant").fill("demo.admin"); await page.getByLabel("Mot de passe").fill("EvidenceDemo-Admin-2026!"); await page.getByRole("button", { name: "Se connecter" }).click();
  await expect(page.getByRole("heading", { name: "Dossier Atlas" })).toBeVisible();
  await expect(page.getByText("Renouvellement le 30 septembre 2027.")).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.filter((item) => ["critical", "serious"].includes(item.impact ?? ""))).toEqual([]);
});

test("real stack completes the sourced synthetic dossier workflow", async ({ page, request }) => {
  test.skip(process.env.E2E_REAL_API !== "1", "Activer E2E_REAL_API=1 pour la pile Docker/API réelle.");
  const password = process.env.E2E_DEMO_ADMIN_PASSWORD ?? "";
  expect(password).not.toBe("");

  await page.goto("/");
  await page.getByLabel("Identifiant").fill("demo.admin");
  await page.getByLabel("Mot de passe").fill(password);
  await page.getByRole("button", { name: "Se connecter" }).click();
  await expect(page.getByRole("heading", { name: "Northwind supplier review" })).toBeVisible();
  await expect(page.getByText("3 terminés")).toBeVisible();

  const uploadResponsePromise = page.waitForResponse(
    (response) => response.request().method() === "POST" && response.url().endsWith("/documents"),
  );
  await page.getByLabel("Je confirme que ce fichier est synthétique.").check();
  await page
    .locator('input[type="file"]')
    .setInputFiles("../../examples/demo-supplier-note.md");
  const uploadResponse = await uploadResponsePromise;
  expect(uploadResponse.status()).toBe(202);
  const uploaded = (await uploadResponse.json()) as { id: string; status: string };
  expect(uploaded.status).toBe("queued");
  const uploadedDocument = page.locator("button.document", {
    hasText: "demo-supplier-note.md",
  });
  await expect(uploadedDocument).toContainText("Terminé", { timeout: 15_000 });

  await page
    .getByLabel("Question sur ce dossier")
    .fill("Quel montant de plateforme annuel est indiqué ?");
  await page.getByRole("button", { name: "Rechercher des preuves" }).click();
  const sourcedAnswer = page.locator("article.answer");
  await expect(sourcedAnswer.getByText("Réponse sourcée")).toBeVisible();
  await expect(sourcedAnswer.locator(".answer-text")).toHaveText("EUR 48,000");
  const sourceButton = sourcedAnswer.getByRole("button", {
    name: /Voir la source northstar_master_services_agreement\.pdf, page 1/,
  });
  await sourceButton.click();
  await expect(
    page.locator(".source-panel").getByText("Annual platform fee: EUR 48,000"),
  ).toBeVisible();

  await page.getByLabel("Question sur ce dossier").fill("Quel est le numéro TVA de Northstar ?");
  await page.getByRole("button", { name: "Rechercher des preuves" }).click();
  const abstention = page.locator("article.answer");
  await expect(abstention.getByText("Abstention explicite")).toBeVisible();
  await expect(abstention.getByText("Insufficient evidence in the selected dossier.")).toBeVisible();

  await expect(page.getByRole("heading", { name: "Extraction structurée" })).toBeVisible();
  const northstarExtraction = page
    .locator("#extraction details")
    .filter({ hasText: "northstar_master_services_agreement.pdf" });
  await northstarExtraction.locator("summary").click();
  await expect(
    northstarExtraction.getByText("Northstar Logistics Systems Ltd."),
  ).toBeVisible();
  await page
    .locator("button.document", { hasText: "northstar_master_services_agreement.pdf" })
    .click();
  await expect(page.locator(".source-panel").getByText(/\[EMAIL REDACTED\]/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Journal d’audit" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "dossier.ask" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Évaluation reproductible" })).toBeVisible();
  const v7Run = page
    .locator("article.evaluation-card")
    .filter({ hasText: "grounded-local-v3.0-frozen-v7" });
  await expect(v7Run.getByText("FAIL")).toBeVisible();
  await expect(v7Run.getByText("80,0 %").first()).toBeVisible();
  await expect(v7Run.getByText("45,7 %").first()).toBeVisible();

  const desktopA11y = await new AxeBuilder({ page }).analyze();
  expect(
    desktopA11y.violations.filter((item) => ["critical", "serious"].includes(item.impact ?? "")),
  ).toEqual([]);
  await page.setViewportSize({ width: 375, height: 812 });
  const mobileA11y = await new AxeBuilder({ page }).analyze();
  expect(
    mobileA11y.violations.filter((item) => ["critical", "serious"].includes(item.impact ?? "")),
  ).toEqual([]);

  const login = await request.post("/api/v1/auth/token", {
    form: { username: "demo.admin", password },
  });
  expect(login.ok()).toBeTruthy();
  const token = ((await login.json()) as { access_token: string }).access_token;
  const deleted = await request.delete(`/api/v1/documents/${uploaded.id}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(deleted.status()).toBe(204);
});
