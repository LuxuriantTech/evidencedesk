import { test } from "@playwright/test";

test("captures the current real dashboard for UX review", async ({ page }) => {
  test.skip(
    process.env.CAPTURE_SCREENSHOTS !== "1",
    "Capture explicite uniquement sur la pile locale réelle.",
  );
  await page.goto("/");
  await page.getByLabel("Identifiant").fill("demo.admin");
  await page
    .getByLabel("Mot de passe")
    .fill(process.env.E2E_DEMO_ADMIN_PASSWORD ?? "");
  await page.getByRole("button", { name: "Se connecter" }).click();
  await page.getByRole("heading", { name: "Northwind supplier review" }).waitFor();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({
    path: "../../docs/screenshots/01-dashboard-desktop.png",
    fullPage: false,
  });

  await page
    .getByLabel("Question sur ce dossier")
    .fill("Quel montant de plateforme annuel est indiqué ?");
  await page.getByRole("button", { name: "Rechercher des preuves" }).click();
  await page.locator("article.answer").getByText("Réponse sourcée").waitFor();
  await page
    .locator("article.answer")
    .getByRole("button", { name: /northstar_master_services_agreement\.pdf, page 1/ })
    .click();
  await page.locator(".workspace-grid").screenshot({
    path: "../../docs/screenshots/02-sourced-answer-desktop.png",
  });

  await page.locator("#evaluation").evaluate((element) => {
    const top = element.getBoundingClientRect().top + window.scrollY - 90;
    window.scrollTo({ top, behavior: "auto" });
  });
  await page.locator("#evaluation").screenshot({
    path: "../../docs/screenshots/03-evaluation-results.png",
  });

  await page.setViewportSize({ width: 375, height: 812 });
  await page.getByRole("heading", { name: "Northwind supplier review" }).scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "../../docs/screenshots/04-dashboard-mobile.png",
    fullPage: false,
  });
});
