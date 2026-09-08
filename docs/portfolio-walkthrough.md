# A sourced answer, with its limits visible

EvidenceDesk is a local prototype for checking synthetic supplier documents. A FastAPI API, Redis/ARQ worker, PostgreSQL/pgvector store and React interface separate document processing from the request and let a reviewer open the cited page beside an answer.

## Try the recorded example

Use the Docker Compose quickstart in the README. Ask `Quel montant de plateforme annuel est indiqué ?` in the synthetic supplier dossier. The verified local example returns `EUR 48,000` and cites `northstar_master_services_agreement.pdf`, page 1. Open the source to check the passage. An unsupported question should result in explicit abstention; a server or processing error is a separate state.

The hosted portfolio walkthrough is static. It does not upload files or execute the local backend. Use only synthetic fixtures when running the application locally.

## The evaluation remains negative

The frozen v7 holdout is `HONEST_NEGATIVE`: Recall@5 100% (25/25 answerable cases), answerable accuracy 36% (9/25), citation precision 80% (12/15), citation recall 48% (12/25), abstention accuracy 80% (12/15), extraction F1 45.67%. These historical records are preserved without rerunning or rescoring. Technical tests and later fixes do not establish an improved holdout result. This is not a production or decision-making service.

## My role

I use AI extensively to build these projects. I understand and review the code, and I am still learning to write it independently.
