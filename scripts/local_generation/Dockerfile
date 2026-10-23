# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
FROM --platform=linux/amd64 ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates libgomp1 libssl3t64 libstdc++6 python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 65532 --no-create-home --shell /usr/sbin/nologin modelrunner \
    && install -d -o root -g root -m 0755 /opt/evidencedesk

COPY scripts/local_generation/__init__.py scripts/local_generation/bundle.py \
    scripts/local_generation/manifest.json scripts/local_generation/verify_and_run.py \
    /opt/evidencedesk/local_generation/

USER 65532:65532
WORKDIR /opt/evidencedesk
ENTRYPOINT ["python3", "-m", "local_generation.verify_and_run"]
