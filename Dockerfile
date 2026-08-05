FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY polke ./polke
RUN pip install --no-cache-dir .

# Both English pipelines the annotators are tested with; pick one at runtime
# with POLKE_SPACY_MODEL (default en_core_web_sm). Both are small (<50 MB).
RUN python -m spacy download en_core_web_sm \
 && python -m spacy download en_core_web_md

EXPOSE 8100

# start-period covers spaCy load + detector build on first boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4)"

CMD ["polke", "serve", "--host", "0.0.0.0", "--port", "8100"]
