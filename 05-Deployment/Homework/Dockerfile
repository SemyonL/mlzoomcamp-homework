FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" ./

RUN uv sync --locked

COPY "q4_server.py" "pipeline_v1.bin" ./

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "q4_server:app", "--host", "0.0.0.0", "--port", "8000" ]