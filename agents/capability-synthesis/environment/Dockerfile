# Attribution analysis runtime.
#
# The analyst receives read-only evidence and a writable work directory.
# It can inspect workbook evidence with openpyxl. No Docker socket or host
# Harbor runtime is mounted here.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash git curl ca-certificates procps coreutils util-linux jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openpyxl==3.1.5

WORKDIR /app

# The one helper the agent is expected to use: how much budget is left.
COPY bin/ /usr/local/bin/
RUN chmod +x /usr/local/bin/time_left
