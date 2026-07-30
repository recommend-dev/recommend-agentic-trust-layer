# OKF verifier UI (okf_server.py). The Claim Checker (server.py) is imported by the
# pipeline but not exposed — this image serves the OKF upload UI only.
#
#   docker build -t okf-verifier .
#   docker run -p 8898:8898 --env-file .env okf-verifier
#
# On Cloud Run the injected PORT env is honoured automatically; run with
# --max-instances 1 (verification jobs live in process memory).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
EXPOSE 8898
CMD ["python", "okf_server.py"]
