# Deploying on Oracle Cloud (Always Free, ARM, 1 OCPU / 6 GB)

The engine needs ~550 MB RAM (all three tiers) and ~100 ms of CPU per real post (long news posts use all 128
tokens), so one core handles roughly **10 posts/second (~870k posts/day)**. That was measured on an Intel
laptop core; Ampere cores differ, so check on your instance with step 3. If you need more throughput, set
`DISABLE_TRANSFORMER=1` (TF-IDF tier only: ~1 ms/post, much lower accuracy). `DISABLE_ENGLISH_EXPERT=1` saves ~270 MB.

## 1. One-time setup on the instance (Ubuntu 22.04/24.04 image)

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker
git clone https://github.com/om0852/Sentiment-NLP.git && cd Sentiment-NLP
# until PR #1 is merged into main, use the PR branch instead:
# git clone -b claude/sentiment-engine-testing-efc0bb https://github.com/om0852/Sentiment-NLP.git && cd Sentiment-NLP
```

Open port 8000 in **two** places. Oracle blocks it by default:

1. OCI console → your VCN → Security List → add an ingress rule: TCP, destination port 8000, from your clients'
   IP range. Use `0.0.0.0/0` only if the API is meant to be public.
2. On the instance's own firewall. The rule must sit **above** Oracle's `REJECT` rule, whose position differs between
   the standard and Minimal Ubuntu images, so look it up instead of hard-coding it:
   ```bash
   REJ=$(sudo iptables -L INPUT -n --line-numbers | awk '/REJECT/{print $1; exit}')
   sudo iptables -I INPUT ${REJ:-5} -m state --state NEW -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save
   sudo iptables -L INPUT -n --line-numbers   # the dpt:8000 ACCEPT line must come before REJECT
   ```

## 2. Build and start

```bash
docker compose -f deploy/oracle/docker-compose.yml up -d --build
curl http://localhost:8000/health
```

The first build takes a while on one core (it compiles nothing, but pip downloads ~300 MB of ARM wheels).

## 3. Check speed and memory on the real hardware

```bash
docker stats --no-stream sentiment
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"texts":["Sensex jumps 900 points as IT stocks rally","Worst service ever, waited 2 hours 😡"]}'
```

Each result carries `tier` (`tfidf` or `transformer`), `content_type` and `uncertain`.

## 4. Updating

```bash
git pull && docker compose -f deploy/oracle/docker-compose.yml up -d --build
```

## Notes

- The API has no authentication. If the port is public, put it behind a reverse proxy with a key, or restrict the
  security-list rule to your own servers' IPs.
- `DISABLE_TRANSFORMER=1` in the compose file switches to the TF-IDF-only engine if you ever need more throughput.
- Nothing here talks to MongoDB. The service only answers `/predict` requests.

## Small instance without Docker (VM.Standard.E2.1.Micro: 1/8 OCPU, 1 GB RAM)

Building the Docker image can run out of memory on 1 GB. Run the engine directly in a virtualenv instead, in light mode
(English expert off, ~230 MB RSS). Tested on Ubuntu 22.04 Minimal (x86_64, Python 3.10).

```bash
# 2 GB swap as a safety net
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
sudo apt-get update && sudo apt-get install -y git python3-venv python3-pip curl
git clone --depth 1 -b claude/sentiment-engine-testing-efc0bb https://github.com/om0852/Sentiment-NLP.git
python3 -m venv ~/venv
~/venv/bin/pip install fastapi uvicorn pydantic numpy onnxruntime tokenizers scikit-learn joblib psutil python-multipart httpx
```

Then create `/etc/systemd/system/sentiment.service`:

```ini
[Unit]
Description=Sentiment engine API
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Sentiment-NLP
Environment=DISABLE_ENGLISH_EXPERT=1
Environment=TRANSFORMER_THREADS=1
Environment=OMP_NUM_THREADS=1
ExecStart=/home/ubuntu/venv/bin/uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now sentiment
curl localhost:8000/health
journalctl -u sentiment -f      # logs
```

Open port 8000 as in step 1. Updating: `cd ~/Sentiment-NLP && git pull && sudo systemctl restart sentiment`.

**Serve on port 80 instead** (no `:8000` in URLs; many VCNs already allow 80 in the default Security List):

```bash
sudo iptables -t nat -I PREROUTING -p tcp --dport 80 -j REDIRECT --to-ports 8000
sudo netfilter-persistent save
curl http://<public-ip>/health
```

The app keeps listening on 8000; the instance firewall must still allow 8000 (step 1.2), because the redirect happens
before that check. The API has no authentication: restrict the Security List source CIDR to your own servers.
On the Micro, light mode handles roughly 50–100k posts/day (1/8 OCPU baseline; it can burst higher for short periods).

## API keys (recommended for any public deployment)

When `SENTIMENT_API_KEYS` is set (comma-separated), every data endpoint (`/predict`, `/api/v1/...`, `/analyze/*`,
`/metrics`) requires `X-API-Key: <key>` or `Authorization: Bearer <key>`; others get `401`. `/health` and the dashboard
page stay public (the dashboard asks for the key once and remembers it in that browser). Unset = no authentication.

```bash
KEY="snt_$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "SENTIMENT_API_KEYS=$KEY" | sudo tee /etc/sentiment.env >/dev/null
sudo chmod 600 /etc/sentiment.env
# add under [Service] in /etc/systemd/system/sentiment.service:   EnvironmentFile=/etc/sentiment.env
sudo systemctl daemon-reload && sudo systemctl restart sentiment
curl -X POST http://<public-ip>/predict -H "Content-Type: application/json" -H "X-API-Key: $KEY" -d '{"texts":["hello"]}'
```

Give each client its own key (`SENTIMENT_API_KEYS=key-app1,key-app2`) so one can be revoked without touching the
others; restart the service after editing the file. For Docker, pass the same variable via `env_file:` in the compose file.
