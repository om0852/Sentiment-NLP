# Deploying on Oracle Cloud (Always Free, ARM, 1 OCPU / 6 GB)

The engine needs ~280 MB RAM and, with the 12-layer transformer, ~100 ms of CPU per real post (long news posts use
all 128 tokens), so one core handles roughly **10 posts/second (~870k posts/day)**. That was measured on an Intel
laptop core; Ampere cores differ, so check on your instance with step 3. If you need more throughput, set
`DISABLE_TRANSFORMER=1` (TF-IDF tier only: ~1 ms/post, much lower accuracy).

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
2. On the instance's own firewall:
   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save
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
