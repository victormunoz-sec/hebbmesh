---
title: Edge WAF & Anti-Scraper Defense for Serverless Databases
summary: Blocking datacenter scrapers and fuzzers at Cloudflare Edge before they trigger D1 row-read explosions or SIEM log poisoning.
sem: bot attack, crawler defense, d1 reads, cloudflare waf
---

# Edge WAF & Anti-Scraper Defense for Serverless Databases

Unverified datacenter scrapers issuing `PUT` probes and `/.env` scans can inflate serverless database reads 40x overnight. Deploying custom Edge WAF rules, rate limits (`>30 reqs/10s`), strict `Cache-Control` exclusions for session cookies, and Zod validation on public telemetry endpoints cuts origin load by >90%. Key concepts evaluated: `Edge-WAF`, `Cache-Poisoning`, `SIEM-Poisoning`, `Rate-Limit`.
