---
title: Payment Idempotency Keys & Webhook Hardening
summary: Architecture checklist for preventing duplicate charges and replay attacks on Stripe webhooks using unique event ledgers and HMAC signature checks.
sem: double billing, stripe replay, payment safety, webhook signature
---

# Payment Idempotency Keys & Webhook Hardening

Webhooks are delivered with at-least-once semantics. Production billing systems must record `event.id` in a unique `stripe_events` table inside the same transaction as ledger updates, returning HTTP 200 idempotently on retries. Key concepts evaluated: `Stripe-Idempotency`, `HMAC-SHA256`, `timingSafeEqual`, `Replay-Guard`.
