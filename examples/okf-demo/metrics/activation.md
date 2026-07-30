---
type: Metric
title: Activation
description: Our internal definition of an activated account.
tags: [metrics, internal]
generated: { by: research_agent/gemini-3.1-flash, at: 2026-07-19T09:00:00Z }
status: stable
x_verification:
  by: recommend-trust-layer/0.1
  at: 2026-07-29T10:59:05Z
  claims:
    - claim: "MCP clients are a type of software or protocol that can be connected to an account."
      verdict: inconclusive
      label: DISPUTED
      truth_score: 63
      confidence: 0.11
      sources:
        - https://modelcontextprotocol.io/docs/learn/client-concepts.md
        - https://modelcontextprotocol.io/docs/getting-started/intro
        - https://modelcontextprotocol.io/specification/2025-03-26/architecture/index
---

# Definition

An account counts as **activated** when it has run at least 3 checks within 7
days of signup and connected at least one MCP client. Weekly activation is the
share of that week's signups that activate.
