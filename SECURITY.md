# Security Policy

## Supported versions

Security fixes land on the latest `main` and the newest release; pin a released version for stability
and upgrade for fixes.

## Reporting a vulnerability

Please report security issues **privately** via GitHub's **"Report a vulnerability"** button
(repository **Security** tab → Advisories) at
<https://github.com/GlycerinLOL/GaDRA/security/advisories>. If that is unavailable, open a minimal issue at
<https://github.com/GlycerinLOL/GaDRA/issues> asking for a private contact channel — **do not** put exploit
details in a public issue. We aim to acknowledge promptly and to coordinate a fix and disclosure.

In your report, please include: the affected version or commit, the environment (OS / Python / GPU), minimal
reproduction steps, and the impact you observed.

## Executing model-generated code (MBPP eval)

The optional MBPP evaluation (`examples/inference.py`, `task: mbpp`) **executes model-generated code** against
the reference tests, gated by `HF_ALLOW_CODE_EVAL=1`. This is arbitrary code execution by design (pass@1
scoring) and the script warns before it runs. Run it **only inside an isolated / containerized environment**
without credentials or sensitive network access. No other code path in this repository executes untrusted
input.

## Secrets

No secrets are stored in this repository. The GPT-judged eval (`bbcqa` / `tiebe`) reads `OPENAI_API_KEY` from
the environment only — never commit keys. The pip package (`pip install gadra`) is the method only and has no
network, eval, or credential surface.
