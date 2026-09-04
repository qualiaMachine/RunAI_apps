# Using the Shared Models (API key only)

For anyone who's been handed a **gateway API key** — a hackathon
participant, a lab member, someone building an app. You call the models
over HTTP from your own laptop, notebook, or server. You don't need a
Run:ai account, and you never log into the cluster.

> If instead you want to *run* your own model, fine-tune, or get a GPU
> workspace, that's a different path — start at the
> [New User Guide](../README.md#new-user-guide).

## What you have

| | |
|---|---|
| **Base URL** | `https://llm-gw01.doit.wisc.edu/v1` |
| **API key** | your `sk-…` — ask Chris (endemann@wisc.edu); it arrives as a 1Password share link |

The gateway speaks the **OpenAI API**. Any client library that lets you
change the base URL works unmodified — the `openai` Python package,
`httr2` in R, LangChain, LlamaIndex, curl, Postman.

## Before anything else: the VPN

You must be on **GlobalProtect** to reach the gateway, including from
on-campus wifi. Nothing below works without it, and the failure looks
like a hang or `Unable to connect to the remote server` rather than a
clear message.

If you're on the VPN and still can't connect, find out which layer is
failing before assuming your key is wrong:

```powershell
Resolve-DnsName llm-gw01.doit.wisc.edu
Test-NetConnection llm-gw01.doit.wisc.edu -Port 443
```

| Result | Meaning |
|---|---|
| DNS fails | Not on the VPN, or a DNS problem — reconnect GlobalProtect |
| `PingSucceeded: True`, `TcpTestSucceeded: False` | **The host is reachable but port 443 is blocked for your VPN address.** Not something you can fix — send Chris the full `Test-NetConnection` output, including `SourceAddress`, so the firewall allowlist can be checked against your VPN range |
| `TcpTestSucceeded: True` | Network is fine; the problem is your key or your request — see the troubleshooting table at the bottom |

That middle case is worth knowing about: access is allowed per VPN
address range, so it's possible to be properly connected and still be
refused.

## Step 1 — Get your key, and save it

**Don't have a key yet?** Ask Chris (endemann@wisc.edu) for one. Tell him
which group or project you're with, so your usage lands under the right
team. He'll send it as a **1Password share link** — that's the only way
keys go out here, so if someone offers to paste one into Teams or an
email, ask for a share link instead.

The link is locked to your `@wisc.edu` address and expires, so open it
reasonably promptly. If it's expired or you lose it, ask for another —
re-sharing is trivial and far better than working around it.

When you open it, **save the item into your own 1Password**. From then
on it's yours. Don't paste the key into a file, a notebook, or a chat
message — it identifies you, and everything you run is recorded against
it.

## Step 2 — Load it into your shell

Best done with the **1Password CLI** (`op`), which reads the key straight
out of your vault so it never appears in your shell history or your code.

Install it once:

```powershell
# Windows
winget install AgileBits.1Password.CLI
```

```bash
# macOS
brew install 1password-cli
```

Full instructions, including Linux:
<https://developer.1password.com/docs/cli/get-started/>

Then enable the desktop integration, which is what lets `op` unlock
without a password: **1Password app → Settings → Developer → "Integrate
with 1Password CLI"**, then quit and reopen the app. Check it works:

```
op whoami
```

If that prints your account, you're set. Now load the key at the start of
each session:

```powershell
# PowerShell — replace with your item's name
$env:OPENAI_API_KEY = op read "op://Private/wams_bbadger/credential"
```

```bash
# bash / zsh
export OPENAI_API_KEY=$(op read 'op://Private/wams_bbadger/credential')
```

**No 1Password CLI, or `op` won't authenticate?** You don't need it —
the share link works on its own. Open the item in 1Password (or the
browser extension), copy the key, and set the variable for this session
only:

```powershell
$env:OPENAI_API_KEY = "sk-..."      # PowerShell
```

```bash
export OPENAI_API_KEY=sk-...        # bash / zsh
```

Still don't put it in your code — a key in a notebook cell gets committed
to git eventually.

> You do **not** need a 1Password account to open a share link, which is
> why this works for collaborators outside UW-Madison. UW-Madison staff
> and students do have 1Password available; DoIT's KB covers accounts and
> [item sharing](https://kb.wisc.edu/security/144574).

## Step 3 — Start your tools from that same terminal

`python`, `jupyter lab`, `R`, `rstudio`, `code .` — launch whichever you
use **from the shell where you just set the variable**.

This is the step people trip on. The variable lives in that one shell
session. A notebook opened from the Start menu or a desktop icon won't
see it, and the client reports a missing API key — which reads like a
broken key rather than a missing step. It's also gone when you close the
terminal, so Step 2 repeats each session unless you add it to your shell
profile.

## Which models can I call?

Ask the gateway rather than trusting a list in a doc — the catalogue
changes:

```powershell
# PowerShell
(Invoke-RestMethod https://llm-gw01.doit.wisc.edu/v1/models `
   -Headers @{ Authorization = "Bearer $env:OPENAI_API_KEY" }).data.id
```

```bash
# bash / zsh
curl -s https://llm-gw01.doit.wisc.edu/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

> **Two PowerShell traps**, and they produce confusing errors rather than
> clear ones:
> - **`curl` is an alias for `Invoke-WebRequest`**, so bash-style `curl -H ...`
>   fails with *"Cannot bind parameter 'Headers'"*. Use `Invoke-RestMethod`
>   as above, or spell it `curl.exe` to get the real curl.
> - **The variable is `$env:OPENAI_API_KEY`, not `$OPENAI_API_KEY`.** The
>   bash spelling is simply undefined in PowerShell, so your header becomes
>   `"Bearer "` and the gateway replies *"Malformed API Key"*.

As of September 2026:

| Model | Type | Use it for |
|-------|------|-----------|
| `qwen3.8-27b` | chat | General text: writing, reasoning, code, summarisation |
| `churro-3b` | chat + vision | OCR of historical documents and handwriting; send page images |
| `qwen3-vl-embedding-8b` | embeddings | 4096-dim vectors for search / RAG; handles text and images |

## Python

The `openai` package picks up `OPENAI_API_KEY` on its own, so the key
never appears in your code:

```python
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="https://llm-gw01.doit.wisc.edu/v1", timeout=300)

resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain PCA in two sentences."},
    ],
)
print(resp.choices[0].message.content)
```

**Streaming**, so long answers appear as they're generated:

```python
stream = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Write a haiku about badgers."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Embeddings** — one call, many texts:

```python
texts = ["the mitochondria is the powerhouse", "badgers dig burrows"]
resp = client.embeddings.create(model="qwen3-vl-embedding-8b", input=texts)
vectors = [d.embedding for d in resp.data]
print(len(vectors), "vectors of", len(vectors[0]), "dimensions")
```

**Images** (CHURRO for document OCR) — images go inline as base64:

```python
import base64

with open("scan.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="churro-3b",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Transcribe this page."},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}],
)
print(resp.choices[0].message.content)
```

## R

Two reasonable paths:

| | Use when |
|---|---|
| **`ellmer`** | Chat. Handles conversation state, streaming, tool calling and structured output. **No embeddings support.** |
| **`httr2`** | Embeddings, or when you want no LLM dependency and full control over the request |

### ellmer (chat)

```r
# install.packages("ellmer")
library(ellmer)

chat <- chat_openai_compatible(
  base_url = "https://llm-gw01.doit.wisc.edu/v1",
  model    = "qwen3.8-27b",
  api_key  = Sys.getenv("OPENAI_API_KEY")
)

chat$chat("Explain PCA in two sentences.")

# the object keeps the conversation, so follow-ups have context
chat$chat("Now give an example with gene expression data.")
```

ellmer moves quickly — check `?chat_openai_compatible` if an argument
name doesn't match. It also ships `chat_vllm()`, a thin wrapper over the
same thing that reads `VLLM_API_KEY` instead; either works against the
gateway.

### httr2 (anything, including embeddings)

```r
# install.packages("httr2")
library(httr2)

gateway <- "https://llm-gw01.doit.wisc.edu/v1"
key     <- Sys.getenv("OPENAI_API_KEY")
stopifnot(nzchar(key))   # fails loudly if RStudio didn't inherit it

chat <- function(prompt, model = "qwen3.8-27b") {
  resp <- request(paste0(gateway, "/chat/completions")) |>
    req_auth_bearer_token(key) |>
    req_timeout(300) |>                       # cold starts, see below
    req_body_json(list(
      model    = model,
      messages = list(list(role = "user", content = prompt))
    )) |>
    req_perform() |>
    resp_body_json()

  resp$choices[[1]]$message$content
}

cat(chat("Explain PCA in two sentences."))
```

**Embeddings** in R:

```r
embed <- function(texts, model = "qwen3-vl-embedding-8b") {
  resp <- request(paste0(gateway, "/embeddings")) |>
    req_auth_bearer_token(key) |>
    req_timeout(300) |>
    req_body_json(list(model = model, input = as.list(texts))) |>
    req_perform() |>
    resp_body_json()

  # one row per input text
  do.call(rbind, lapply(resp$data, \(d) unlist(d$embedding)))
}

m <- embed(c("badgers dig burrows", "the mitochondria is the powerhouse"))
dim(m)   # 2 x 4096
```

> **If `Sys.getenv("OPENAI_API_KEY")` comes back empty in RStudio**, it
> was launched without inheriting your shell environment (Step 3). Either
> start RStudio from the terminal where you set the variable, or add the
> line to `~/.Renviron` — never to your `.R` script.

## The first call can take a couple of minutes

Some models are configured to release their GPU when idle. The first
request after a quiet period **waits while a GPU replica starts** —
typically 90 seconds or so. The connection is held open the whole time;
nothing is lost.

Practical consequences:

- **Set a generous client timeout.** The examples above use 300 seconds.
  A default 30- or 60-second timeout will give up mid-startup and look
  like a failure.
- **Don't treat a slow first call as broken.** Try a second request
  before reporting a problem — if the second is fast, that was a cold
  start working exactly as designed.
- `qwen3.8-27b` stays warm; `churro-3b` and `qwen3-vl-embedding-8b`
  are the ones that sleep.

## Always call the gateway URL

You may come across a direct model hostname ending in
`deepthought.doit.wisc.edu`. **Don't use it.** Those answer without a key,
so calls that bypass the gateway don't appear in usage reporting — and
unattributed traffic is what gets a pilot's capacity questioned. Use
`https://llm-gw01.doit.wisc.edu/v1` for everything.

## Check your own usage

No login needed — your key can query itself:

```powershell
# PowerShell
Invoke-RestMethod https://llm-gw01.doit.wisc.edu/key/info `
  -Headers @{ Authorization = "Bearer $env:OPENAI_API_KEY" }
```

```bash
# bash / zsh
curl -s https://llm-gw01.doit.wisc.edu/key/info \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

That shows your key's limits and what it's spent so far.

## When something breaks

| What you see | What it usually means |
|---|---|
| Hang, or DNS/connection error | Not on GlobalProtect |
| `Malformed API Key ... Ensure Key has 'Bearer ' prefix` | Your key never made it into the header. In PowerShell, check you wrote `$env:OPENAI_API_KEY` and not `$OPENAI_API_KEY`, and that any `$headers` variable was built *after* setting it — it captures the value at assignment. In Python, restart the process after setting the variable |
| `Cannot bind parameter 'Headers'` or `A drive with the name 'https' does not exist` | You ran a bash `curl` command in PowerShell, where `curl` aliases `Invoke-WebRequest`. Use `Invoke-RestMethod`, or `curl.exe` |
| `Invalid proxy key` / 401 | Wrong key, or it expired — ask for a new share link |
| `404 ... Model Group=...` | Model name typo, or it was removed. Re-check `/v1/models` |
| Timeout on the first call | Cold start — raise your client timeout to 300s and retry |
| 429 | Rate limited. Back off and retry; if it's persistent, ask for a higher limit |

Anything else, a model that's consistently unavailable, or a key you've
lost: contact Chris (endemann@wisc.edu) — a replacement key is a
one-minute job. Include the model name and the exact error text — the error body
from the gateway says which layer failed.

## What this doesn't cover

A gateway key lets you **call** the shared models. It doesn't give you a
Run:ai account, a GPU, storage on the cluster, or the ability to host
your own model. If you need those, that's the
[New User Guide](../README.md#new-user-guide) — start with
[00 Overview](00-overview.md) and mention what you're trying to do.

Also worth knowing: this is a **pilot**. Read the
[Usage Policy](usage-policy.md) before putting real data through it —
public data only, and no availability guarantees.
