# Using the Shared Models (API key only)

For anyone who's been handed a **gateway API key** — a hackathon
participant, a lab member, someone building an app. You call the models
over HTTP from your own laptop, notebook, or server. You don't need a
Run:ai account, and you never log into the cluster.

The gateway is at **`https://llm-gw01.doit.wisc.edu/v1`** and speaks the
OpenAI API, so any client that lets you set a base URL works unmodified:
the `openai` Python package, `httr2` in R, LangChain, LlamaIndex, curl,
Postman.

| Model | Type | Use it for |
|-------|------|-----------|
| `qwen3.8-27b` | chat | General text: writing, reasoning, code, summarisation |
| `churro-3b` | chat + vision | OCR of historical documents and handwriting; send page images |
| `qwen3-vl-embedding-8b` | embeddings | 4096-dim vectors for search / RAG; handles text and images |

That's the catalogue as of September 2026. It changes: check this doc for
the latest, or once you're connected, run the command under
[Listing models](#listing-models) to see exactly what your key can call.
Want a model that isn't hosted? Talk to Chris — but note the whole
pilot currently runs on 96 GB of VRAM total, so it's a trade-off, not
an add.

> **A key gets you these models, not the cluster.** Running your own
> model, fine-tuning, or getting a GPU workspace needs a Run:ai account,
> which is a separate request rather than something a key upgrades into.
> Available on request, but GPU time isn't guaranteed — see
> [Scope](#scope).

## PowerShell and bash

Commands below come in pairs: **PowerShell first, then bash/zsh**. Three
differences account for most of the confusion:

| | PowerShell | bash / zsh |
|---|---|---|
| Set a variable | `$env:NAME = "value"` | `export NAME=value` |
| Use a variable | `$env:NAME` | `$NAME` |
| `curl` | an *alias* for `Invoke-WebRequest` — use `Invoke-RestMethod`, or `curl.exe` for the real thing | real curl |

PowerShell keeps environment variables in a separate `env:` namespace.
`$OPENAI_API_KEY` without that prefix is an ordinary PowerShell variable
that doesn't exist — it evaluates to empty silently, so your header
becomes `"Bearer "` and the gateway answers *"Malformed API Key"*. Bash
has no such split, which is why the two spellings differ.

> **On Windows, use PowerShell rather than Git Bash** for anything
> involving `op`. The 1Password desktop integration refuses connections
> from Git Bash and reports *"account is not signed in"* no matter how the
> app is configured. Git Bash is fine for everything else.

## Network access

Two things must be true before any of this works:

1. **You're on GlobalProtect**, including from on-campus wifi.
2. **Your NetID has been added to the firewall rule.** Access to the
   gateway is granted per person at the campus firewall, so being on the
   VPN isn't enough on its own. Chris arranges this when you request a
   key — but it's a manual step with a lead time, so if you've just been
   given a key, check it's been done before assuming something's broken.

Neither failure announces itself. Both look like a hang or
`Unable to connect to the remote server`.

If you're on the VPN and still can't connect, find out which layer is
failing before assuming your key is wrong:

```powershell
# PowerShell
Resolve-DnsName llm-gw01.doit.wisc.edu
Test-NetConnection llm-gw01.doit.wisc.edu -Port 443
```

```bash
# bash / zsh — 401 means you reached the gateway and it wants a key,
# which is the result you want here. A hang or connection error is
# VPN or firewall.
curl -s -o /dev/null -w "%{http_code}\n" --max-time 10 \
  https://llm-gw01.doit.wisc.edu/v1/models
```

| Result | Meaning |
|---|---|
| DNS fails | Not on the VPN, or a DNS problem — reconnect GlobalProtect |
| `PingSucceeded: True`, `TcpTestSucceeded: False` | **Almost always your NetID isn't in the firewall rule yet.** The host is reachable, but the firewall drops your connection before the gateway ever sees it. Nothing you can fix and nothing to do with your key — message Chris with your NetID and the full `Test-NetConnection` output |
| `TcpTestSucceeded: True` | Network is fine; the problem is your key or your request — see the troubleshooting table at the bottom |

## Step 1 — Get your key, and save it

**Don't have a key yet?** Request one through the
[Badger Brain access form](https://forms.gle/vkcLzApNrX7KbkTP9). It asks
which group or project you're with, so your usage lands under the right
team, and your NetID, so the firewall rule can be updated. The key
arrives as a **1Password share link** — that's the only way keys go out
here, so if someone offers to paste one into Teams or an email, ask for a
share link instead.

The link is locked to your `@wisc.edu` address and expires, so open it
reasonably promptly. If it's expired or you lose it, ask for another —
re-sharing is trivial and far better than working around it.

When you open it, **save the item into your UW-Madison 1Password
account** — every NetID has one; DoIT's KB covers
[receiving shared items](https://kb.wisc.edu/security/144574). From then
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

**If `op` won't authenticate**, open the item in the 1Password app or
browser extension, copy the key, and set the variable for this session
only:

```powershell
$env:OPENAI_API_KEY = "sk-..."      # PowerShell
```

```bash
export OPENAI_API_KEY=sk-...        # bash / zsh
```

Either way, confirm it landed before moving on — this prints only the
first few characters, so it's safe on a shared screen:

```powershell
$env:OPENAI_API_KEY.Substring(0,6)     # PowerShell — expect sk-...
```

```bash
echo ${OPENAI_API_KEY:0:6}             # bash / zsh — expect sk-...
```

An error or blank line means it didn't take. Still don't put the key in
your code — one in a notebook cell gets committed to git eventually.

## Step 3 — Start your tools from that same terminal

`python`, `jupyter lab`, `R`, `rstudio`, `code .` — launch whichever you
use **from the shell where you just set the variable**.

This is the step people trip on. The variable lives in that one shell
session. A notebook opened from the Start menu or a desktop icon won't
see it, and the client reports a missing API key — which reads like a
broken key rather than a missing step. It's also gone when you close the
terminal, so Step 2 repeats each session unless you add it to your shell
profile.

## Listing models

The table at the top is a snapshot. To see the current list:

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

If this errors in PowerShell, it's almost always one of the two
differences in [PowerShell and bash](#powershell-and-bash).

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

> **`Error: nzchar(key) is not TRUE` means `Sys.getenv("OPENAI_API_KEY")`
> came back empty** — RStudio was started without inheriting your shell
> environment (Step 3). Your key is fine. Two ways to fix it:
>
> **Persist it** (what most RStudio users want):
>
> ```r
> usethis::edit_r_environ()      # opens ~/.Renviron
> ```
>
> Add one line — no quotes, no `export`:
>
> ```
> OPENAI_API_KEY=sk-...
> ```
>
> Then **Session → Restart R**. `.Renviron` is read only at startup, so
> without the restart it still looks empty. Check with
> `nchar(Sys.getenv("OPENAI_API_KEY"))`.
>
> Call `edit_r_environ()` with no arguments so it edits the **user-level**
> file in your home directory. `edit_r_environ("project")` writes one into
> the project folder, where it gets committed. And never put the key in a
> `.R` script.
>
> **Just for this session**, if you'd rather not keep the key on disk —
> pops a dialog, you paste into it, and nothing lands in your console
> history or a file:
>
> ```r
> Sys.setenv(OPENAI_API_KEY = rstudioapi::askForSecret("OPENAI_API_KEY"))
> ```
>
> Gone when R restarts. `.Renviron` is plaintext on
> disk, so this is the better choice on a shared machine.

### Keeping the key out of `.Renviron`

There is no 1Password SDK for R — the official ones are Go, JS and Python
— so the options are the CLI or the OS keychain.

**Launch RStudio through `op run`.** 1Password resolves the reference at
launch and RStudio inherits the real value; nothing is written to disk.
Put a *reference* (not a key) in `rstudio.env`:

```
OPENAI_API_KEY=op://<vault>/<your item>/credential
```

```powershell
# Windows PowerShell / Windows Terminal — close RStudio first
op run --env-file=.\rstudio.env -- "C:\Program Files\RStudio\rstudio.exe"
```

```bash
# macOS
op run --env-file=./rstudio.env -- open -a RStudio
```

That file is safe to commit — it contains no secret.

> **Not from RStudio's Terminal pane.** That tab is a separate process
> from the R console, so variables set there never reach `Sys.getenv()`,
> and relaunching RStudio from inside RStudio does nothing for the session
> you're in. `op run` has to come first, from a real terminal, with
> RStudio closed. If you're already in a session and don't want to
> restart, use `keyring` below instead.

**`keyring`**, if `op` won't cooperate. Uses Windows Credential Manager
or the macOS Keychain, so still no plaintext file:

```r
keyring::key_set("litellm")                                  # once, prompts
Sys.setenv(OPENAI_API_KEY = keyring::key_get("litellm"))     # each session
```

**Calling `op` from inside R** (`system2("op", c("read", "op://..."))`)
usually fails with *"account is not signed in"* — the desktop integration
authorises by calling application, and `rsession` isn't one it accepts.
One line to try, but don't plan around it.

## Cold starts

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

## Use the gateway URL, not model hostnames

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

## Troubleshooting

| What you see | What it usually means |
|---|---|
| Hang, or `Unable to connect to the remote server` | Not on GlobalProtect, or your NetID isn't in the firewall rule yet — see [Network access](#network-access) to tell which |
| `Malformed API Key ... Ensure Key has 'Bearer ' prefix` | Your key never made it into the header. In PowerShell, check you wrote `$env:OPENAI_API_KEY` and not `$OPENAI_API_KEY`, and that any `$headers` variable was built *after* setting it — it captures the value at assignment. In Python, restart the process after setting the variable |
| `Cannot bind parameter 'Headers'` or `A drive with the name 'https' does not exist` | You ran a bash `curl` command in PowerShell, where `curl` aliases `Invoke-WebRequest`. Use `Invoke-RestMethod`, or `curl.exe` |
| `Invalid proxy key` / 401 | Wrong key, or it expired — ask for a new share link |
| `404 ... Model Group=...` | Model name typo, or it was removed. Re-check `/v1/models` |
| Timeout on the first call | Cold start — raise your client timeout to 300s and retry |
| 429 | Rate limited. Back off and retry; if it's persistent, ask for a higher limit |

Lost your key? Request a replacement through the
[access form](https://forms.gle/vkcLzApNrX7KbkTP9) — it's a one-minute
job on our side. Anything else, or a model that's consistently
unavailable: contact Chris, and include the model name and the exact
error text; the error body from the gateway says which layer failed.

## Scope

A gateway key lets you **call** the models in the catalogue. It doesn't
give you a Run:ai account, a GPU, storage on the cluster, or the ability
to host your own model.

**Cluster access is available on request, not by default.** If you want
to explore fine-tuning or run something the gateway can't do, talk to
Chris. The pilot has two GPUs for at least the next six months, and the
shared endpoints already live on them, so GPU time for your own workload
can't be promised and may well not materialise. Worth asking; not worth
planning a project around.

**Asking for another model is fine.** It's a config change plus a
pipeline deploy on our side, not a rebuild of anything. A model already
on the cluster can usually be exposed within minutes; one that has to be
downloaded and given its own GPU workload takes longer. Either way, ask
sooner rather than at the moment you need it.

If you do get an account, the
[New User Guide](../README.md#new-user-guide) describes what's involved,
starting with [00 Overview](00-overview.md).

This is a **pilot**. Read the
[Usage Policy](usage-policy.md) before putting real data through it —
public data only, and no availability guarantees.
