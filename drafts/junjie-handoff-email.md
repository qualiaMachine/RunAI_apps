**To:** Junjie + student
**Subject:** Cluster access docs ready — storage update

---

Hi Junjie,

Quick update — the onboarding docs for the RunAI cluster are ready, so
your student can get started whenever they're free. The full New User
Guide lives here:

https://github.com/qualiaMachine/runai_apps#new-user-guide

The first step (Step 01) is the access flow, which is a little
two-stage so worth flagging up front:

1. Your student emails me and Mike at DoIT and tells us roughly what
   they're looking to do (notebook with a GPU, a model to fine-tune,
   an app to host, etc.) so we can pick the right project assignment
   and GPU quota.
2. We send back the portal URL for the cluster's RunAI web UI.
3. They connect to the **campus VPN**, open the URL, and log in once.
   The dashboard will look mostly empty — that first login is just to
   register their identity in RunAI so we can attach things to it.
4. They ping us back to say they've logged in. We then attach the
   project, GPU quota, and the cluster's `shared-models` Data Volume
   (pre-cached weights for Qwen, Jina, etc.) to their account.
5. Reload the dashboard and they're in — workspaces, inference
   endpoints, data sources, all clickable from the UI. No CLI
   required.

After that initial setup, every subsequent session is just "VPN +
URL + log in." Steps 02–05 of the guide walk through spinning up a
first workspace with a GPU, sharing a model as a vLLM endpoint, the
storage primitives, and example apps (OCR, RAG) they can copy from.

One note on storage — we just ordered some 1 TB NVMe drives to better
support larger-scale work and model training on the cluster. It'll be
a minute before we can comfortably handle the full hundreds of GB
you'd mentioned, but the docs will get your student into the cluster
with GPUs available right now, and they can ramp up from smaller-
scale runs in the meantime.

I'm waiting to hear back on a timeline for the expanded storage —
hoping it's only a couple of weeks. I'll pass that along as soon as I
know more.

Let me know if you or your student hit anything rough in the docs and
I'll get it fixed.

Thanks,
Chris
