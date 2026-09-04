#!/usr/bin/env python3
"""Provision LiteLLM gateway teams and per-user virtual keys from a roster.

Reads a CSV of people, creates any teams that don't exist yet, mints one
virtual key per person, files each key in 1Password, and emits a
recipient-restricted share link per person. Nothing is created by hand in
the dashboard.

The gateway's model catalogue is git-managed in the `se-litellm` repo
(`store_model_in_db: false`). Teams, users and keys are the opposite kind
of object -- rows in the proxy's Postgres -- so they are created through
the running proxy's API, which is what this script does. There is nothing
here to commit.

Usage (Git Bash, Windows or Linux/Mac):

    # 1. write a roster (see --example)
    python scripts/provision_gateway_keys.py --example > roster.csv

    # 2. see what would happen -- no changes, no keys minted
    python scripts/provision_gateway_keys.py roster.csv

    # 3. do it
    python scripts/provision_gateway_keys.py roster.csv --apply

Re-running is safe: anyone who already has a key item in the vault is
skipped, so you can add rows to the roster and re-run to onboard only the
new people.

Requires: python 3.9+, the 1Password CLI (`op`) signed in. No pip installs.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_GATEWAY = "https://llm-gw01.doit.wisc.edu"
DEFAULT_VAULT = "DoIT-AI"
DEFAULT_MASTER_KEY_REF = "op://DoIT-AI/LiteLLM gateway/master key"

EXAMPLE_CSV = """\
netid,team,email,rpm_limit,duration
bbadger,wams,bbadger@wisc.edu,,
osky,wams,osky@wisc.edu,,
astudent,marathon-team-07,astudent@wisc.edu,,7d
bstudent,marathon-team-07,bstudent@wisc.edu,,7d
"""

COLUMNS = "netid,team,email,rpm_limit,duration"


class Fatal(Exception):
    pass


# --------------------------------------------------------------------------
# 1Password
# --------------------------------------------------------------------------

def op(*args, check=True):
    """Run the 1Password CLI and return (stdout, returncode, stderr)."""
    try:
        r = subprocess.run(["op", *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise Fatal(
            "1Password CLI not found. Install it and run `op signin`:\n"
            "  Windows: winget install AgileBits.1Password.CLI\n"
            "  macOS:   brew install 1password-cli"
        )
    if check and r.returncode != 0:
        raise Fatal(f"op {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip(), r.returncode, r.stderr.strip()


def op_check_signin():
    """Preflight. Reports op's own error rather than guessing at the cause."""
    out, rc, err = op("whoami", check=False)
    if rc == 0:
        return
    raise Fatal(
        "`op whoami` failed. Its output was:\n"
        f"  {err or '(no error output)'}\n\n"
        "Run `op whoami` yourself in this same shell — if it works there but\n"
        "not here, the CLI is authenticating interactively and can't prompt\n"
        "from inside a script. Common fixes:\n"
        "  - 1Password app > Settings > Developer > 'Integrate with 1Password\n"
        "    CLI', then unlock the app. `eval $(op signin)` is a no-op when\n"
        "    the desktop integration is on, so it won't help by itself.\n"
        "  - Or use a service account, which needs no interactive unlock:\n"
        "      export OP_SERVICE_ACCOUNT_TOKEN=ops_...\n"
    )


def op_item_exists(title, vault):
    _, rc, _err = op("item", "get", title, "--vault", vault, check=False)
    return rc == 0


def op_item_create(title, vault, key, netid, gateway, tags):
    op(
        "item", "create",
        "--category", "API Credential",
        "--vault", vault,
        "--title", title,
        "--tags", ",".join(tags),
        f"credential={key}",
        f"username={netid}",
        f"base url[text]={gateway}/v1",
    )


def op_item_share(title, vault, email, expires):
    out, _rc, _err = op(
        "item", "share", title,
        "--vault", vault,
        "--emails", email,
        "--expires-in", expires,
        "--view-once",
    )
    # `op item share` prints the URL, sometimes with surrounding chatter.
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("https://"):
            return line
    return out


# --------------------------------------------------------------------------
# LiteLLM proxy API
# --------------------------------------------------------------------------

def api(gateway, master_key, path, payload=None, method=None):
    url = f"{gateway.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data else "GET"),
        headers={
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise Fatal(f"{method or 'GET'} {path} -> HTTP {e.code}\n{body}")
    except urllib.error.URLError as e:
        raise Fatal(
            f"Cannot reach {url}: {e.reason}\n"
            "Are you on GlobalProtect (campus VPN)?"
        )


def list_teams(gateway, master_key):
    """Return {team_alias: team_id}. Tolerates either response shape."""
    raw = api(gateway, master_key, "/team/list")
    teams = raw.get("teams", raw) if isinstance(raw, dict) else raw
    if not isinstance(teams, list):
        raise Fatal(f"Unexpected /team/list response: {str(raw)[:300]}")
    out = {}
    for t in teams:
        alias = t.get("team_alias") or t.get("team_id")
        if alias:
            out[alias] = t["team_id"]
    return out


def create_team(gateway, master_key, alias):
    r = api(gateway, master_key, "/team/new", {"team_alias": alias})
    team_id = r.get("team_id")
    if not team_id:
        raise Fatal(f"/team/new returned no team_id: {str(r)[:300]}")
    return team_id


def generate_key(gateway, master_key, netid, team_id, rpm, duration, team):
    payload = {
        "key_alias": f"{team}-{netid}",
        "team_id": team_id,
        "user_id": netid,
        "metadata": {"team": team, "netid": netid},
    }
    if rpm:
        payload["rpm_limit"] = int(rpm)
    if duration:
        payload["duration"] = duration
    # No "models" field: keys inherit the team's access, which is the whole
    # catalogue. Naming an access group that isn't in litellm_config.yaml
    # produces a key that is rejected on every call.
    r = api(gateway, master_key, "/key/generate", payload)
    key = r.get("key")
    if not key:
        raise Fatal(f"/key/generate returned no key: {str(r)[:300]}")
    return key


# --------------------------------------------------------------------------
# Roster
# --------------------------------------------------------------------------

def read_roster(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise Fatal(f"{path} has no data rows.")
    missing = {"netid", "team", "email"} - set(rows[0])
    if missing:
        raise Fatal(
            f"{path} is missing column(s): {', '.join(sorted(missing))}\n"
            f"Expected header: {COLUMNS}"
        )
    for i, r in enumerate(rows, start=2):
        for col in ("netid", "team", "email"):
            if not (r.get(col) or "").strip():
                raise Fatal(f"{path} line {i}: '{col}' is empty.")
    return rows


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Provision LiteLLM teams and per-user keys from a roster CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("roster", nargs="?", help=f"CSV with header: {COLUMNS}")
    p.add_argument("--apply", action="store_true",
                   help="actually create things (default: dry run)")
    p.add_argument("--gateway", default=os.environ.get("LITELLM_GATEWAY", DEFAULT_GATEWAY))
    p.add_argument("--vault", default=os.environ.get("OP_VAULT", DEFAULT_VAULT),
                   help="1Password vault to file keys in")
    p.add_argument("--master-key-ref", default=DEFAULT_MASTER_KEY_REF,
                   help="op:// reference to the gateway master key")
    p.add_argument("--expires-in", default="14d",
                   help="share-link lifetime (default 14d)")
    p.add_argument("--out", default="share-links.csv",
                   help="where to write netid,share_link (default share-links.csv)")
    p.add_argument("--example", action="store_true",
                   help="print an example roster CSV and exit")
    args = p.parse_args()

    if args.example:
        # Prefer the committed template so the two can't drift; the
        # embedded copy keeps --example working if the script is run
        # on its own (e.g. copied onto a workspace).
        template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "roster.example.csv")
        try:
            with open(template, encoding="utf-8") as f:
                print(f.read(), end="")
        except OSError:
            print(EXAMPLE_CSV, end="")
        return 0
    if not args.roster:
        p.error("roster CSV required (or --example)")

    rows = read_roster(args.roster)
    op_check_signin()

    master_key = op("read", args.master_key_ref)[0]
    if not master_key:
        raise Fatal(f"No value at {args.master_key_ref}")

    teams = list_teams(args.gateway, master_key)
    wanted = sorted({r["team"].strip() for r in rows})
    new_teams = [t for t in wanted if t not in teams]

    # ---- plan ----
    print(f"Gateway : {args.gateway}")
    print(f"Vault   : {args.vault}")
    print(f"Roster  : {args.roster} ({len(rows)} people, {len(wanted)} teams)")
    print()

    if new_teams:
        print("Teams to create:")
        for t in new_teams:
            print(f"  + {t}")
    else:
        print("Teams: all exist already.")

    todo, skipped = [], []
    for r in rows:
        title = f"LiteLLM — {r['netid'].strip()}"
        (skipped if op_item_exists(title, args.vault) else todo).append(r)

    print(f"\nKeys to mint: {len(todo)}")
    for r in todo:
        print(f"  + {r['netid']:<12} team={r['team']:<20} rpm={r.get('rpm_limit') or 'default'}"
              f" duration={r.get('duration') or 'none'}")
    if skipped:
        print(f"\nAlready have a key item in {args.vault} (skipping): "
              + ", ".join(r["netid"] for r in skipped))

    if not args.apply:
        print("\nDry run. Re-run with --apply to create these.")
        return 0
    if not todo and not new_teams:
        print("\nNothing to do.")
        return 0

    # ---- apply ----
    print()
    for t in new_teams:
        teams[t] = create_team(args.gateway, master_key, t)
        print(f"created team {t} ({teams[t]})")

    links = []
    for r in todo:
        netid, team = r["netid"].strip(), r["team"].strip()
        title = f"LiteLLM — {netid}"
        key = generate_key(
            args.gateway, master_key, netid, teams[team],
            (r.get("rpm_limit") or "").strip(), (r.get("duration") or "").strip(), team,
        )
        op_item_create(title, args.vault, key, netid, args.gateway,
                       ["litellm", team])
        del key  # the value stays in 1Password, not in this process
        link = op_item_share(title, args.vault, r["email"].strip(), args.expires_in)
        links.append((netid, r["email"].strip(), link))
        print(f"minted + filed + shared: {netid}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["netid", "email", "share_link"])
        w.writerows(links)

    print(f"\n{len(links)} key(s) issued. Share links -> {args.out}")
    print("That file holds links, not keys: each is restricted to one address,")
    print(f"single-view, and expires in {args.expires_in}. Safe to email.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fatal as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
