# Running open-rlcd on a server

For the platform team that runs `rlcd serve` for other people. The model itself is trained elsewhere (a laptop,
[teammate-quickstart.md](teammate-quickstart.md)) and arrives here as a bundle; this page is about the container,
the keys, the disk, and what to watch. What the API means is in [control-layer.md](control-layer.md) and
[contract.md](contract.md).

The short version:

* one container, one process, one directory of state (`RLCD_HOME`);
* **one serving process per home** — no `--workers`, no replicas on a shared volume (see [Concurrency](#concurrency));
* TLS and the network boundary belong to a proxy in front; the service checks two bearer keys;
* back up the home; everything else is in the image or in your project repository.

## What runs where

| piece | where | notes |
|---|---|---|
| the image | `Dockerfile` | Python 3.12 slim, dependencies pinned by `uv.lock`, runs as uid/gid `10001`, no shell tools beyond the base image |
| the service | `rlcd serve --host 0.0.0.0 --port 8095` | one uvicorn process; requests are answered on its thread pool |
| state | `/var/lib/rlcd` (`RLCD_HOME`, a declared volume) | models, receipts, feedback, the Bernoulli ledger, caches |
| configuration | `/project/rlcd.yaml` (`RLCD_PROJECT`, mounted read-only) | policy, rollout, routes, batteries, pack settings; read at start |
| TLS | Caddy (`compose.yaml`, `deploy/Caddyfile`) | the only published ports are the proxy's 443 and 80 |

The image contains the code, its locked dependencies and the synthetic demo bundle. It contains no keys, no model
of yours and no data: `.dockerignore` is an allow-list, so a local `var/` or `.env` cannot reach the build.

### Build

`compose.yaml` builds the image itself and runs it as `open-rlcd:local` (or as `RLCD_IMAGE`, when that is set).
It takes the extras from `RLCD_EXTRAS`, in the environment or in the `.env` file next to `compose.yaml`:

```bash
docker compose build                                  # core only: about 460 MB
RLCD_EXTRAS=documents docker compose build            # + onnxruntime, faiss, tokenizers, pypdf: about 615 MB
```

Without compose, `docker build -t open-rlcd:local --build-arg RLCD_EXTRAS=documents .` builds the same image. An
image built under another name is not the one compose runs.

`RLCD_EXTRAS` is a space-separated list of the extras in `pyproject.toml` (`documents`, `qdrant`, `pgvector`,
`neo4j`, `parquet`, ...). Build the image with the extras the model was trained with: a bundle whose encoder is
ONNX needs `documents`; a tabular project that reads Parquet needs `parquet`. The demo bundle and a documents
project with the hashing encoder run on the core image. `docker compose up` builds only when the image it names
(`RLCD_IMAGE`, default `open-rlcd:local`) does not exist yet, so after changing `RLCD_EXTRAS` run
`docker compose build` again (or `up -d --build`), or the old image keeps serving. An ONNX model on a core-only
image answers `/health` and `/ready` as usual and fails every decision that runs a trained head with 503
`encoder_unavailable`.

Compose builds from the checkout it runs in and tags the result with whatever `RLCD_IMAGE` names. Keep the
compose directory on the release that `RLCD_IMAGE` names: after an upgrade or an image rollback (see
[Upgrades and rollback](#upgrades-and-rollback)) the two can differ, and then `docker compose build` or
`up --build` puts this checkout's code under the other release's tag. While they differ, change the extras with
`docker build -t <that tag> --build-arg RLCD_EXTRAS=... .` in that release's checkout instead.

### Try it (the demo model)

```bash
export RLCD_API_KEY=$(openssl rand -hex 32) RLCD_ADMIN_KEY=$(openssl rand -hex 32)
docker compose up -d --build

# trust Caddy's internal CA for this test
docker compose exec -T proxy cat /data/caddy/pki/authorities/local/root.crt > rlcd-root.crt
curl --cacert rlcd-root.crt https://localhost/health
curl --cacert rlcd-root.crt https://localhost/v1/systemone \
  -H "Authorization: Bearer $RLCD_API_KEY" -H 'content-type: application/json' -d '{
  "state": {"query": "Who leads the team that owns the K-200 scanner?"},
  "questions": {"kind": {"type": "choice", "head": "response_type"}}}'
```

With no project mounted and an empty home, the service imports the bundled demo model (synthetic data only) and
serves it, exactly as `uv run rlcd serve` does after a clone. That is a smoke test, not a deployment.

### Run your project

Write the settings into a `.env` file next to `compose.yaml` (git ignores it). Run compose from that directory:
it then reads the file for every command, so every `docker compose` command on this page, and every one you
type later, includes the project overlay:

```bash
cat > .env <<'EOF'
COMPOSE_FILE=compose.yaml:deploy/compose.project.yaml
# the directory that holds rlcd.yaml
RLCD_PROJECT_DIR=/srv/acme
RLCD_DOMAIN=rlcd.example.com
RLCD_TLS=you@example.com
# the extras the model needs (see Build); leave the line out for the core image
RLCD_EXTRAS=documents
EOF
docker compose build
docker compose up -d
```

Keep the two keys in your secret store and export them in the shell that runs compose. If they do go into
`.env`, keep that file out of version control and readable only by operators.

**Every compose command needs the overlay.** `deploy/compose.project.yaml` is the only place that sets
`RLCD_PROJECT` and mounts the project. A container that compose creates or recreates without it (`up` after a key
rotation, an upgrade or an image rollback, or a one-off `run`) starts with no project. With a built-in pack,
nothing says so. It serves the model in the home on the code defaults: rollout `full` (whatever rlcd.yaml says,
`shadow` included), redaction of secrets only, and none of rlcd.yaml's routes, batteries or regression cases.
`/health` and `/ready` stay green. On a home with no model yet, it imports the synthetic demo model and makes it
`rlcd-latest`, with one line in the container log. `COMPOSE_FILE` in `.env` prevents that. Without it, pass
`-f compose.yaml -f deploy/compose.project.yaml` to every `docker compose` command, including the ones on this
page.

`docker compose up` builds only when the image does not exist yet. `build` comes first here because "Try it" may
have left a core-only `open-rlcd:local`.

A fresh home with a project has no model: `/health` reports `"model": "rlcd-0.1.0"` (the untrained id) and
`"ready": false`, `GET /ready` answers 503, `rlcd serve` says so on stderr when it starts, and every decision
abstains until you [import one](#moving-a-model-from-a-laptop-to-the-server).

In the container:

* `RLCD_HOME` wins over `home:` in rlcd.yaml, and the image passes `--host 0.0.0.0 --port 8095`, which win over
  `serve.host` / `serve.port`. Everything else in rlcd.yaml applies.
* rlcd.yaml is read once at start; there is no hot reload. After changing it (rollout mode, routes, batteries,
  redaction): `docker compose restart rlcd`, then check `rollout` in `/health`. `policy` and a head's `stakes`
  need a retrain, not a restart: a version serves the policy it was verified with. The table "When does a change
  take effect?" in [configuration.md](configuration.md#when-does-a-change-take-effect) lists every section.
* The project mount is read-only. One thing writes to the project directory: feedback tagged `regression` and
  `rlcd case add` append to `regressions.jsonl` there. On a read-only mount the feedback is still stored and the
  response says `regression_case: {stored: false}`. To keep those cases on the server, set
  `regressions: /var/lib/rlcd/regressions.jsonl` in rlcd.yaml.
* The entrypoint refuses to start when `RLCD_PROJECT` is set but holds no rlcd.yaml (exit 64), or when the home
  is not writable by uid 10001 (exit 73). `rlcd` itself refuses the first case too (`serve` and every other command that loads a project, exit 1), so
  a mistyped mount never serves the demo model, with or without the entrypoint.
* `rlcd serve` on a host that is not loopback (the image's `--host 0.0.0.0`) refuses to start without
  `RLCD_API_KEY` (exit 2), also when only `RLCD_ADMIN_KEY` is set. `--insecure-no-auth` overrides it for a
  network you trust; `compose.yaml` requires both keys anyway.

Any other `rlcd` command runs in the same image, against the same home and project (with `COMPOSE_FILE` in
`.env`, as above):

```bash
docker compose exec rlcd rlcd check                 # inside the running container
docker compose run --rm rlcd receipts verify        # a one-off container (when the service is stopped)
```

The CLI records `$USER` as the author of a promote, rollback, confirmation or kill-switch change, and the
container has none: pass `--by NAME` (`promote`, `rollback`, `automation`, `receipts confirm`) and put the
change ticket in `--reason`. Without `--by` and without `$USER`, promote, rollback and confirmations record
`unknown`.

## Environment variables

Read by the service (every `os.environ` lookup in `open_rlcd/`):

| variable | effect | default |
|---|---|---|
| `RLCD_HOME` | the state directory; overrides `home:` in rlcd.yaml | image: `/var/lib/rlcd` |
| `RLCD_PROJECT` | a project directory or an rlcd.yaml path; a path without an rlcd.yaml is an error | unset: the nearest rlcd.yaml above the working directory, else no project |
| `RLCD_API_KEY` | when set, every route except `/health` and `/ready` needs it | unset: **the API is open**, and `rlcd serve` refuses a non-loopback host (exit 2) unless `--insecure-no-auth` is passed |
| `RLCD_ADMIN_KEY` | when set, the three operator routes need this key instead; it protects nothing else | unset: the API key (or nobody) guards them |
| `RLCD_AUTOMATION` | `off` (also `0`, `false`, `disabled`, `no`) forces the kill switch off; the API cannot switch it back on | unset |
| `RLCD_ALLOW_REMOTE` | `1` lets connectors and LLM judges talk to non-loopback hosts (same as `security.allow_remote: true`) | unset: loopback only |
| `RLCD_MAX_BODY_BYTES` | largest request body; larger is a 413 | 16 MiB (`deploy/Caddyfile` has the same limit, `max_size 16MiB`; raise both together) |
| `RLCD_PACK` | the pack when rlcd.yaml names none | `documents` |
| `RLCD_DEBUG` | `1`: the CLI prints a traceback instead of one line | unset |
| `HF_HUB_OFFLINE` | `1`: the ONNX encoder is never downloaded; a missing file is an error | unset |

Not needed on a server: `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` (the labeling judges, used on the training
machine) and `RLCD_URL` (the Python client's default base URL). The image also sets
`HF_HOME=/var/lib/rlcd/cache/huggingface` so the Hugging Face client keeps its own files inside the volume.

Ports: `8095` inside the container (fixed by the image's command and its `HEALTHCHECK`); the proxy publishes
`443` and `80` (`RLCD_HTTPS_PORT`, `RLCD_HTTP_PORT` to change the host side).

## HTTPS and keys

The service speaks plain HTTP and has no TLS options: terminate TLS in front of it and do not publish 8095.
`compose.yaml` does this with Caddy; `RLCD_TLS` picks the certificate:

| `RLCD_TLS` | certificate | use when |
|---|---|---|
| `internal` (default) | Caddy's own CA | trying it out, or a private name; clients must trust `/data/caddy/pki/authorities/local/root.crt` |
| an e-mail address | public, by ACME | `RLCD_DOMAIN` resolves to this host and 80/443 are reachable from the internet |
| `/certs/fullchain.pem /certs/privkey.pem` | your own | a corporate CA; mount the files into the `proxy` service |

Keys are compared in constant time and sent as `Authorization: Bearer <key>` (or `X-API-Key`).

| caller presents | `/health` | every other `GET`, and the decision routes (`systemone`, `rank`, `decide`, `evaluate`, `feedback`, `outcome`, `receipts/{id}/check`) | `POST /v1/train`, `POST /v1/automation`, `POST /v1/receipts/{id}/confirm` |
|---|---|---|---|
| nothing | yes | 401 | 401 |
| `RLCD_API_KEY` | yes | yes | 401 |
| `RLCD_ADMIN_KEY` | yes | yes | yes |

That table is with both keys set, which is what `compose.yaml` requires. The other combinations:

* only `RLCD_API_KEY`: whoever holds it can also retrain, switch automation back on and confirm its own decisions;
* only `RLCD_ADMIN_KEY`: the three operator routes are protected and **everything else is open**;
* neither: everything is open.

In the last two cases `rlcd serve` refuses to start on a host that is not loopback (exit 2, one line that says
which routes would be open). `--insecure-no-auth` starts it anyway, with a warning; on loopback it starts as
before.

What the keys do not do: there is one API key for all callers (no per-caller keys, scopes or rate limits), and
`/health` and `/ready` are always open. `/health` shows the model version, the head names, the rollout mode and the kill-switch state;
with a key set it hides the home path. Keep it off the public internet if that is too much.

**Rotation.** Keys are read from the process environment, so a new key takes effect when the container is
recreated. With the keys exported from a secret store:

```bash
export RLCD_API_KEY=$(openssl rand -hex 32)      # update your secret store first
docker compose up -d rlcd                        # recreates the container with the new value
```

With the keys in `.env`, change the `RLCD_API_KEY=` line there instead, then run `docker compose up -d rlcd`.
An `export` overrides `.env` only in the shell that ran it: the next `docker compose up` from another shell or
session (an upgrade, an image rollback, a restore) reads the old key from `.env`, so the rotated-out key works
again and callers holding the new one get 401.

On a project deployment the recreated container gets the project only through `COMPOSE_FILE` in `.env` (see
[Run your project](#run-your-project)). Without it, the container comes back on the code defaults.

There is no overlap window: after the restart the old key is a 401, which clients must not retry. Roll the new
key out to callers and restart in the same change window. The admin key rotates the same way and affects only
operators. The kill switch, the model alias and all state live in the home and survive the restart.

## Storage

Everything the service persists is under `RLCD_HOME`. Nothing is written anywhere else (the compose file runs the
root filesystem read-only, with a tmpfs `/tmp`).

| path | what | written | class |
|---|---|---|---|
| `registry.json` | the versions, which one `rlcd-latest` points at, and the history of what made each version latest (train, import, promote, rollback, and a weight edit `rlcd patch` promoted), which `rlcd rollback` follows | on train, import, promote, rollback, patch (whole file, atomic replace) | **precious** |
| `models/<version>/` | one bundle per version: `model.json`, weights, calibrators, reports | once per version, never changed after | **precious** (re-importable only if you kept the exported bundle) |
| `decisions.jsonl` | the receipts: one hash-chained line per decision, with the logged state | appended per decision | **precious**, the audit trail; cannot be rebuilt |
| `feedback.jsonl` | labels posted to `/v1/feedback`; training rows for the next model | appended | **precious** |
| `outcomes.jsonl` | task outcomes posted to `/v1/outcome` | appended | **precious** for the metrics |
| `confirmations.jsonl` | who confirmed which confirm-tier decision | appended | **precious**, part of the audit trail |
| `bernoulli.json` | the live right / wrong counts per head that gate `act` | rewritten on each scored feedback | **precious**: holdout counts can be rebuilt from the bundle, online outcomes cannot |
| `bernoulli.archive.json` | the ledger tables of earlier versions, used by rollback | on train, promote, rollback | **precious** |
| `automation.json`, `automation.jsonl` | the kill-switch state and its history | on `POST /v1/automation`, `rlcd automation` | **precious** (a restore must not silently switch automation back on) |
| `labels/` | items and judge / human labels | by `rlcd label ...` | **precious** where labeling happens (usually the training machine, not the server) |
| `tabular/category_salt` | salt for hashed categorical columns (tabular pack) | once | keep with the home; each bundle also stores the salt it was trained with |
| `corpus/` | documents pack: chunks, vector index, entity graph, BM25. With a remote store in index mode (`pack_config.vector` / `graph`) the vectors or the graph live in that store instead; with `vector` in attach mode `corpus/` is not used | by `rlcd ingest` | rebuildable from the source documents (re-ingest); back up a remote store with that store's own tools |
| `encoders/` | the downloaded ONNX encoder (~135 MB) | first use | rebuildable (downloads again, unless `HF_HUB_OFFLINE=1`) |
| `cache/embeddings/*.sqlite` | embedding cache, one file per encoder hash | continuously | rebuildable; safe to delete while stopped |
| `cache/huggingface/` | the Hugging Face client's own files | first download | rebuildable |
| `reports/` | `rlcd report` output | on demand | rebuildable |
| `*.tmp` | temporary files of an atomic replace (`registry.json.<pid>.<thread>.tmp`, ...) and of an ingest (under `corpus/`, such as `meta.json.tmp`) | transient | ignore; a leftover can be deleted while the service is stopped |
| `*.lock` | inter-process lock files next to the files they guard (`registry.json.lock`, `bernoulli.json.lock`, ...) | on first use, never removed | keep: never delete one while any `rlcd` process runs; harmless in a backup |
| `decisions.jsonl.bak` | a copy of the log while `rlcd receipts prune` rewrites it | removed when the prune succeeds | one that is still there means a prune failed: keep it, and restore `decisions.jsonl` from it if `rlcd receipts verify` fails |

Not in the home: `rlcd.yaml`, `regressions.jsonl` and your source documents live in the project directory. Keep
that in version control.

**Growth.** The four `.jsonl` logs are append-only and never rotated by the service. `decisions.jsonl` grows
fastest because it holds request states (`serve.log_states`, on by default, secrets redacted). To bound it:

```bash
docker compose stop rlcd
docker compose run --rm rlcd receipts prune --before 2026-10-01   # drops old states, keeps digests: the chain still verifies
docker compose start rlcd
```

Pruning rewrites the file; run it with the service stopped. Feedback can no longer train from a pruned decision.
`/v1/metrics` and `/v1/review` read the logs from the start on every call, so they slow down as the logs grow.

**Receipts are tamper-evident, not tamper-proof.** Anyone who can write the home can rewrite the whole chain.
`rlcd receipts verify` prints the head digest of each chain (`head` for `decisions.jsonl`, `confirmations.head`
for `confirmations.jsonl`): store both somewhere the service cannot write (your log system, a ticket, an
object-lock bucket) on a schedule, and compare on restore.

### Back up

Stop the writer, copy the directory, start again. The copy is then consistent across the logs, the ledger and the
registry.

```bash
docker compose stop rlcd
docker compose run --rm -T rlcd receipts verify | tee receipts-head.json        # note both chains' "head" and "receipts"
docker run --rm -v rlcd_rlcd-home:/var/lib/rlcd:ro --entrypoint tar open-rlcd:local \
  -C /var/lib/rlcd --exclude ./cache -czf - . > rlcd-home-$(date +%F).tgz
docker compose start rlcd
```

Record the head after the stop. A decision logged between the two steps would be in the archive but not in the
recorded head, and the restore check below would then fail for a good backup. `-T` keeps the command's stderr
out of the JSON. (`rlcd_rlcd-home` is the compose volume. `open-rlcd:local` is the image `compose.yaml` builds;
any image of this project will do, since only its `tar` runs.) A filesystem or volume snapshot taken while
stopped is equally good. Add `--exclude ./encoders --exclude ./corpus` if you would rather re-download and
re-ingest.

A copy taken while the service runs is usable but not guaranteed consistent: a log may end in a half-written line
(readers skip it and `verify` lists it under `unreadable`), and the ledger may be one feedback ahead of or behind
the logs. Prefer the stopped copy; it takes seconds.

### Restore and verify

```bash
docker compose down                                   # volumes stay
docker run --rm -v rlcd_rlcd-home:/var/lib/rlcd --entrypoint sh open-rlcd:local \
  -c 'find /var/lib/rlcd -mindepth 1 -delete'         # empty the volume (skip for a new one)
docker run --rm -i -v rlcd_rlcd-home:/var/lib/rlcd --entrypoint tar open-rlcd:local \
  -C /var/lib/rlcd -xzf - < rlcd-home-2026-11-10.tgz

docker compose run --rm rlcd receipts verify          # exit 1 on a broken chain
docker compose run --rm rlcd check                    # the pre-launch checklist, exit 1 on a FAIL
docker compose run --rm rlcd automation status        # the kill switch is as you left it
docker compose up -d
curl --cacert rlcd-root.crt https://localhost/health  # "model" is the version you expect
```

(With a project, these commands rely on `COMPOSE_FILE` in `.env`, as everywhere on this page: see
[Run your project](#run-your-project).)

`receipts verify` must report `"ok": true`, and `"head"` and `"receipts"`, and the same two under
`"confirmations"`, must equal what you noted at backup time. `ok` alone does not prove completeness: a log cut
off at the end still verifies, which is why the head digests are recorded outside the home. For a bind-mounted home instead of a named volume, `chown -R 10001:10001`
the directory on the host.

## Concurrency

**Run exactly one `rlcd serve` process per `RLCD_HOME`.** That means: no `uvicorn --workers N`, no second
replica mounted on the same volume, no shared (RWX / NFS) home between hosts, and no rolling update that runs old
and new side by side. In Compose, do not `--scale rlcd`; in an orchestrator, one replica with a stop-then-start
(recreate) update strategy.

Inside that one process, requests run concurrently on a thread pool and every write is serialised by a lock, so
one process is safe under load. Every file a second process may write (the server next to an `rlcd` admin
command) is also guarded by an inter-process file lock (`<file>.lock` next to it, `flock`: local POSIX filesystems
only). What still lives in one process's memory is why a second *server* on the same home is not supported:

| state | how it is written | two processes on one home |
|---|---|---|
| `decisions.jsonl` (receipt chain) | append under an exclusive file lock; the chain tip is re-read when another writer was there | safe on a local POSIX filesystem: still one chain |
| `feedback.jsonl`, `outcomes.jsonl`, `confirmations.jsonl` | one appended line under a file lock | safe: no row is torn or lost. A decision id missing from a process's in-memory index is looked up in the shared `decisions.jsonl`, so another process's decisions are found (each such miss reads the whole log) |
| `bernoulli.json` (the ledger that gates `act`) | read, change, atomic replace, all under a file lock | safe: no update is lost |
| `registry.json` (versions, the `rlcd-latest` alias) | read, change, atomic replace, all under a file lock | safe: version ids are never claimed twice. A long `import` holds the lock, so a train finishing meanwhile waits for it |
| the alias and the kill switch, read side | re-read from disk on every request | fine: every process sees a promote, rollback or `automation off` at once |
| training jobs | in memory; "one at a time" is enforced per process | a second process starts a second train; `GET /v1/train/{id}` only works on the process that started it |
| embedding cache (`cache/embeddings/*.sqlite`) | one SQLite connection per process (WAL mode) | writers take turns through SQLite's own lock, and a writer waits up to 5 s before its request fails: fine for an admin command next to the server, not designed for several busy writers |

The `rlcd` CLI is a second process too. It is meant to be used against a live home for the operator commands,
and the service picks the result up without a restart:

| command while the service runs | safe? |
|---|---|
| `receipts verify`, `receipts show`, `metrics`, `review`, `automation status`, `calibration`, and `check` without regression cases | yes (read-only) |
| `replay`, `regress`, and `check` when the project has regression cases | yes, but not read-only: they encode states through the embedding cache, so they write `cache/embeddings/*.sqlite` (see the table above) and need a writable home |
| `automation off` / `on` | yes (one atomic file) |
| `promote`, `rollback`, `import` | yes: they rewrite `registry.json` and `bernoulli.json` under the same file locks the server takes, so a feedback call landing in the same instant is still counted |
| `receipts prune`, `train` | stop the service first (prune rewrites the log; a CLI train and an API train are not mutually excluded) |

### Scaling

* **Up first.** Heads take well under a millisecond; the text encoder dominates. One process on a few cores
  goes a long way. Measure with `rlcd bench` in the container before adding anything.
* **Out means more homes, not more workers.** Run N independent instances, each with its own volume, and import
  the same bundle into each. Each instance then has its own receipt chain, feedback log, ledger, version ids and
  kill switch. Consequences you have to route around:
  * `/v1/feedback`, `/v1/outcome`, `/v1/receipts/{id}/...` must reach the instance that made the decision (pin a
    caller or a tenant to an instance; do not round-robin);
  * the kill switch is per home: set `RLCD_AUTOMATION=off` on every instance, or call each one;
  * the Bernoulli evidence and the metrics are per instance; feedback for retraining has to be collected from
    each home.
* Callers that only use `/v1/systemone` and `/v1/rank` with `serve.receipts: decisions` (the default) write no
  receipts, so for them the instances are interchangeable.

There is no shared-state backend (database, object store) to point several replicas at.

## Moving a model from a laptop to the server

A trained model travels as a bundle directory: JSON, safetensors and LightGBM text, no pickles, no training rows.

On the training machine:

```bash
rlcd check                       # the checklist, against the version you are about to ship
rlcd export ./acme-bundle        # the current rlcd-latest (or --model VERSION); the target must be empty or new
```

Copy `acme-bundle/` to the server by whatever you trust (it is model weights learned from your data: treat it
like the data). On the server, from the directory with `compose.yaml` and its `.env` (the overlay comes from
`COMPOSE_FILE` there, see [Run your project](#run-your-project)):

```bash
# 1. register it without serving it
docker compose run --rm -v "$PWD/acme-bundle":/bundle:ro rlcd import /bundle --keep-latest
#    -> imported /bundle as rlcd-0.1.4

# 2. look at it
docker compose exec rlcd rlcd check --model rlcd-0.1.4
docker compose exec rlcd rlcd regress --model rlcd-0.1.4      # if the project has regression cases
docker compose exec rlcd rlcd replay --candidate rlcd-0.1.4   # logged decisions under the old and the new version

# 3. serve it
docker compose exec rlcd rlcd promote rlcd-0.1.4 --reason "CHG-1234"

# 4. if it misbehaves
docker compose exec rlcd rlcd rollback --reason "INC-42"      # back to the version that served before; or --to VERSION
```

How these really behave:

* **Import validates and renumbers.** Every file is checked against the sha256 in `model.json`, LightGBM members
  are parsed in a throwaway subprocess, and the bundle is copied into `models/` under the **next version id of
  this home**. The id on the server is therefore not the id on the laptop; `registry.json` records
  `imported_from`. Without `--keep-latest`, import also points `rlcd-latest` at the new version immediately.
* **Promote and rollback only move the alias.** No retrain, no restart: the running service reads the alias on
  every request. Each move is recorded in `registry.json` `history` with the reason. Callers that pin
  `"model": "rlcd-0.1.3"` are unaffected; receipts issued under the old alias target become stale
  (`/v1/receipts/{id}/check` says so).
* **The Bernoulli ledger follows the version.** Promoting a version restarts each head's table from that
  version's own holdout (or from its archived table if it served before), so evidence earned by one model is
  never credited to another.
* **`rlcd rollback` without `--to`** goes to the version that was latest before the current one and skips
  versions that were themselves rolled back from. With nothing earlier it exits 1.
* **The first import into an empty home** with `--keep-latest` leaves the service untrained until you promote.

What a bundle does not carry, for the documents pack:

| needed at serve time | how it gets to the server |
|---|---|
| the encoder (ONNX) | downloaded from the Hugging Face Hub into `encoders/` on first use, and its sha256 must match the bundle's. No egress: copy `encoders/` from the training home and set `HF_HUB_OFFLINE=1` |
| the corpus, if callers rely on open-rlcd's own retrieval | `rlcd ingest` on the server against the mounted documents, or copy `corpus/` from the training home. Callers that send their own `candidates` need none. A copied `corpus/` names the remote stores that hold its index (Qdrant, pgvector, Neo4j in index mode): the server needs the same `pack_config.vector` / `graph`, network access to them and their credentials in its environment. A store in attach mode needs only the last two |
| feedback posted in production | it is in the server's `feedback.jsonl`. Training reads the home it runs against: train where the feedback is, or copy the file to the training home |

`POST /v1/train` exists and works on a server, but it trains from the labels and feedback in the server's home
and commits the result as `rlcd-latest` at once. Most teams keep training on the laptop and leave that route to
the admin key.

## Upgrades and rollback

**A new image on the same home:**

```bash
# 1. back up the home (above)
# 2. try the new image against a copy, not the live home. NEW_IMAGE is the new release built under a tag of its
#    own, with the same RLCD_EXTRAS, in a checkout of that release:
#      docker build -t open-rlcd:2026-12 --build-arg RLCD_EXTRAS=documents .
#    A tag of its own keeps the old image for the rollback below.
docker volume create rlcd-upgrade-test
docker run --rm -i -v rlcd-upgrade-test:/var/lib/rlcd --entrypoint tar NEW_IMAGE -C /var/lib/rlcd -xzf - < rlcd-home-2026-11-10.tgz
docker run --rm -v rlcd-upgrade-test:/var/lib/rlcd -v /srv/acme:/project:ro -e RLCD_PROJECT=/project NEW_IMAGE check
docker run --rm -v rlcd-upgrade-test:/var/lib/rlcd -v /srv/acme:/project:ro -e RLCD_PROJECT=/project NEW_IMAGE regress
docker run --rm -v rlcd-upgrade-test:/var/lib/rlcd -v /srv/acme:/project:ro -e RLCD_PROJECT=/project NEW_IMAGE \
  ask --query "a question your users ask"
docker volume rm rlcd-upgrade-test
# 3. switch: set RLCD_IMAGE=NEW_IMAGE in .env (edit the line if there is one already), then
docker compose up -d rlcd          # compose stops the old container before starting the new one
```

`RLCD_IMAGE` goes into `.env`, not in front of one command. Every later compose command has to use the same
image: a key rotation, a `run --rm` for verify, prune or a restore. A command without it falls back to
`open-rlcd:local`. `up` then recreates the service on the old image, or builds one from whatever the checkout
holds, and a one-off `run` executes old code against a home the new code wrote.

Switch the compose directory to the new release's checkout in step 3 as well. Compose builds from that checkout
into the tag `RLCD_IMAGE` names, so while the two differ a `docker compose build` or `up --build` (after an
extras change, say) puts the old code under the new release's tag, and `up` serves it against a home the new
code wrote.

What can stop an old model from answering under new code:

| check | where it is recorded | on mismatch |
|---|---|---|
| bundle format | `model.json` `format` (`open-rlcd-bundle/1`) | the bundle is refused at load |
| `feature_version` | `model.json`, e.g. `documents-v2:hashing-256-s0:0b1d19ee81e5` = the pack's feature schema, the encoder id, the encoder file's hash | every request to that head is refused until a model trained on the new features is imported. In the documents pack, a release that changes the number of features in a view is refused first, with a 422 `feature_schema_mismatch`. A change to the schema label or the encoder id alone gives a 422 `feature_version_mismatch`. A release that changes a pack's features changes the schema part. `/ready` stays 200 either way, which is why step 2 runs against a copy |
| encoder hash | `model.json` `encoder.sha256` | the encoder refuses to load: same model file, or retrain |
| extras | the image's `RLCD_EXTRAS` | an ONNX bundle on a core-only image fails when the encoder is first needed |

There is no migration step and no schema version on the home: a release either reads the files as they are or
the checks above say why not. That is why step 2 runs against a copy.

**Rolling back the image:** set `RLCD_IMAGE` in `.env` back to the previous image (remove the line to return to
`open-rlcd:local`), switch the compose directory back to that release's checkout, then
`docker compose up -d rlcd`. Without the checkout switch, a later `docker compose build` or `up --build` rebuilds
the new release's code under the previous image's tag and silently undoes the rollback. If the new image trained
or imported models, or you are unsure the old code reads what the new one wrote, restore the backup from step 1
instead.

**Rolling back the model** is independent of the image: `rlcd rollback` (above).

## Monitoring

| endpoint | key | use |
|---|---|---|
| `GET /health` | none | liveness: 200 whenever the process is up. The image's `HEALTHCHECK` and the proxy's upstream check call it |
| `GET /ready` | none | readiness: 200 when a trained model can decide, 503 (`kind: not_ready`, with `reasons`) while the service is untrained, its pack cannot load or its policy cannot be served. It builds no encoder and opens no corpus, so it still answers 200 when a missing extra or encoder files missing under `HF_HUB_OFFLINE=1` fail every decision that runs a trained head with 503 `encoder_unavailable`, and when a broken corpus fails retrieval (a `/v1/decide` sent without `candidates`: see the smoke decisions below) |
| `GET /v1/check` | API | the pre-launch checklist as JSON: `ok`, and per item pass / warn / fail |
| `GET /v1/metrics?since=2026-11-01` | API | action mix, false automation, online reliability against the holdout, review load, latency. JSON, not Prometheus format |
| `GET /v1/calibration?head=answerable` | API | the holdout numbers a head was accepted on |
| `GET /v1/automation` | API | the kill switch, the rollout mode, the switch history |
| `GET /v1/review` | API | decisions waiting for a person |

`/health` answers `"status": "ok"` whenever the process is up: it says "restart me or not", nothing about the
model. Use `/ready` (or the `ready` field of `/health`) for "send traffic or not": a Kubernetes `readinessProbe`,
a load balancer's health check, a deploy script that waits for the import. Keep the liveness probe on `/health`:
an untrained service is alive, and restarting it does not train it. Neither endpoint touches the encoder or the
corpus, so also send smoke decisions:

* `POST /v1/systemone` with a state you know, expecting 200. It needs the encoder, never the corpus. It writes
  no receipt unless the request sets `"receipt": true` or `serve.receipts` is `all`, so it can also run as a
  periodic probe.
* If your callers rely on open-rlcd's retrieval (a `/v1/decide` without `candidates`), `POST /v1/decide` with a
  known query and no candidates, expecting 200 and `"candidates_from": "corpus"` with no warning about
  retrieval. The corpus is opened only for such a decide: an unreadable corpus, or a vector / graph store that
  cannot be reached, fails it with 503 `connector_unavailable`, and a corpus it cannot use (built with another
  encoder, an unknown format, a store configuration the store refuses, such as a host that is not allowed)
  answers 200 with `"candidates_from": "none"` and the warning that retrieval is unavailable.

  A decide is a real decision. It writes a receipt and counts in `/v1/metrics`. Unless its action is `act` (and
  none is while the kill switch is off, or while `rollout.mode` is `shadow` or `assist` and no `rollout.heads`
  entry sets a head to `full` or `limited`; `/health` shows only `rollout.mode`), it also waits in
  `GET /v1/review` and in `review_load.pending` until a `POST /v1/feedback` names it; a confirmation does not
  take it out. That feedback is a training row for the next model and updates the head's Bernoulli table, so
  post it only with a label you know is right. Neither endpoint filters on tags. So run this probe after each
  deploy, upgrade, import and ingest rather than as a frequent poll, and tag it (`"tags": ["smoke"]`): the tag is
  kept in the receipt's `audit` and shown on its review item.

| alert when | where | why |
|---|---|---|
| `/health` fails or the container is `unhealthy` | probe | the process is down |
| `/ready` answers 503 (`/health` `ready` is `false`) | probe | no trained model can decide: `reasons` says why (untrained, the pack cannot load, an unusable stored policy) |
| `/health` `model` equals `untrained_id` (`rlcd-0.1.0`) | `/health` | no model is loaded: every decision abstains (an empty or wrong volume) |
| `/health` `heads` has an `error` key | `/health` | the pack itself cannot load (an unknown pack, or a custom pack whose module fails to import); `/ready` is 503 too. A missing extra, missing encoder files or an unreadable corpus do not show here: see the 5xx row |
| `/health` `model` changes when nobody promoted | `/health` | someone trained, imported or rolled back |
| `/health` `automation` is `false` | `/health` | the kill switch is off: informational, but someone should own it |
| `drift_suspected` is non-empty | `/v1/metrics` | online act-tier accuracy is below what the holdout promised: retrain or roll back |
| `review_load.review_over_capacity` is true | `/v1/metrics` | more confirm decisions per day than `serve.review_capacity_per_day` |
| the automatic share (`actions.automatic`) drops or the abstain share jumps | `/v1/metrics` | inputs changed, or a cap was applied |
| `rlcd receipts verify` exits 1, or its `receipts` or `confirmations.receipts` count is lower than the last recorded one | scheduled job | the audit trail (decisions or confirmations) was edited or truncated. Compare counts, not the head digests, which every new decision or confirmation changes |
| `ok` is false | `/v1/check` | an item fails: the receipt chain does not verify (`receipts.chain`), the bundle does not validate (`model.bundle`), a regression case fails (`regressions.cases`), rlcd.yaml has an error (`config.valid`), ... |
| `counts.warn` rises, or `secrets.redact`, `rollback.ready` or `automation.switch` turns to `warn` | `/v1/check` | warnings never make `ok` false: redaction off, no rollback target and the kill switch off are warnings |
| rate of 5xx; rate of 401 | proxy log | failures, including the per-request 503s that `/health` and `/ready` cannot see (`encoder_unavailable`, `connector_unavailable`: the kind is only in the response body, and the documents pack's own 503s are not logged); a caller with an old key |
| free space on the home volume | host | the logs are never rotated |

`/v1/metrics` reads the full logs on each call: poll it every few minutes with `since`, not every few seconds.
Outcome-based numbers (false automation, drift) only exist if callers post `/v1/feedback` and `/v1/outcome`.

**Logs.** The service writes to stdout / stderr: uvicorn's access line per request and the `open_rlcd` logger for
errors. A 500, and a 503 from a configured vector or graph store, carry only the exception's type in the body;
the details are in the container log (`docker compose logs -f rlcd`). The documents pack's own 503s differ:
`encoder_unavailable` and a local-corpus `connector_unavailable` carry the error message in the body (the encoder
one can include the absolute path of `encoders/` in the home), and nothing is logged for them. Their reason is
in the response the caller received. Verbosity: override the command with
`["serve", "--host", "0.0.0.0", "--port", "8095", "--log-level", "warning"]`. Request bodies are
not logged here; they are in `decisions.jsonl` (redacted) when `serve.log_states` is on. The proxy logs JSON
access lines with status codes.

**The kill switch.** One call caps every decision at `confirm`; decisions are still made, logged and returned.

```bash
curl https://rlcd.example.com/v1/automation -H "Authorization: Bearer $RLCD_ADMIN_KEY" \
  -H 'content-type: application/json' -d '{"enabled": false, "reason": "INC-42", "by": "oncall"}'
# or, on the host
docker compose exec rlcd rlcd automation off --reason "INC-42"
```

It is a file in the home: it takes effect on the next request and survives restarts and upgrades. To make sure
nobody switches it back on through the API, put `RLCD_AUTOMATION=off` into `.env` and recreate the container
with `docker compose up -d rlcd` (`docker compose restart` keeps the old environment). `POST /v1/automation
{"enabled": true}` is then a 409 until the variable is removed the same way. Callers that hold an older
`act` receipt find out through `/v1/receipts/{id}/check`, which reports it stale.

## Hardening checklist

- [ ] Both `RLCD_API_KEY` and `RLCD_ADMIN_KEY` are set, long and random, different from each other, and come from
      a secret store (not from the image, rlcd.yaml or a committed `.env`).
- [ ] Decision callers hold only the API key. The admin key is with operators and the reviewers' tooling.
- [ ] Port 8095 is not published; only the proxy reaches it. TLS terminates at the proxy with a certificate
      clients verify.
- [ ] The container runs as uid 10001 with a read-only root filesystem, `cap_drop: [ALL]` and
      `no-new-privileges` (as in `compose.yaml`). The project mount is read-only.
- [ ] Exactly one service container uses the home volume; the update strategy is stop-then-start.
- [ ] On a project deployment, `.env` sets `COMPOSE_FILE` with the overlay (and `RLCD_IMAGE` after an upgrade),
      so no compose command recreates the service without its project.
- [ ] The home is on encrypted storage: `decisions.jsonl` and `feedback.jsonl` hold request contents. Redaction
      (`serve.redact`) is pattern-based; set `[secrets, pii]` if states can hold e-mail addresses or card
      numbers, or `serve.log_states: false` to keep states out of the receipts entirely.
- [ ] Backups run on a schedule, a restore has been rehearsed, and both receipt head digests (decisions and
      confirmations) are recorded outside the home.
- [ ] A retention period is decided and `rlcd receipts prune` is scheduled for it.
- [ ] `RLCD_ALLOW_REMOTE` is unset unless a connector needs a remote host, and then egress is limited to that
      host. With `HF_HUB_OFFLINE=1` and a pre-seeded `encoders/`, the service needs no egress at all.
- [ ] `rlcd check` passes on the server, the rollout mode is the one you intend (`shadow` first), and
      `rlcd rollback` has a target.
- [ ] The alerts above exist, and someone is named for the kill switch.
- [ ] Request size and rate limits are set at the proxy for your traffic; the service has no rate limiting.

## What this guide does not give you

* **No high availability.** One process per home is a single point of failure by design. Recovery is restart, or
  restore. There is no replication, leader election or shared-state backend.
* **No Kubernetes manifests, Helm chart or Terraform.** The image is orchestrator-neutral; if you write a
  manifest, it is one replica, a `Recreate` strategy, a ReadWriteOnce volume at `/var/lib/rlcd`, the project as a
  read-only ConfigMap or volume, and `/health` as the probe.
* **No zero-downtime key rotation, per-caller keys, scopes, rate limits or SSO.** Two shared bearer keys. Put an
  API gateway in front if you need more.
* **No Prometheus endpoint, tracing or structured application logs.** `/v1/metrics` is a JSON report you poll.
* **No log rotation or archival** for the home's `.jsonl` files, and no automatic backup.
* **No published image.** You build it from the repository; the build was verified on linux/arm64 and is built
  on linux/amd64 in CI.
* **No home migrations.** Upgrades are verified by trying the new image on a copy.
* **No tamper-proof audit trail** without your help: anchor both receipt head digests (decisions and
  confirmations) outside the home.
