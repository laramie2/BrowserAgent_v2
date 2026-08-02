# Native Wiki Cluster

This directory can run the local Wikipedia ZIM service without Docker. It starts
multiple `kiwix-serve` processes over one or more read-only ZIM paths and
exposes a single load-balanced entry point on port `22015`. By default, each
ZIM path runs two backend processes.

## Prepare Wiki Assets

From the project root, the training preparation CLI verifies and extracts the
bundled Kiwix 3.3.0 archive, downloads the Wiki ZIM parts, and assembles the ZIM
locally:

```bash
python3 scripts/prepare_training.py prepare
```

The default layout stores one physical ZIM and `./wiki_cluster/start.sh` uses
one ZIM path. To expose four compatible paths without duplicating the ZIM
bytes:

```bash
python3 scripts/prepare_training.py prepare --wiki-copies 4
ZIM_COPIES=4 ./wiki_cluster/start.sh
```

## Start

```bash
cd /path/to/BrowserAgent_v2/wiki_cluster
./start.sh
```

Default entry URL:

```text
http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
```

If `tools/kiwix-tools_linux-x86_64-3.3.0/kiwix-serve` exists, `start.sh`
uses it automatically. This matches the old Docker image
`ghcr.io/kiwix/kiwix-serve:3.3.0`. If `kiwix-serve` is elsewhere, point to it
explicitly:

```bash
KIWIX_SERVE_BIN=/path/to/kiwix-serve ./start.sh
```

## Useful Environment Variables

```bash
ZIM_ROOT=/path/to/webarena_zim
ZIM_NAME=wikipedia_en_all_maxi_2022-05.zim
ZIM_COPIES=1
WORKERS_PER_ZIM=2
ZIM_PATHS=/path/to/copy1.zim,/path/to/copy2.zim
PORT_START=22115
LB_PORT=22015
LB_HOST=0.0.0.0
STATE_DIR=/tmp/wiki_cluster_run
WATCHDOG_INTERVAL=15
WATCHDOG_HEALTH_TIMEOUT=12
WATCHDOG_MAX_FAILURES=3
```

## Check and Stop

```bash
./check.sh
./stop.sh
```

Logs and pid files are written under `wiki_cluster/run/`.

`start.sh` also launches `wiki_watchdog.sh`, which restarts an individual
`kiwix-serve` backend when its pid exits or its health check fails repeatedly.

`docker-compose.yml` is kept as the old Docker-based launcher for machines that
can run Docker directly.
