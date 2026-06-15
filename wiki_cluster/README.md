# Native Wiki Cluster

This directory can run the local Wikipedia ZIM service without Docker. It starts
multiple `kiwix-serve` processes over four read-only ZIM copies and exposes a
single load-balanced entry point on port `22015`. By default, each ZIM copy runs
two backend processes.

## Start

```bash
cd /data/yutao/lzt/BrowserAgent_v2/wiki_cluster
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
ZIM_COPIES=4
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
