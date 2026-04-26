#!/usr/bin/env bash

echo "===== docker ps ====="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "wiki-lb|wikipedia-"

echo
echo "===== backend health ====="
for port in 22115 22116 22117 22118; do
    echo -n "checking backend $port ... "
    curl -I -s "http://localhost:${port}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing/" | head -n 1
done

echo
echo "===== load balancer health ====="
curl -I -s "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing/" | head -n 1

echo
echo "===== recent logs ====="
docker logs --tail=20 wiki-lb 2>/dev/null || true
for name in wikipedia-1 wikipedia-2 wikipedia-3 wikipedia-4; do
    echo
    echo "--- $name ---"
    docker logs --tail=5 "$name" 2>/dev/null || true
done
