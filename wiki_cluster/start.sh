#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "[1/4] 停掉可能占用 22015 的旧容器..."
docker rm -f wikipedia-4 wikipedia-3 wikipedia-2 wikipedia-1 wiki-lb 2>/dev/null || true

echo "[2/4] 启动 compose 集群..."
docker-compose up -d

echo "[3/4] 等待服务启动..."
sleep 5

echo "[4/4] 当前容器状态："
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "wiki-lb|wikipedia-"

echo
echo "测试入口 URL："
echo "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
