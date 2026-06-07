# 运行server.py前需要运行以下指令，否则mini_webarena包无法正常被引用
export PYTHONPATH=$PYTHONPATH:/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2
# 自动获取当前目录并添加到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/verl-tool

# 开两个进程运行以下指令开启server与测试text_browser
unset http_proxy https_proxy all_proxy

# Keep the text-browser smoke-test/eval server lightweight. Without these
# limits, serve.py's defaults expand to 4 backend workers and 4096 threads.
export TEXT_BROWSER_RAY_NUM_CPUS=${TEXT_BROWSER_RAY_NUM_CPUS:-4}
export TEXT_BROWSER_ACTION_TIMEOUT_SEC=${TEXT_BROWSER_ACTION_TIMEOUT_SEC:-120}
export TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC=${TEXT_BROWSER_ENV_RPC_TIMEOUT_SEC:-110}
export VT_HEALTH_CHECK_TIMEOUT=${VT_HEALTH_CHECK_TIMEOUT:-180}

python -m verl_tool.servers.serve \
	--tool_type text_browser \
	--url=http://localhost:5000/get_observation \
	--uvi_workers 1 \
	--router_workers 1 \
	--workers_per_tool 16 \
	--max_concurrent_requests 16 \
	--thread_pool_size 64 \
	--request_timeout 120
