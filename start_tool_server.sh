# 运行server.py前需要运行以下指令，否则mini_webarena包无法正常被引用
export PYTHONPATH=$PYTHONPATH:/DATA/disk0/yjb/yutao/lzt/BrowserAgent_v2
# 自动获取当前目录并添加到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/verl-tool

# 开两个进程运行以下指令开启server与测试text_browser
unset http_proxy https_proxy all_proxy

python -m verl_tool.servers.serve \
	--tool_type text_browser \
	--url=http://localhost:5000/get_observation \
	--uvi_workers 1 \
	--router_workers 1 \
	--workers_per_tool 120