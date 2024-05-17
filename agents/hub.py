from langchain import hub

# 初始化 Hub 实例
hub_instance = hub()

# 列出可用的模型、代理和工具
models = hub_instance.list_models()
agents = hub_instance.list_agents()
tools = hub_instance.list_tools()

print("可用的模型:", models)
print("可用的代理:", agents)
print("可用的工具:", tools)
react_prompt = hub.pull("hwchase17/react")
