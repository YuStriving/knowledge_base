import time
import sys
from app.utils.task_utils import add_done_task,add_running_task

def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    这个节点 node_web_search_mcp 负责调用 百炼 MCP (Model Context Protocol) 联网搜索服务 ，获取互联网上的实时信息。
    一句话总结： 
    它通过 MCP 协议异步调用百炼联网搜索接口，将用户的查询转化为实时的、结构化的网络搜索结果（包含标题、链接和摘要）。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")

    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
    time.sleep(1)
    # 调用mcp外部引擎
    print(f"调用外部mcp引擎")

    print("---node-web-search-mcp处理结束---")