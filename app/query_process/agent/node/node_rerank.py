import time
import sys
from app.utils.task_utils import add_running_task, add_done_task

def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    1. 合并文档 ：将来自 RRF（本地检索）和 Web Search（联网搜索）的文档合并到一个池子中。
    2. 精确打分 ：使用重排序模型计算每个文档与用户问题的相关性得分。
    3. 动态截断 ：根据得分的“断崖式下跌”点，智能截取 TopK（最多 10 条），只保留高质量结果，过滤凑数的低分文档。
    """
    print("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    time.sleep(1)
    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))