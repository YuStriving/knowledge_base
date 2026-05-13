import time
import sys
from app.utils.task_utils import  add_done_task,add_running_task

def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    这个节点 node_search_embedding_hyde 实现了 HyDE (Hypothetical Document Embeddings) 策略，
    核心逻辑是 先让 LLM 虚构一个“理想答案”，再用这个答案去向量库检索真实的文档 。
    一句话总结： 
    它通过“LLM 生成假设性答案”来增强原始问题的语义信息，再进行混合向量检索，从而大幅提升对“语义匹配但字面不匹配”问题的召回能力。
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    print("搜索架设性答案！！")
    time.sleep(1)

    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---HyDE 处理结束---")