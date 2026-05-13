import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())
def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    1. 提取与改写 ：结合历史对话提取商品名，并将模糊问题改写为完整独立的精准问题。
    2. 向量化检索 ：将提取出的商品名在 Milvus 向量库中进行混合搜索。
    3. 标准化对齐 ：根据评分高低自动对齐标准型号，或生成反问让用户手动确认。
    4. 同步历史记录 ：将改写后的问题、确认的商品名和处理状态实时写入 MongoDB 数据库。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    logger.info(">>> node_item_name_confirm: 开始处理")
    
    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    # 标记任务开始
    add_running_task(session_id, "node_item_name_confirm", is_stream)

    # 1. 获取历史记录
    history = get_recent_messages(session_id, limit=10)
    logger.info(f"Node: 获取到 {len(history)} 条历史消息")

    # 2. 保存用户当前消息 (初始保存，后续 step 7 会更新)
    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")

    # 3. 提取信息
    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)
    
    # 更新 State 中的 rewrite_query
    state["rewritten_query"] = rewritten_query

    align_result = {}

    # 4. & 5. 如果有提取到商品名，进行搜索和对齐
    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: 未提取到商品名，跳过向量检索")

    # 6. 检查确认状态
    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)

    # 7. 写入最终历史
    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)

    # 将 history 存入 state，供后续节点（如 node_answer_output）使用
    final_state["history"] = history

    # 标记任务完成
    add_done_task(session_id, "node_item_name_confirm", is_stream)
    
    logger.info(f"Node: 处理结束, Final State Item Names: {final_state.get('item_names')}")
    return final_state