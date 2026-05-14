import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000
def node_import_kg(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入知识图谱 (node_import_kg)
    为什么叫这个名字: 构建 Knowledge Graph (KG) 并存入 Neo4j。
    未来要实现:
    1. 调用 LLM 从文本中抽取实体 (Entity) 和关系 (Relation)。
    2. 连接 Neo4j 数据库。
    3. 执行 Cypher 语句将图谱数据写入数据库。
    这个节点 node_answer_output 是知识库查询的“最后一公里”，负责 生成最终回答 并 交付给用户 。

    它整合了之前所有步骤的成果，通过以下 5 个核心动作完成任务：

    1. 检查前置答案 ：如果之前步骤（如商品名确认节点）已经生成了追问或拒绝回答，直接输出，跳过 LLM 生成。
    2. 构建 Prompt ：将用户问题、历史对话、以及 Rerank 后的 TopK 高质量文档片段（包含元数据）组装成一段严谨的提示词。
    3. LLM 生成与流式推送 ：调用大模型生成最终答案。如果是流式模式，会**逐字推送（Delta）**给前端，实现打字机效果。
    4. 图片提取与增强 ：从引用文档中自动提取图片 URL（包括网页链接和本地 Markdown 图片），为纯文本答案补充视觉信息。
    5. 收尾与存档 ：将最终答案和提取的图片写入 MongoDB 历史记录，并向前端发送包含完整信息（答案+图片）的 FINAL 信号 。
    代码逻辑：
    1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
    2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案
    3 阶段三：调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
    4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
    5 做最后一次push操作（主要是为了触发前端图片渲染)
        {
            "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
            "status": "completed",
            "image_urls": [
                "http://local-server/images/panel_view.jpg",
                "http://local-server/images/button_detail.jpg"
            ]
        }
    
    """
    logger.info("---node_answer_output (答案生成) 节点开始处理---")
    add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  
    # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
    answer_exists = step_1_check_answer(state)
  
    # 阶段二  如果没有answer则 构建 Prompt
    if not answer_exists:
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt

    # 阶段三：  如果没有answer则 调用大模型输出答案
    step_3_generate_response(state, prompt)

    # 提取图片URL（用于历史记录和前端展示）
    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])

    # 阶段四：把答案写入到mongodb的history中
    if state.get("answer"):
        logger.info("---写入MongoDB历史记录---")
        step_4_write_history(state, image_urls=image_urls)

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  
    # 阶段五: 流式输出结束，发送 final 事件 [最后兜底，确保图片都能争取渲染和结束]
    logger.info(f"---发送 final 事件---图片为：{image_urls}")
    if state.get("is_stream"):
        push_to_session(
        state['session_id'],
        SSEEvent.FINAL,
        {
            "answer": state["answer"],
            "status": "completed",
            "image_urls": image_urls  # 发送图片URL给前端
        }
    )
  
    logger.info("---node_answer_output 节点处理结束---")
    return state
