# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
import sys
from typing import List, Dict, Any, Tuple

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

from app.utils.escape_milvus_string_utils import escape_milvus_string
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
   【核心节点】商品主体名称识别（node_item_name_recognition）
    整体流程：提取输入→构建上下文→大模型识别→回填数据→生成向量→存入Milvus
    核心目的：利用大模型从文档切片中精准识别商品/主体名称，并生成双路向量（稠密+稀疏）存入数据库
    后续扩展点：支持多主体识别、增加商品属性提取、对接其他向量库等
    :param state: 项目状态字典（ImportGraphState），必须包含chunks/file_title/task_id
    :return: 更新后的状态字典，新增item_name键，且chunks列表中每个元素新增item_name字段
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【商品名称识别】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state.get("task_id", ""), node_name)
    try:
        # ===================================== 步骤1：提取并校验输入数据 =====================================
        # 作用：从状态字典提取文件标题和切片列表，校验数据完整性
        # 输出：文件标题、切片列表；若无切片则抛出异常或终止
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            logger.warning(f">>> 节点执行警告：{node_name}（无有效切片数据），跳过识别")
            return state

        # ===================================== 步骤2：构建大模型识别上下文 =====================================
        # 作用：截取前N个切片的内容，拼接成大模型可阅读的上下文，用于辅助识别
        # 输出：拼接后的上下文字符串
        context = step_2_build_context(chunks)

        # ===================================== 步骤3：调用大模型识别商品名称 =====================================
        # 作用：构造Prompt，调用LLM从上下文和标题中提取最核心的商品名称
        # 输出：识别出的商品名称字符串（如 "iPhone 15 Pro"）
        item_name = step_3_call_llm(file_title, context)

        # ===================================== 步骤4：回填商品名称到状态和切片 =====================================
        # 作用：将识别结果写入状态字典，并同步更新到每一个Chunk对象的元数据中
        # 输出：状态字典新增item_name，chunks列表被就地修改
        step_4_update_chunks(state, chunks, item_name)

        # ===================================== 步骤5：生成双路向量（稠密+稀疏） =====================================
        # 作用：调用BGE-M3模型，为商品名称生成稠密语义向量和稀疏关键词向量
        # 输出：dense_vector（List[float]）、sparse_vector（Dict[int, float]）
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)

        # ===================================== 步骤6：存入Milvus向量数据库 =====================================
        # 作用：将商品名称及其双路向量存入Milvus的 item_names 集合，用于后续检索
        # 输出：无返回值，数据已持久化
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 节点执行完成日志
        logger.info(f">>> 核心节点执行完成：【商品名称识别】{node_name}，识别结果：{item_name}，已存入Milvus")

    except Exception as e:
        # 全局异常捕获：保证节点执行失败不崩溃整个流程，记录详细错误日志便于排查
        logger.error(f">>> 核心节点执行失败：【商品名称识别】{node_name}，错误信息：{str(e)}", exc_info=True)
        # 可选：失败时设置默认值或标记状态
        state["item_name"] = "未知商品"

    # 返回更新后的状态（供下游节点使用）
    return state


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """
    步骤 1: 接收并校验流程输入（商品名称识别的前置数据处理）
    核心作用：
        1. 从流程状态中提取文件标题、文本切片核心数据
        2. 做多层空值兜底，避免后续流程因空值报错
        3. 基础数据类型校验，保证下游流程输入有效性
    依赖的状态数据（上游节点产出）：
        - state["file_title"]: 上游提取的文件标题（优先使用）
        - state["file_name"]: 原始文件名（file_title为空时兜底）
        - state["chunks"]: 文本切片列表（每个切片为字典，含title/content等字段）
    返回值：
        Tuple[str, List[Dict]]: (处理后的文件标题, 校验后的文本切片列表)
    """
    # 多层兜底获取文件标题：优先file_title → 其次file_name → 空字符串
    file_title = state.get("file_title", "") or state.get("file_name", "")
    # 获取文本切片列表：空值时返回空列表，避免后续遍历报错
    chunks = state.get("chunks") or []

    # 二次兜底：file_title仍为空时，尝试从第一个有效切片中提取
    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning("state中无有效file_title，已从第一个切片中提取兜底标题")

    # 空值日志提示：文件标题为空时不中断流程，仅记录警告
    if not file_title:
        logger.warning("state中缺少file_title和file_name，后续大模型识别可能精度下降")

    # 数据类型校验：确保chunks为有效非空列表，否则返回空列表
    if not isinstance(chunks, list) or not chunks:
        logger.warning("state中chunks为空或非列表类型，无法进行商品名称识别")
        return file_title, []

    logger.info(f"步骤1：输入校验完成，获取到{len(chunks)}个有效文本切片")
    return file_title, chunks
