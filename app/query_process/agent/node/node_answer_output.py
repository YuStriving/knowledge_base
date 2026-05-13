from .state import ImportGraphState
import sys

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
    """
    print(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    return state