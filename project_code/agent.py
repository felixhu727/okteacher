# agent.py
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
import logging
from langchain_community.chat_models import ChatHunyuan




# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_rag_agent():
    """初始化RAG代理并返回agent_executor"""
    try:
        load_dotenv()
        
        # 记录环境变量状态（不记录敏感信息）
        logger.info("初始化RAG代理...")
        # logger.info(f"TENCENT_SECRET_ID 存在: {'TENCENT_SECRET_ID' in os.environ}")
        # logger.info(f"TENCENT_SECRET_KEY 存在: {'TENCENT_SECRET_KEY' in os.environ}")


        # # 初始化混元大模型
        # llm = ChatHunyuan(
        #     hunyuan_app_id=1370946641,
        #     hunyuan_secret_id=os.environ["TENCENT_SECRET_ID"],  
        #     hunyuan_secret_key=os.environ["TENCENT_SECRET_KEY"],  
        #     streaming=True,
        # )


        logger.info(f"DEEPSEEK_API_KEY 存在: {'DEEPSEEK_API_KEY' in os.environ}")
        llm = init_chat_model("deepseek-chat", model_provider="deepseek")

        embeddings = DeterministicFakeEmbedding(size=4096)
        vector_store = InMemoryVectorStore(embeddings)
        
        # 获取数据-split-embed-存到vector_store中
        logger.info("加载文档数据...")
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        
        # Index chunks
        logger.info("将文档添加到向量存储...")
        _ = vector_store.add_documents(documents=all_splits)
        
        # 工具：query查询关键词->retrieved_docs文档中的相关内容
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        memory = MemorySaver()
        agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
        
        logger.info("RAG代理初始化成功")
        return agent_executor
        
    except Exception as e:
        logger.error(f"RAG代理初始化失败: {str(e)}")
        raise

def process_message(agent_executor, message, thread_id="default_thread"):
    """处理单条消息并返回响应"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # 调用代理处理消息
        result = agent_executor.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )
        
        # 提取AI响应
        ai_response = result["messages"][-1].content
        return ai_response
        
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}")
        raise

def stream_messages(agent_executor, message, thread_id="default_thread"):
    """流式处理消息（生成器）"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # 流式调用代理
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values", 
            config=config
        ):
            # 只返回内容部分
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    yield last_message.content
                    
    except Exception as e:
        logger.error(f"流式处理消息时出错: {str(e)}")
        yield f"错误: {str(e)}"

# 只有在直接运行此文件时才执行终端聊天界面
if __name__ == "__main__":
    try:
        # 尝试导入IPython（仅用于终端模式）
        from IPython.display import Image, display
        
        agent_executor = initialize_rag_agent()
        
        # 画图
        try:
            display(Image(agent_executor.get_graph().draw_mermaid_png()))
        except Exception as e:
            print(f"无法显示图形: {e}")
            
        # 运行终端聊天界面
        print("开始对话（输入 'quit' 或 'exit' 以退出）\n")
        while True:
            user_input = input("\n用户: ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("对话结束。")
                break
                
            try:
                # 使用默认线程ID
                response = process_message(agent_executor, user_input, "terminal_session")
                print(f"\n助手: {response}")
            except Exception as e:
                print(f"\n处理消息时出错: {e}")
                
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请检查环境变量和依赖项")