import os
import uuid
from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import requests
import json
import time
import asyncio
from tripo3d import TripoClient
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件中的环境变量

# 定义状态管理
class GraphState(TypedDict):
    """状态定义"""
    messages: Annotated[List, "对话消息历史"]
    user_input: Annotated[str, "用户输入"]
    intent: Annotated[str, "用户意图"]
    keywords: Annotated[str, "提取的关键词"]
    model_path: Annotated[str, "生成的3D模型路径"]
    should_generate: Annotated[bool, "是否需要生成3D模型"]

    use_history_model: Annotated[bool, "是否使用历史模型"]
    history_keywords: Annotated[str, "匹配的历史关键词"]

# 全局历史记录存储
model_history = {}  

# 初始化LLM
def init_llm():
    """初始化DeepSeek LLM"""
    return ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY")  
    )



# 意图分析节点
def intent_analysis_node(state: GraphState) -> GraphState:
    """分析用户意图并提取关键词"""
    print("=== 意图分析节点 ===")
    
    llm = init_llm()
    
    # 系统提示词
    system_prompt = """你是一个3D模型生成助手。请分析用户的意图：
    1. 如果用户想要生成3D模型，请提取描述物体的关键词
    2. 如果不是，请礼貌地请求用户描述具体要生成的物体
    
    返回JSON格式：
    {
        "intent": "generate_3d" 或 "clarify",
        "keywords": "提取的关键词" 或 "",
        "response": "给用户的回复"
    }"""
    
    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_input"]}
    ]
    
    # 调用LLM
    response = llm.invoke(messages)
    
    try:
        # 解析JSON响应
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "intent": "clarify",
                "keywords": "",
                "response": "请描述具体要生成的3D物体，比如'一个白色卷毛的小狗'或'一个红色的苹果'"
            }
    except:
        result = {
            "intent": "clarify",
            "keywords": "",
            "response": "请描述具体要生成的3D物体"
        }
    
    # 更新状态
    state["intent"] = result["intent"]
    state["keywords"] = result["keywords"]
    state["should_generate"] = result["intent"] == "generate_3d"
    state["use_history_model"] = False
    state["history_keywords"] = ""
    
    # 如果不是生成意图，添加回复消息
    if not state["should_generate"]:
        state["messages"].append(AIMessage(content=result["response"]))
    
    print(f"意图分析结果: {result}")
    return state

# 历史匹配节点
def history_matching_node(state: GraphState) -> GraphState:
    """检查当前关键词是否与历史记录匹配"""
    print("=== 历史匹配节点 ===")
    
    if not state["keywords"] or not model_history:
        print("无关键词或无历史记录，跳过匹配")
        return state
    
    current_keywords = state["keywords"]
    
    # 简单的一一匹配检查
    for history_keywords, model_path in model_history.items():
        # 计算字符级别的匹配度
        match_ratio = calculate_simple_match(current_keywords, history_keywords)
        
        if match_ratio >= 0.9:  # 90%匹配度
            state["use_history_model"] = True
            state["model_path"] = model_path
            state["history_keywords"] = history_keywords
            state["should_generate"] = False
            print(f"找到历史匹配: {history_keywords} -> {model_path} (匹配度: {match_ratio:.2f})")
            return state
    
    print("未找到匹配的历史记录")
    return state

# 历史模型响应节点
def history_model_response_node(state: GraphState) -> GraphState:
    """如果找到历史匹配，直接返回历史模型"""
    print("=== 历史模型响应节点 ===")
    
    if not state["use_history_model"]:
        return state
    
    llm = init_llm()
    
    response = llm.invoke([
        {"role": "system", "content": f"你发现用户当前请求与历史生成记录高度相似。当前关键词：{state['keywords']}，历史关键词：{state['history_keywords']}。请告诉用户可以直接使用之前生成的模型，并询问是否需要重新生成。"},
        {"role": "user", "content": state["user_input"]}
    ])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"返回历史模型响应: {state['model_path']}")
    return state

# 简单匹配度计算
def calculate_simple_match(text1: str, text2: str) -> float:
    """计算两个文本的简单匹配度"""
    if not text1 or not text2:
        return 0.0
    
    # 转换为字符集合
    set1 = set(text1)
    set2 = set(text2)
    
    # 计算Jaccard相似度
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


# 模型生成节点
def model_generation_node(state: GraphState) -> GraphState:
    """调用Tripo3D生成3D模型并保存到历史记录"""
    print("=== 模型生成节点 ===")
    
    try:
        # 确保输出目录存在
        os.makedirs("./static/output", exist_ok=True)
        
        # 调用Tripo3D API
        print(f"正在为关键词 '{state['keywords']}' 生成3D模型...")
        result = tripo3d_api_call(state["keywords"], "./static/output")
        
        # 处理返回结果
        if result and result[0]:  # 检查是否成功
            success, actual_filename = result
            # 使用API实际生成的文件名构建URL
            model_path = f"http://localhost:5000/static/output/{actual_filename}"
            state["model_path"] = model_path
            
            # 保存到历史记录
            model_history[state["keywords"]] = model_path
            
            # 添加成功消息
            state["messages"].append(AIMessage(
                content=f"3D模型已生成！"
            ))
            print(f"模型生成成功并保存到历史: {model_path}")
        else:
            state["messages"].append(AIMessage(
                content="模型生成失败，请稍后重试或尝试其他描述"
            ))
            print("模型生成失败")
            
    except Exception as e:
        print(f"模型生成错误: {e}")
        state["messages"].append(AIMessage(
            content="生成过程中出现错误，请稍后重试"
        ))
    
    return state

# Tripo3D API调用函数
def tripo3d_api_call(keywords: str, output_dir: str):
    """真实的Tripo3D API调用，返回成功状态和实际文件名"""
    try:
        # 从环境变量获取API密钥
        api_key = os.getenv("TRIPOAI_API_KEY")
        if not api_key:
            print("错误: 未设置TRIPOAI_API_KEY环境变量")
            return False, None
        
        # 定义异步函数
        async def generate_model():
            async with TripoClient(api_key=api_key) as client:
                # 提交生成任务
                task_id = await client.text_to_model(
                    prompt=keywords,
                    negative_prompt="low quality, blurry",  
                )
                print(f"任务ID: {task_id}")

                # 等待任务完成
                task = await client.wait_for_task(task_id, verbose=True)
                task.status="success"
                
                if task.status == "success":
                    # 下载模型文件到指定目录
                    files = await client.download_task_models(task, output_dir)
                    print(f"下载的文件列表: {files}")
                    
                    if files and isinstance(files, dict) and 'pbr_model' in files:
                        # 获取实际生成的文件名
                        file_path = files['pbr_model']
                        actual_filename = os.path.basename(file_path)
                        print(f"实际生成的文件名: {actual_filename}")
                        print(f"模型已保存到: {os.path.join(output_dir, actual_filename)}")
                        return True, actual_filename
                    else:
                        print("未找到下载的文件或文件格式不正确")
                        return False, None
                else:
                    print(f"任务失败，状态: {task.status}")
                    return False, None

        # 运行异步函数并返回结果
        return asyncio.run(generate_model())
        
    except Exception as e:
        print(f"Tripo3D API调用错误: {e}")
        return False, None



# Tripo3D模型生成节点
def tripo3d_generation_node(state: GraphState) -> GraphState:
    """调用Tripo3D生成3D模型"""
    print("=== Tripo3D生成节点 ===")
    
    try:
        # 确保输出目录存在
        os.makedirs("./static/output", exist_ok=True)
        
        # 生成唯一文件名
        filename = f"model_{uuid.uuid4().hex[:8]}.glb"
        file_save_path = f"./static/output/{filename}"

        model_path = f"http://localhost:5000/static/output/{filename}"
        
        # 调用Tripo3D API
        print(f"正在为关键词 '{state['keywords']}' 生成3D模型...")
        success = tripo3d_api_call(state["keywords"], file_save_path)
        
        if success:
            state["model_path"] = model_path
            state["messages"].append(AIMessage(
                content=f"3D模型已生成！保存路径: {model_path}\n关键词: {state['keywords']}"
            ))
            print(f"模型生成成功: {model_path}")
        else:
            state["messages"].append(AIMessage(
                content="模型生成失败，请稍后重试或尝试其他描述"
            ))
            print("模型生成失败")
            
    except Exception as e:
        print(f"Tripo3D生成错误: {e}")
        state["messages"].append(AIMessage(
            content="生成过程中出现错误，请稍后重试"
        ))
    
    return state

# 路由判断函数
def should_generate_model(state: GraphState) -> str:
    """判断是否应该生成3D模型"""
    if state["should_generate"] and state["keywords"]:
        return "generate_3d"
    else:
        return "end"


# 工作流示例
def create_workflow():
    """创建简化的工作流"""
    workflow = StateGraph(GraphState)
    
    # 添加节点
    workflow.add_node("intent_analysis", intent_analysis_node)
    workflow.add_node("history_matching", history_matching_node)
    workflow.add_node("history_response", history_model_response_node)
    workflow.add_node("model_generation", model_generation_node)
    
    # 设置入口点
    workflow.set_entry_point("intent_analysis")
    
    # 添加边
    workflow.add_conditional_edges(
        "intent_analysis",
        lambda state: "history_matching" if state["should_generate"] else END
    )
    
    workflow.add_conditional_edges(
        "history_matching", 
        lambda state: "history_response" if state["use_history_model"] else "model_generation"
    )
    
    workflow.add_edge("history_response", END)
    workflow.add_edge("model_generation", END)
    
    return workflow.compile()
# 主执行类
class Tripo3DGenerator:
    def __init__(self):
        self.workflow = create_workflow()
        self.conversation_history = []
    
    def process_request(self, user_input: str) -> Dict:
        """处理用户请求"""
        
        # 初始化状态
        initial_state = {
            "messages": self.conversation_history.copy(),
            "user_input": user_input,
            "intent": "",
            "keywords": "",
            "model_path": "",
            "should_generate": False
        }
        
        # 添加用户消息到历史
        self.conversation_history.append(HumanMessage(content=user_input))
        
        # 执行工作流
        print(f"\n处理请求: {user_input}")
        final_state = self.workflow.invoke(initial_state)
        
        # 更新对话历史
        if final_state["messages"] and len(final_state["messages"]) > len(self.conversation_history):
            new_messages = final_state["messages"][len(self.conversation_history):]
            self.conversation_history.extend(new_messages)
        
        # 返回结果
        result = {
            "response": final_state["messages"][-1].content if final_state["messages"] else "无回复",
            "intent": final_state["intent"],
            "keywords": final_state["keywords"],
            "model_path": final_state.get("model_path", ""),
            "should_generate": final_state["should_generate"]
        }
        
        # 打印状态信息
        self.print_status(final_state)
        
        return result
    
    def print_status(self, state: GraphState):
        """打印当前状态信息"""
        print("\n=== 状态信息 ===")
        print(f"用户输入: {state['user_input']}")
        print(f"识别意图: {state['intent']}")
        print(f"提取关键词: {state['keywords']}")
        print(f"模型路径: {state.get('model_path', '未生成')}")
        print(f"需要生成: {state['should_generate']}")
        print("================\n")
    
    def get_conversation_history(self) -> List:
        """获取对话历史"""
        return self.conversation_history
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

# 使用示例
def main():
    # 初始化生成器
    generator = Tripo3DGenerator()

    print("开始对话（输入 'quit' 或 'exit' 以退出）\n")

    while True:
        user_input = input("\n用户: ")
        

        if user_input.lower() in ['quit', 'exit']:
            print("对话结束。")
            break
        result = generator.process_request(user_input)
        print(f"AI回复: {result['response']}")
        

