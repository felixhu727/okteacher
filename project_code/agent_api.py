from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from datetime import datetime
from flask import send_from_directory
import time


from tripo_3Dgenerate import (
    GraphState, model_history, init_llm, intent_analysis_node,
    history_matching_node, history_model_response_node, 
    model_generation_node, create_workflow, Tripo3DGenerator
)

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局对话历史记录
conversation_history = []

# 初始化3D生成器
generator = Tripo3DGenerator()



@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/Three.html')
def three():
    return send_from_directory('.', 'Three.html')

@app.route('/evalate.html')
def evalate():
    return send_from_directory('.', 'evalate.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    处理用户输入，返回AI回复
    """
    start_time=time.time()
    try:
        # 获取JSON数据
        data = request.get_json(force=True)
        if not data or 'user_input' not in data:
            return jsonify({'error': '缺少user_input参数'}), 400
        
        user_input = data['user_input']
        
        # 处理用户请求
        result = generator.process_request(user_input)

        end_time=time.time()
        generate_time=end_time-start_time
        
        # 构建响应
        response = {
            'response': result['response'],
            'intent': result['intent'],
            'keywords': result['keywords'],
            'model_path': result.get('model_path', ''),
            'should_generate': result['should_generate'],
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'generate_time':generate_time
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'处理请求时出错: {str(e)}', 'success': False}), 500

@app.route('/api/conversation', methods=['GET'])
def get_conversation():
    """
    获取对话历史记录
    """
    try:
        # 从生成器获取对话历史
        history = generator.get_conversation_history()
        
        # 转换为前端友好的格式
        formatted_history = []
        for msg in history:
            if hasattr(msg, 'type'):
                if msg.type == 'human':
                    formatted_history.append({
                        'role': 'user',
                        'content': msg.content,
                        'type': 'human'
                    })
                elif msg.type == 'ai':
                    formatted_history.append({
                        'role': 'assistant', 
                        'content': msg.content,
                        'type': 'ai'
                    })
            else:
                formatted_history.append({
                    'role': 'system',
                    'content': str(msg),
                    'type': 'system'
                })
        
        return jsonify({
            'history': formatted_history,
            'total_messages': len(formatted_history),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'获取对话历史时出错: {str(e)}', 'success': False}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """
    清空对话历史
    """
    try:
        generator.clear_history()
        return jsonify({'message': '对话历史已清空', 'success': True})
        
    except Exception as e:
        return jsonify({'error': f'清空历史时出错: {str(e)}', 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查端点
    """
    return jsonify({
        'status': 'healthy',
        'service': '3D Model Generation API',
        'timestamp': datetime.now().isoformat(),
        'model_history_count': len(model_history)
    })

@app.route('/api/model-history', methods=['GET'])
def get_model_history():
    """
    获取模型生成历史
    """
    return jsonify({
        'total_models': len(model_history),
        'models': model_history
    })

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    # 启动Flask应用
    print("启动3D模型生成API服务...")
    app.run(host='0.0.0.0', port=5000, debug=True)
