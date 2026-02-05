import sys
import os
import json
import time
# 引用 agent 代码 (根据你的目录结构调整)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent_core import EyeLLMAgent

def run_benchmark(data_path, map_name='TH'):
    # 1. 加载测试数据
    with open(data_path, 'r', encoding='utf-8') as f:
        test_dataset = json.load(f)
        
    # 2. 初始化 Agent
    # 注意：这里我们还没 Key，所以实际运行会报错，但逻辑是通的
    try:
        agent = EyeLLMAgent(map_name=map_name)
    except Exception:
        print("警告：Agent 初始化失败（可能是缺 API Key），仅演示流程。")
        return

    total_predictions = 0
    correct_predictions = 0
    results_log = []

    print(f"开始评测，共 {len(test_dataset)} 条轨迹...")

    # 3. 循环测试
    for case in test_dataset:
        full_path = case['path']
        if len(full_path) < 3: continue # 轨迹太短没法测
        
        # 切片：用前 N-1 个点预测第 N 个点
        # 例如路径 [A, B, C, D]
        # 测试 1: 输入 [A, B], 真实值 C
        # 测试 2: 输入 [A, B, C], 真实值 D
        
        for i in range(2, len(full_path)):
            history = full_path[:i]
            ground_truth = full_path[i]
            
            print(f"\nCase {case['walker_id']} Step {i}: History={history[-2:]} -> GT={ground_truth}")
            
            try:
                # 调用你的 Agent
                response_json = agent.predict_next_gaze(history)
                # 解析 LLM 返回的 JSON 字符串
                # (这里假设 LLM 很听话返回了标准 JSON，实际可能需要更强的解析器)
                pred_data = json.loads(response_json)
                predicted_id = pred_data.get('prediction_id')
                
                # 判定
                is_correct = (predicted_id == ground_truth)
                if is_correct:
                    correct_predictions += 1
                    print("✅ 正确")
                else:
                    print(f"❌ 错误 (预测: {predicted_id})")
                
                total_predictions += 1
                
            except Exception as e:
                print(f"⚠️ 推理错误: {e}")
                
    # 4. 输出最终分数
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n{'='*30}")
        print(f"最终准确率 (Accuracy): {accuracy:.2f}%")
        print(f"总样本数: {total_predictions}")
        print(f"{'='*30}")

if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), 'synthetic_test_data.json')
    run_benchmark(data_file)