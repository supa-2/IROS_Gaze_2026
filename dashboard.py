import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import json
import time

# --- 引入我们的核心模块 ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'skills', 'topology'))
from graph_engine import TopologyEngine
# 引入 Agent (即使没有 Key，我们先引用进来，后面做个 Mock 模式)
try:
    from agent_core import EyeLLMAgent
except ImportError:
    pass

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="Eye-LLM Control Center")

st.title("👁️ Eye-LLM: Spatial Intent Prediction System")
st.markdown("### IROS 2026 Project Demo | Gaze-to-Action Architecture")

# --- 侧边栏：设置 ---
st.sidebar.header("1. Environment Setup")
map_option = st.sidebar.selectbox("Choose Map / Project", ["TH", "OS"])
model_option = st.sidebar.selectbox("LLM Model", ["gpt-4o", "qwen2.5:7b (Local)", "Mock (Testing)"])

# 初始化地图引擎
@st.cache_resource 
def load_engine(map_name):
    return TopologyEngine(map_name)

try:
    engine = load_engine(map_option)
    st.sidebar.success(f"Map '{map_option}' Loaded: {len(engine.graph.nodes)} nodes")
except Exception as e:
    st.sidebar.error(f"Failed to load map: {e}")
    st.stop()

# --- 侧边栏：模拟输入 ---
st.sidebar.header("2. User Simulation")
# 获取所有节点供选择
all_nodes = list(engine.graph.nodes())
# 简单的搜索框
current_gaze = st.sidebar.selectbox("Current Gaze Fixation (User Location)", all_nodes)

# 模拟历史路径
if 'history' not in st.session_state:
    st.session_state.history = [current_gaze]

if st.sidebar.button("Step Forward (Add to History)"):
    st.session_state.history.append(current_gaze)

st.sidebar.write("Gaze History:", st.session_state.history)

if st.sidebar.button("Clear History"):
    st.session_state.history = [current_gaze]

# --- 主界面：地图可视化 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Spatial Topology Visualization")
    
    # 绘图逻辑
    fig, ax = plt.subplots(figsize=(10, 6))
    G = engine.graph
    
    # 布局算法 (Spring Layout)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # 1. 绘制底图
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#DDDDDD', alpha=0.8)
    
    # 2. 高亮：历史路径 (蓝色)
    if len(st.session_state.history) > 0:
        nx.draw_networkx_nodes(G, pos, nodelist=st.session_state.history, node_color='blue', node_size=400, label="History")
        # 画轨迹线
        path_edges = list(zip(st.session_state.history, st.session_state.history[1:]))
        if path_edges:
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2, alpha=0.6)

    # 3. 高亮：当前位置 (红色)
    nx.draw_networkx_nodes(G, pos, nodelist=[current_gaze], node_color='red', node_size=600, label="Current")
    
    # 标签
    # 只显示当前点周围的标签，防止太乱
    labels = {n: n for n in G.nodes if n in st.session_state.history or n in list(G.neighbors(current_gaze))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family='sans-serif')
    
    st.pyplot(fig)

# --- 右侧栏：Agent 推理 ---
with col2:
    st.subheader("🧠 Agent Reasoning")
    
    if st.button("🔮 Predict Next Intention", type="primary"):
        with st.spinner("Agent is thinking..."):
            # 这里调用真实的 Agent
            try:
                if model_option == "Mock (Testing)":
                    # 模拟返回，无需 Key
                    time.sleep(1) # 假装思考
                    # 简单规则：找第一个邻居
                    neighbors = list(G.neighbors(current_gaze))
                    pred = neighbors[0] if neighbors else "None"
                    result = {
                        "prediction_id": pred,
                        "reasoning": "Mock Mode: Selected first neighbor based on flow logic."
                    }
                    response_content = json.dumps(result)
                else:
                    # 真实调用 (需要你填好 Key 或者配置好 Ollama)
                    agent = EyeLLMAgent(map_name=map_option, model_name=model_option)
                    response_content = agent.predict_next_gaze(st.session_state.history)
                
                # 解析显示
                try:
                    res_json = json.loads(response_content)
                    
                    st.success(f"Prediction: {res_json.get('prediction_id')}")
                    
                    st.info("Reasoning Trace:")
                    st.write(res_json.get('reasoning'))
                    
                    # 4. 在图上高亮预测点 (绿色)
                    # (由于 matplotlib 是静态的，这里需要重新绘制，简单起见我们直接文字提示)
                    st.caption(f"Visual feedback updated: Green node {res_json.get('prediction_id')} would be highlighted.")
                    
                except Exception as e:
                    st.error(f"Parsing Error: {response_content}")
                    
            except Exception as e:
                st.error(f"Agent Error: {str(e)}")
                st.warning("Hint: Check your API Key or Local Model connection.")

    # 显示环境上下文 (Context)
    with st.expander("查看当前节点的拓扑上下文 (Affordances)"):
        node_data = engine.query_node(current_gaze)
        st.json(node_data)