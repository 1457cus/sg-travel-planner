import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import webbrowser

# -------------------- 路径配置必须放在最前面 --------------------
# 获取当前文件所在目录
current_dir = Path(__file__).parent

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="韶关AI旅游助手", layout="wide")

# -------------------- 初始化设置 --------------------
print("[DEBUG] 当前工作目录:", os.getcwd())
print("[DEBUG] 当前文件目录:", current_dir)

# 构建数据文件路径
data_dir = current_dir / "data"
attractions_path = data_dir / "sg_attractions_cleaned.csv"
food_path = data_dir / "sg_food_cleaned.csv"
culture_path = data_dir / "sg_culture_cleaned.csv"

print("[DEBUG] 景点文件路径:", attractions_path)
print("[DEBUG] 美食文件路径:", food_path)
print("[DEBUG] 文化文件路径:", culture_path)

# -------------------- 环境变量处理 --------------------
# 在 Streamlit Cloud 上只使用 secrets
print("[DEBUG] 尝试获取 DeepSeek API 密钥...")
deepseek_api_key = st.secrets.get("DEEPSEEK_KEY", None)

if not deepseek_api_key:
    st.error("未找到 DeepSeek API 密钥，请检查 Streamlit Secrets 设置")
    st.stop()
else:
    print(f"[DEBUG] 从 Streamlit Secrets 获取密钥: {deepseek_api_key[:4]}...")

# -------------------- DeepSeek API 配置 --------------------
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# 在侧边栏添加配置信息
with st.sidebar:
    st.header("旅行参数")
    days = st.slider("旅行天数", 1, 7, 3)
    budget = st.number_input("预算（元）", 500, 10000, 1500)
    interest = st.selectbox("兴趣主题", ["历史", "自然", "美食", "亲子"])
    
    st.divider()
    st.header("API设置")
    
    st.success(f"✅ API 密钥已通过 Streamlit Secrets 获取")
    st.info(f"当前模型: {MODEL_NAME}")
    
    # 使用按钮代替 link_button
    if st.button("🌐 检查 DeepSeek 状态", key="status_check_button"):
        webbrowser.open_new_tab("https://platform.deepseek.com/api")
        st.toast("已在浏览器中打开 DeepSeek API 文档")
    
    # 添加 Streamlit Cloud 说明
    st.divider()
    st.markdown("""
        **Streamlit Cloud 说明:**
        1. API 密钥通过 Secrets 管理
        2. 所有数据文件必须上传到 GitHub
        3. 文件路径已适配云端环境
    """)

# 创建 DeepSeek API 客户端
class DeepSeekClient:
    def __init__(self, api_key, base_url=DEEPSEEK_API_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"[DEBUG] 使用 API 密钥: {api_key[:4]}...")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException))
    )
    def chat_completions(self, model, messages, temperature=0.7, max_tokens=2000):
        """调用 DeepSeek 聊天完成 API"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            print(f"[DEBUG] 发送请求到: {url}")
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"API 请求失败: HTTP {e.response.status_code}")
            st.json(e.response.json())
            raise
        except httpx.RequestError as e:
            st.error(f"网络连接错误: {str(e)}")
            raise

# 创建客户端实例
client = DeepSeekClient(api_key=deepseek_api_key)

# 测试 API 连接
try:
    print("[DEBUG] 测试 API 连接...")
    test_response = client.chat_completions(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "测试连接，请回复'连接成功'"}],
        max_tokens=10
    )
    if 'choices' in test_response and len(test_response['choices']) > 0:
        test_message = test_response['choices'][0]['message']['content']
        print(f"[DEBUG] API 连接测试成功: {test_message}")
        st.sidebar.success("✅ 连接测试成功")
    else:
        st.sidebar.error("❌ API 响应格式异常")
        st.sidebar.json(test_response)
except Exception as e:
    st.sidebar.error(f"❌ 连接测试失败: {str(e)}")
    st.sidebar.warning("请检查 API 密钥和网络连接")

# -------------------- 数据加载与预处理 --------------------
def clean_text(text):
    """清理文本中的非法字符"""
    if isinstance(text, str):
        return text.encode('utf-8', 'ignore').decode('utf-8')
    return text

def load_and_preprocess_data():
    """加载并预处理景点、美食、文化数据"""
    try:
        print("[DEBUG] 加载景点数据...")
        # 景点数据
        attractions = pd.read_csv(
            attractions_path,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="warn"
        )
        attractions.columns = [clean_text(col) for col in attractions.columns]
        attractions = attractions.applymap(clean_text)
        attractions["景点特色说明"] = attractions["景点特色说明"].fillna("暂无特色说明").astype(str)
        
        print("[DEBUG] 加载美食数据...")
        # 美食数据
        foods = pd.read_csv(
            food_path,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="warn"
        )
        foods.columns = [clean_text(col) for col in foods.columns]
        foods = foods.applymap(clean_text)
        foods["特色菜"] = foods["特色菜"].fillna("暂无推荐菜")
        
        print("[DEBUG] 加载文化数据...")
        # 文化数据
        culture = pd.read_csv(
            culture_path,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="warn"
        )
        culture.columns = [clean_text(col) for col in culture.columns]
        culture = culture.applymap(clean_text)
        
        return attractions, foods, culture

    except Exception as e:
        st.error(f"数据加载失败：{str(e)}")
        # 添加详细错误信息
        st.error(f"文件路径: {attractions_path}")
        st.error(f"当前目录内容: {os.listdir(current_dir)}")
        st.error(f"data目录内容: {os.listdir(data_dir) if os.path.exists(data_dir) else '目录不存在'}")
        st.stop()

# 加载数据
attractions, foods, culture = load_and_preprocess_data()

# 数据完整性检查
if len(attractions) == 0:
    st.error("景点数据为空，请检查数据文件！")
    st.stop()

if len(foods) == 0:
    st.error("美食数据为空，请检查数据文件！")
    st.stop()

print(f"[DEBUG] 景点记录数：{len(attractions)}")
print(f"[DEBUG] 美食记录数：{len(foods)}")
print(f"[DEBUG] 文化记录数：{len(culture)}")

# -------------------- Streamlit 界面设计 --------------------
st.title("🚩 韶关个性化旅游攻略生成器")
st.markdown("""
    <style>
    /* 主按钮样式 */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
        transition: transform 0.2s;
    }
    
    /* 状态按钮样式 */
    .status-button {
        background-color: #2196F3 !important;
    }
    
    /* 错误信息样式 */
    .stAlert {
        border-left: 4px solid #ff4b4b;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- 动态生成提示词 --------------------
def build_prompt(days, budget, interest):
    """构建 DeepSeek 提示词模板"""
    try:
        print("[DEBUG] 构建提示词...")
        template_path = current_dir / "prompt_template.txt"
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # 安全抽样景点（最多3个）
        sample_size = min(3, len(attractions))
        sampled_attractions = attractions.sample(sample_size) if sample_size > 0 else attractions
        attractions_info = [
            f"{row['名称']}（{row.get('景点特色说明', '暂无说明')}" 
            for _, row in sampled_attractions.iterrows()
        ]

        # 安全抽样餐厅（最多2个）
        sample_size = min(2, len(foods))
        sampled_foods = foods.sample(sample_size) if sample_size > 0 else foods
        food_info = [
            f"{row['店名']}（人均{row.get('人均消费', '?')}元）"
            for _, row in sampled_foods.iterrows()
        ]

        # 安全处理文化体验
        if len(culture) > 0:
            cultural_activity = culture.sample(1).iloc[0]["名称"]
        else:
            cultural_activity = "自由探索当地文化"

        return template.format(
            days=days,
            budget=budget,
            interest=interest,
            attractions="、".join(attractions_info),
            food="、".join(food_info),
            culture=cultural_activity
        )

    except Exception as e:
        st.error(f"提示词生成失败：{str(e)}")
        st.stop()

# -------------------- 生成攻略逻辑 --------------------
def get_ai_response(prompt):
    """调用 DeepSeek API 获取响应"""
    try:
        print("[DEBUG] 调用 DeepSeek API...")
        response = client.chat_completions(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000
        )
        return response
    except Exception as e:
        print(f"[ERROR] API 调用失败: {str(e)}")
        raise

if st.button("✨ 一键生成攻略", key="generate_button"):
    with st.spinner("AI 正在规划行程..."):
        try:
            prompt = build_prompt(days, budget, interest)
            print("[DEBUG] 生成的提示词：\n", prompt)
            
            start_time = time.time()
            response = get_ai_response(prompt)
            elapsed = time.time() - start_time
            print(f"[DEBUG] API 响应时间: {elapsed:.2f} 秒")
            
            if 'choices' in response and len(response['choices']) > 0:
                itinerary = response['choices'][0]['message']['content']
                st.success("✅ 攻略生成成功！")
                st.markdown(itinerary)
                
                # 添加下载按钮
                st.download_button(
                    "📥 下载攻略", 
                    itinerary, 
                    file_name=f"韶关{days}日{interest}主题旅游攻略.md",
                    mime="text/markdown",
                    key="download_button"
                )
            else:
                st.error("API 响应格式异常，无法获取攻略内容")
                st.json(response)  # 显示原始响应用于调试
            
        except Exception as e:
            st.error(f"生成失败：{str(e)}")
            st.markdown("""
                **请尝试以下解决方法：**
                1. 检查 API 密钥是否正确
                2. 确保数据文件存在且格式正确
                3. 稍后再试（API 服务可能暂时不可用）
            """)
            
            if st.button("🌐 检查 DeepSeek 状态", key="status_button", help="点击在浏览器中打开 DeepSeek API 文档"):
                webbrowser.open_new_tab("https://platform.deepseek.com/api")
                st.toast("已在浏览器中打开 DeepSeek API 文档")

# -------------------- 调试信息 --------------------
# 添加文件路径显示
st.sidebar.divider()
st.sidebar.subheader("调试信息")
st.sidebar.write(f"当前目录: {current_dir}")
st.sidebar.write(f"数据目录: {data_dir}")

# 添加页脚
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 30px;">
        <p>韶关 AI 旅游助手 v1.4 | 技术支持: 韶关市旅游局</p>
        <p>© 2025 韶关智慧旅游项目 | 使用 DeepSeek API</p>
        <p>部署环境: Streamlit Cloud</p>
    </div>
""", unsafe_allow_html=True)