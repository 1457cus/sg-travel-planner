import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import webbrowser

# -------------------- 页面配置必须放在最前面 --------------------
st.set_page_config(page_title="韶关AI旅游助手", layout="wide")

# -------------------- 初始化设置 --------------------
print("[DEBUG] 当前工作目录:", os.getcwd())
data_path = Path("data/sg_attractions_cleaned.csv")
print("[DEBUG] 文件绝对路径:", data_path.absolute())
print("[DEBUG] 文件是否存在:", data_path.exists())

# -------------------- 环境变量处理 --------------------
load_dotenv()  # 加载 .env 文件

# 调试信息 - 检查环境变量和Secrets
print("[DEBUG] 尝试获取 DeepSeek API 密钥...")

# 打印相关环境变量
print("[DEBUG] 环境变量列表:")
for key, value in os.environ.items():
    if "DEEPSEEK" in key or "KEY" in key:
        print(f"  {key}: {value}")

# 尝试访问 secrets
try:
    print("[DEBUG] 尝试访问 Streamlit Secrets")
    print(f"[DEBUG] 可用的 Secrets 键: {list(st.secrets.keys())}")
    if "DEEPSEEK_KEY" in st.secrets:
        print(f"[DEBUG] 找到 DEEPSEEK_KEY: {st.secrets['DEEPSEEK_KEY'][:4]}...")
except Exception as e:
    print(f"[ERROR] 访问 Secrets 失败: {str(e)}")

# 尝试从多个来源获取 DeepSeek API 密钥
deepseek_api_key = None
key_sources = []

# 1. 尝试从环境变量获取
if os.getenv("DEEPSEEK_API_KEY"):
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    key_sources.append("环境变量 (DEEPSEEK_API_KEY)")
    print("[DEBUG] 从环境变量 DEEPSEEK_API_KEY 获取密钥")

if not deepseek_api_key and os.getenv("DEEPSEEK_KEY"):
    deepseek_api_key = os.getenv("DEEPSEEK_KEY")
    key_sources.append("环境变量 (DEEPSEEK_KEY)")
    print("[DEBUG] 从环境变量 DEEPSEEK_KEY 获取密钥")

# 2. 尝试从 Streamlit Secrets 获取
try:
    if st.secrets.get("DEEPSEEK_KEY"):
        deepseek_api_key = st.secrets.get("DEEPSEEK_KEY")
        key_sources.append("Streamlit Secrets")
        print("[DEBUG] 从 Streamlit Secrets 获取密钥")
except Exception as secrets_error:
    print(f"[WARNING] Streamlit Secrets 访问失败: {str(secrets_error)}")

# 3. 尝试从 .env 文件获取（dotenv 已加载）
if not deepseek_api_key and os.getenv("DEEPSEEK_API_KEY"):
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    key_sources.append(".env 文件")
    print("[DEBUG] 从 .env 文件获取密钥")

# 4. 如果以上都没有，将在侧边栏让用户输入
if not deepseek_api_key:
    st.warning("⚠️ 未找到 DeepSeek API 密钥")
    st.info("""
        **请提供您的 DeepSeek API 密钥：**
        1. 在侧边栏输入密钥
        2. 或创建 secrets.toml 文件
        3. 或设置环境变量
        
        [获取 DeepSeek API 密钥](https://platform.deepseek.com/)
    """)
    key_sources.append("用户输入")
    print("[WARNING] 未找到任何来源的 API 密钥")

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
    
    # 如果尚未获取到密钥，显示输入框
    if not deepseek_api_key:
        deepseek_api_key = st.text_input(
            "🔑 输入 DeepSeek API 密钥", 
            type="password",
            help="可在 https://platform.deepseek.com/ 获取"
        )
        if not deepseek_api_key:
            st.error("请提供 API 密钥以继续")
            st.stop()
        st.success("✅ API 密钥已输入")
        key_sources = ["用户输入"]  # 重置来源
    else:
        st.success(f"✅ API 密钥已通过 {', '.join(key_sources)} 获取")
    
    st.info(f"当前模型: {MODEL_NAME}")
    
    # 使用按钮代替 link_button
    if st.button("🌐 检查 DeepSeek 状态", key="status_check_button"):
        webbrowser.open_new_tab("https://platform.deepseek.com/api")
        st.toast("已在浏览器中打开 DeepSeek API 文档")
    
    # 添加创建 secrets.toml 的说明
    st.divider()
    st.markdown("""
        **创建 secrets.toml 文件:**
        1. 在项目根目录创建 `.streamlit` 文件夹
        2. 在 `.streamlit` 文件夹中创建 `secrets.toml` 文件
        3. 添加以下内容:
        ```
        DEEPSEEK_KEY = "您的API密钥"
        ```
    """)
    st.markdown(f"""
        **当前项目路径:** `{os.getcwd()}`
        **Secrets 预期路径:** `{os.getcwd()}\\.streamlit\\secrets.toml`
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
if deepseek_api_key:
    client = DeepSeekClient(api_key=deepseek_api_key)
else:
    st.error("未设置 API 密钥，无法创建客户端")
    st.stop()

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
            "data/sg_attractions_cleaned.csv",
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
            "data/sg_food_cleaned.csv",
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
            "data/sg_culture_cleaned.csv",
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="warn"
        )
        culture.columns = [clean_text(col) for col in culture.columns]
        culture = culture.applymap(clean_text)
        
        return attractions, foods, culture

    except Exception as e:
        st.error(f"数据加载失败：{str(e)}")
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
        with open("prompt_template.txt", "r", encoding="utf-8") as f:
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
print("[DEBUG] 景点数据样例：\n", attractions.head(2).to_string())
print("[DEBUG] 美食数据样例：\n", foods.sample(2).to_string())
if len(culture) > 0:
    print("[DEBUG] 文化数据样例：\n", culture.sample(1).to_string())
else:
    print("[DEBUG] 无文化数据")
    
# 添加页脚
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 30px;">
        <p>韶关 AI 旅游助手 v1.3 | 技术支持: 韶关市旅游局</p>
        <p>© 2025 韶关智慧旅游项目 | 使用 DeepSeek API</p>
    </div>
""", unsafe_allow_html=True)