import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if not hasattr(np, 'bool'):
    np.bool = bool

def setup_chinese_font():
    """设置中文字体（云端优先加载本地fonts目录内的CJK字体）"""
    try:
        import os
        import matplotlib.font_manager as fm

        # 优先尝试系统已安装字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'SimHei',
            'Microsoft YaHei',
            'PingFang SC',
            'Hiragino Sans GB',
            'Noto Sans CJK SC',
            'Source Han Sans SC'
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"使用中文字体: {font}")
                return font

        # 若系统无中文字体，尝试从./fonts 目录加载随应用打包的字体
        candidates = [
            'NotoSansSC-Regular.otf',
            'NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf',
            'SimHei.ttf',
            'MicrosoftYaHei.ttf'
        ]
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        if os.path.isdir(fonts_dir):
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    try:
                        fm.fontManager.addfont(fpath)
                        fp = fm.FontProperties(fname=fpath)
                        fam = fp.get_name()
                        matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        print(f"使用本地打包字体: {fam} ({fname})")
                        return fam
                    except Exception as ie:
                        print(f"加载本地字体失败 {fname}: {ie}")

        # 兜底：使用英文字体（中文将显示为方框）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("未找到中文字体，使用默认英文字体")
        return None

    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False # 确保可以显示负号

# ==============================================================================
# 自定义CSS样式 - 医疗专业风格
# ==============================================================================
st.markdown("""
<style>
    /* 整体背景 */
    .main {
        background-color: #f8fafc;
    }
    
    /* 标题样式 */
    h1 {
        color: #1e40af;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    
    h2 {
        color: #1e3a8a;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #334155;
        font-weight: 600;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    
    /* 输入区域卡片 */
    .input-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* 结果展示卡片 */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .result-card-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .result-card-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* 概率数值样式 */
    .probability-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    /* 风险徽章 */
    .risk-badge {
        display: inline-block;
        padding: 0.75rem 2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .badge-low {
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    
    .badge-medium {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }
    
    .badge-high {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    
    /* 建议卡片 */
    .advice-card {
        background-color: white;
        border-left: 5px solid;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .advice-low {
        border-left-color: #10b981;
    }
    
    .advice-medium {
        border-left-color: #f59e0b;
    }
    
    .advice-high {
        border-left-color: #ef4444;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 0.75rem 3rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.4);
    }
    
    /* 指标标签 */
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    /* 分隔线 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. 项目名称和配置 
# ==============================================================================
st.set_page_config(
    page_title="基于随机森林算法的AML患儿allo-HSCT后并发II-IV度aGVHD预测模型",
    page_icon="🩸", 
    layout="wide"
)

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 


global feature_names_display, feature_dict, variable_descriptions


feature_names_display = [
    'TBI',                # 预处理方案包括全身照射
    'Tx_post_Relapse',    # 复发后移植
    'HLA',                # 人类白细胞抗原
    'Dx_to_Tx_Time',      # 诊断至移植时间
    'Ferritin',           # 铁蛋白
    'TBIL'                # 总胆红素
]

# 6个特征的中文名称
feature_names_cn = [
    '预处理方案包括全身照射', 
    '复发后移植', 
    '人类白细胞抗原', 
    '诊断至移植时间', 
    '铁蛋白', 
    '总胆红素'
]

# 用于英文键名到中文显示名的映射
feature_dict = dict(zip(feature_names_display, feature_names_cn))

# 变量说明字典：键名已修改为模型要求的格式
variable_descriptions = {
    'TBI': '预处理方案包括全身照射（0=否，1=是）',
    'Tx_post_Relapse': '复发后移植（0=否，1=是）',
    'HLA': '人类白细胞抗原（0=HLA非全相合，1=HLA全相合）',
    'Dx_to_Tx_Time': '诊断至移植时间（月）',
    'Ferritin': '铁蛋白（ng/mL）',
    'TBIL': '总胆红素（μmol/L）'
}

@st.cache_resource
def load_model(model_path: str = './rf_model.pkl'):
    """加载模型文件，优先使用joblib，其次pickle"""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # 尝试获取模型内部特征名
        model_feature_names = None
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            try:
                # 针对XGBoost/LightGBM等尝试获取booster
                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None:
                    model_feature_names = booster.feature_names
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型，请检查文件路径和格式: {e}")


def main():
    global feature_names_display, feature_dict, variable_descriptions

    # ==============================================================================
    # 2. 侧边栏和主标题 
    # ==============================================================================
    # 侧边栏标题 - 移除图片，优化样式
    st.sidebar.markdown("""
        <div style="padding: 1rem 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 1rem;">
            <h2 style="color: #1e40af; font-size: 1.25rem; margin: 0;">📊 系统导航</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div style="background-color: #dbeafe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="color: #1e40af; margin: 0 0 0.5rem 0; font-size: 1rem;">关于本系统</h3>
            <p style="color: #334155; font-size: 0.875rem; margin: 0; line-height: 1.5;">
                这是一个基于随机森林算法的<strong>AML患儿allo-HSCT后并发II-IV度急性移植物抗宿主病（aGVHD）</strong>预测系统，用于评估患儿发生aGVHD的风险。
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
        <div style="background-color: #f0fdf4; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #bbf7d0;">
            <h3 style="color: #166534; margin: 0 0 0.5rem 0; font-size: 1rem;">预测结果</h3>
            <p style="color: #334155; font-size: 0.875rem; margin: 0; line-height: 1.5;">
                系统输出：<br/>
                • <strong>II-IV度aGVHD</strong>发生概率<br/>
                • 未发生<strong>II-IV度aGVHD</strong>概率<br/>
                • 风险分层（低/中/高）
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 添加变量说明到侧边栏
    with st.sidebar.expander("📋 变量说明"):
        for feature in feature_names_display:
            st.markdown(f"""
                <div style="background-color: #f8fafc; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 3px solid #3b82f6;">
                    <strong style="color: #1e40af;">{feature_dict.get(feature, feature)}</strong><br/>
                    <span style="color: #64748b; font-size: 0.875rem;">{variable_descriptions.get(feature, '无详细说明')}</span>
                </div>
            """, unsafe_allow_html=True)

    # 主页面标题 - 优化样式
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 16px; margin-bottom: 2rem;">
            <h1 style="color: #1e40af; margin: 0; font-size: 2rem;">基于随机森林算法的AML患儿allo-HSCT后并发II-IV度aGVHD预测模型</h1>
            <p style="color: #3b82f6; margin-top: 0.5rem; font-size: 1.1rem;">请在下方录入全部特征后进行预测</p>
        </div>
    """, unsafe_allow_html=True)

    # 加载模型
    try:
        model, model_feature_names = load_model('./rf_model.pkl')
        st.sidebar.success("✅ 模型加载成功")
    except Exception as e:
        st.sidebar.error(f"❌ 模型加载失败: {e}")
        return


    # ==============================================================================
    # 3. 特征输入控件 - 卡片式布局
    # ==============================================================================
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 📝 患者指标录入")
    
    # 使用 3 列布局来容纳 6 个特征 (3x2=6)
    col1, col2, col3 = st.columns(3) 
    
    # 类别变量的格式化函数 - 完全保持您原来的格式
    to_cn_tbi = lambda x: "是" if x == 1 else "否"
    to_cn_relapse = lambda x: "是" if x == 1 else "否"
    to_cn_hla = lambda x: "HLA全相合" if x == 1 else "HLA非全相合"

    # --- 第 1 列 (特征 1-2) --- 保持您的原始变量名
    with col1:
        # 1. TBI（预处理方案包括全身照射）：0/1
        tbi = st.selectbox("预处理方案包括全身照射", options=[0, 1], format_func=to_cn_tbi, index=0, key='tbi') 
        # 2. Tx_post_Relapse（复发后移植）：0/1
        tx_post_relapse = st.selectbox("复发后移植", options=[0, 1], format_func=to_cn_relapse, index=0, key='relapse') 

    # --- 第 2 列 (特征 3-4) --- 保持您的原始变量名
    with col2:
        # 3. HLA（人类白细胞抗原）：0=HLA非全相合，1=HLA全相合
        hla = st.selectbox("人类白细胞抗原", options=[0, 1], format_func=to_cn_hla, index=0, key='hla')
        # 4. Dx_to_Tx_Time（诊断至移植时间）：月
        dx_to_tx_time = st.number_input("诊断至移植时间（月）", value=6.0, step=0.1, min_value=0.0, max_value=120.0, key='dx_time')

    # --- 第 3 列 (特征 5-6) --- 保持您的原始变量名
    with col3:
        # 5. Ferritin（铁蛋白）：ng/mL
        ferritin = st.number_input("铁蛋白（ng/mL）", value=1000.0, step=1.0, min_value=0.0, key='ferritin')
        # 6. TBIL（总胆红素）：μmol/L
        tbil = st.number_input("总胆红素（μmol/L）", value=20.0, step=0.1, min_value=0.0, key='tbil')
    
    st.markdown('</div>', unsafe_allow_html=True)

    # 预测按钮 - 居中显示
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
    with col_btn_center:
        predict_button = st.button("开始预测", type="primary", use_container_width=True)

    if predict_button:
        # 根据模型的特征顺序构建输入DataFrame - 保持您的原始变量名
        user_inputs = {
            'TBI': tbi,
            'Tx_post_Relapse': tx_post_relapse,
            'HLA': hla,
            'Dx_to_Tx_Time': dx_to_tx_time,
            'Ferritin': ferritin,
            'TBIL': tbil,
        }

        # 特征对齐逻辑 - 完全保持您的原始代码
        if model_feature_names:
            # 简化特征名映射（假设模型特征名与 feature_names_display 相似）
            alias_to_user_key = {f: f for f in feature_names_display}
            
            resolved_values = []
            missing_features = []
            for c in model_feature_names: # 遍历模型要求的特征名
                ui_key = alias_to_user_key.get(c, c) 
                val = user_inputs.get(ui_key, user_inputs.get(c, None)) 
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)

            if missing_features:
                st.error(f"以下模型特征未在页面录入或名称不匹配：{missing_features}。\n请核对特征名（注意大小写）。")
                with st.expander("调试信息：模型与输入特征名对比"):
                    st.write("模型特征名：", model_feature_names)
                    st.write("页面输入键：", list(user_inputs.keys()))
                return

            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:
            # 如果无法获取模型特征名，则使用 feature_names_display 顺序
            ordered_cols = feature_names_display
            input_df = pd.DataFrame([[user_inputs[c] for c in ordered_cols]], columns=ordered_cols)

        # 简单检查缺失
        if input_df.isnull().any().any():
            st.error("存在缺失的输入值，请完善后重试。")
            return

        # 确保 input_df 中的数据类型为数字
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            except Exception:
                pass

        # 进行预测（概率）- 完全保持您的原始代码
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                # 假设第1列为阴性（未发生），第2列为阳性（发生）
                if len(proba) == 2:
                    no_agvhd_prob = float(proba[0])
                    agvhd_prob = float(proba[1]) # II-IV度aGVHD发生概率
                else:
                    raise ValueError("predict_proba返回的维度异常")
            else:
                # 预测失败的退路，概率近似
                if hasattr(model, 'decision_function'):
                    score = float(model.decision_function(input_df))
                    agvhd_prob = 1 / (1 + np.exp(-score))
                    no_agvhd_prob = 1 - agvhd_prob
                else:
                    pred = int(model.predict(input_df)[0])
                    agvhd_prob = float(pred)
                    no_agvhd_prob = 1 - agvhd_prob

            # 风险分层 - 保持您的原始阈值
            risk_level = "低风险" if agvhd_prob < 0.3 else ("中等风险" if agvhd_prob < 0.7 else "高风险")
            risk_color_class = "low" if agvhd_prob < 0.3 else ("medium" if agvhd_prob < 0.7 else "high")
            
            # 显示预测结果 - 卡片式布局（保持您的原始文字）
            st.markdown("---")
            st.markdown("## II-IV度aGVHD风险预测结果")
            
            # 概率展示卡片 - 保持您的原始文字
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="result-card result-card-low">
                        <div style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">未发生II-IV度aGVHD概率</div>
                        <div class="probability-value">{no_agvhd_prob:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col2:
                card_class = "result-card-low" if agvhd_prob < 0.3 else ("result-card-medium" if agvhd_prob < 0.7 else "result-card-high")
                st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">II-IV度aGVHD发生概率</div>
                        <div class="probability-value">{agvhd_prob:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)

            # 风险等级徽章 - 保持您的原始文字
            st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="color: #64748b; font-size: 1rem; margin-bottom: 0.5rem;">综合风险评估</div>
                    <span class="risk-badge badge-{risk_color_class}">{risk_level}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # ====== 诊疗建议 - 卡片式布局（完全保持您的原始文字）======
            st.markdown("---")
            st.markdown("## 诊疗建议")
            
            if agvhd_prob < 0.3:
                st.markdown("""
                    <div class="advice-card advice-low">
                        <h4 style="color: #059669; margin: 0 0 0.75rem 0;">低风险</h4>
                        <p style="color: #334155; line-height: 1.6; margin: 0;">
                            建议继续标准免疫抑制方案，维持常规监测。密切观察患儿皮肤、胃肠道及肝脏功能变化，定期检测血常规及生化指标。
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            elif agvhd_prob < 0.7:
                st.markdown("""
                    <div class="advice-card advice-medium">
                        <h4 style="color: #d97706; margin: 0 0 0.75rem 0;">中等风险</h4>
                        <p style="color: #334155; line-height: 1.6; margin: 0;">
                            建议加强免疫抑制治疗监测，考虑优化钙调磷酸酶抑制剂剂量。密切关注铁蛋白和总胆红素水平变化，加强感染预防措施，必要时进行早期干预。
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="advice-card advice-high">
                        <h4 style="color: #dc2626; margin: 0 0 0.75rem 0;">高风险</h4>
                        <p style="color: #334155; line-height: 1.6; margin: 0;">
                            需立即强化免疫抑制治疗，考虑增加糖皮质激素或抗胸腺细胞球蛋白（ATG）应用。建议住院密切监测，积极预防感染，必要时转入移植专科病房进行强化管理。
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            # ==========================

        except Exception as e:
            st.error(f"预测或结果展示失败: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # 版权或说明 - 保持您的原始文字
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #94a3b8; font-size: 0.875rem; padding: 1rem 0;">
            <p>© 2026 基于机器学习的AML患儿allo-HSCT后并发II-IV度aGVHD预测模型</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()