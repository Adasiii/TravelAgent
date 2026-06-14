from __future__ import annotations

import json

import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover
    folium = None
    st_folium = None

from travel_agent.agent import TravelReActAgent
from travel_agent.rag import LocalRAG
from travel_agent.schemas import AgentStep, TravelPreferences, TripPlan


st.set_page_config(page_title="TravelAgent", page_icon="🧭", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1080px; }
h1 { font-weight: 700; letter-spacing: -0.5px; }
/* 行程条目卡片 */
.itin-item {
  border-left: 3px solid #2563eb;
  padding: 0.35rem 0.8rem;
  margin: 0.45rem 0;
  background: rgba(37, 99, 235, 0.04);
  border-radius: 0 8px 8px 0;
}
.itin-time { font-weight: 700; color: #2563eb; }
.itin-meta { color: #6b7280; font-size: 0.82rem; }
.itin-reason { color: #374151; font-size: 0.9rem; margin-top: 0.2rem; }
div[data-testid="stExpander"] { border-radius: 10px; }
</style>
"""


@st.cache_resource
def get_agent() -> TravelReActAgent:
    return TravelReActAgent()


@st.cache_resource
def get_rag() -> LocalRAG:
    return LocalRAG()


def ensure_state() -> None:
    st.session_state.setdefault("plan", None)
    st.session_state.setdefault("steps", [])
    st.session_state.setdefault("last_prefs", None)


_DAY_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]


def render_map(plan_json: dict) -> None:
    map_uri = plan_json.get("map_uri", "")

    days_points = []
    for day in plan_json.get("days", []):
        pts = [
            (item["lat"], item["lng"], item.get("name", ""))
            for item in day.get("items", [])
            if item.get("lat") is not None and item.get("lng") is not None
        ]
        if pts:
            days_points.append((day.get("date", ""), day.get("theme", ""), pts))

    if map_uri:
        st.link_button("🗺️ 在高德地图打开每日路线行程", map_uri, use_container_width=True)
        st.caption("由高德 maps_schema_personal_map 按每日行程顺序串联点位生成，点击在高德地图查看规划路线。")
    elif not days_points:
        note = plan_json.get("map_note", "")
        if note:
            st.warning(note)
        else:
            st.info("接入高德 MCP（MCP_MODE=real 并配置 MCP_AMAP_ENDPOINT）后，这里会生成按每日路线串联的高德地图链接。")

    if folium is None or st_folium is None:
        if not map_uri:
            st.info("未安装 folium / streamlit-folium，暂不显示内联地图预览。")
        return
    if not days_points:
        return

    first = days_points[0][2][0]
    m = folium.Map(location=[first[0], first[1]], zoom_start=12)
    for idx, (date, theme, pts) in enumerate(days_points):
        color = _DAY_COLORS[idx % len(_DAY_COLORS)]
        label = date or theme or f"第 {idx + 1} 天"
        for order, (lat, lng, name) in enumerate(pts, start=1):
            folium.Marker(
                [lat, lng],
                popup=f"{label} · {order}. {name}",
                icon=folium.Icon(color=color),
            ).add_to(m)
        if len(pts) >= 2:
            folium.PolyLine(
                [(lat, lng) for lat, lng, _ in pts], color=color, weight=4, opacity=0.7, tooltip=label
            ).add_to(m)
    st_folium(m, height=520, use_container_width=True)


def render_result(plan: TripPlan, steps: list[AgentStep], show_trace: bool, show_json: bool) -> None:
    plan_json = plan.model_dump()

    if plan.summary:
        st.markdown(plan.summary)

    tab_plan, tab_map, tab_budget, tab_debug = st.tabs(["📅 每日行程", "🗺️ 地图", "💰 预算 / 建议", "🔍 调试"])

    with tab_plan:
        for day in plan.days:
            with st.expander(f"{day.date} · {day.theme}", expanded=True):
                for item in day.items:
                    cost = f" · ¥{item.estimated_cost:g}" if item.estimated_cost else ""
                    dur = f"{item.duration_min} 分钟" if item.duration_min else "-"
                    addr = f" · {item.address}" if item.address else ""
                    st.markdown(
                        f'<div class="itin-item">'
                        f'<span class="itin-time">{item.time}</span>　{item.name}'
                        f'<div class="itin-meta">{item.type} · {dur}{cost}{addr}</div>'
                        f'<div class="itin-reason">{item.reason}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if day.notes:
                    st.info(day.notes)

    with tab_map:
        render_map(plan_json)

    with tab_budget:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**预算拆分**")
            st.json(plan.budget_breakdown)
        with col_b:
            st.markdown("**住宿建议**")
            st.json(plan.hotels)
            st.markdown("**餐饮建议**")
            st.json(plan.restaurants)
        for warning in plan.warnings:
            st.warning(warning)
        if plan.sources:
            st.caption("知识来源：" + "、".join(plan.sources))

    with tab_debug:
        if show_trace:
            for i, step in enumerate(steps, start=1):
                with st.expander(f"Step {i}: {step.action or 'thought'}"):
                    st.markdown(f"**Thought**: {step.thought}")
                    st.code(json.dumps(step.action_input, ensure_ascii=False, indent=2), language="json")
                    st.text(step.observation)
        if show_json:
            st.download_button(
                "下载 itinerary.json",
                data=json.dumps(plan_json, ensure_ascii=False, indent=2),
                file_name="itinerary.json",
                mime="application/json",
            )
            st.json(plan_json)


def generate_plan(prefs: TravelPreferences) -> None:
    """运行 Agent 并把结果写入 session_state（两种输入模式共用）。"""
    st.session_state["plan"] = None
    st.session_state["steps"] = []
    st.session_state["last_prefs"] = prefs.model_dump()

    agent = get_agent()
    streamed_text = ""

    with st.status("Agent 正在检索知识库并调用工具...", expanded=True) as status:
        chunks, steps, rag_hits = agent.stream_plan_text(prefs)
        for step in steps:
            st.write(f"`{step.action}` · {step.thought}")
        status.update(label="模型正在流式生成旅行方案...", state="running", expanded=False)

    stream_box = st.empty()
    for chunk in chunks:
        streamed_text += chunk
        stream_box.markdown(streamed_text + "▌")
    stream_box.markdown(streamed_text)

    plan, steps = agent.finalize_streamed_plan(prefs, streamed_text, steps, rag_hits)
    st.session_state["plan"] = plan
    st.session_state["steps"] = steps
    st.success("旅行方案生成完成")


def structured_form() -> TravelPreferences | None:
    """原有的参数选择模式。"""
    with st.form("trip_form_structured", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            destination = st.text_input("目的地", value="杭州")
            origin = st.text_input("出发地", value="")
            start_date = st.date_input("开始日期")
        with col2:
            end_date = st.date_input("结束日期")
            people = st.number_input("人数", min_value=1, value=2, step=1)
            budget = st.number_input("预算", min_value=0.0, value=5000.0, step=500.0)
        with col3:
            budget_type = st.selectbox(
                "预算类型",
                ["total", "per_person"],
                format_func=lambda x: "总预算" if x == "total" else "人均预算",
            )
            transport_preference = st.selectbox("交通偏好", ["mixed", "walking", "transit", "driving", "bicycling"])
            travel_style = st.text_input("旅行风格", value="舒适、少绕路、经典景点优先")

        hotel_preference = st.text_input("住宿偏好", value="交通便利，性价比优先")
        food_preference = st.text_input("餐饮偏好", value="当地特色，评分较高")
        constraints = st.text_area("特殊约束", value="", placeholder="例如：带老人、小朋友；不吃辣；每天不要超过 2 万步")

        submitted = st.form_submit_button("生成旅行方案", type="primary", use_container_width=True)

    if not submitted:
        return None

    days = max((end_date - start_date).days + 1, 1)
    return TravelPreferences(
        destination=destination,
        origin=origin,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        days=days,
        people=int(people),
        budget=float(budget),
        budget_type=budget_type,
        travel_style=travel_style,
        transport_preference=transport_preference,
        hotel_preference=hotel_preference,
        food_preference=food_preference,
        constraints=constraints,
    )


def freeform_input() -> TravelPreferences | None:
    """自由输入模式：一句话描述需求，由 LLM 解析成结构化偏好。"""
    with st.form("trip_form_free", clear_on_submit=False):
        text = st.text_area(
            "用一段话描述你的旅行需求",
            height=140,
            placeholder="例如：7 月初想和家人去杭州玩两天，预算 5000 元，带老人不想太累，喜欢西湖这种经典景点，住得交通方便点，想吃地道杭帮菜。",
        )
        submitted = st.form_submit_button("解析并生成方案", type="primary", use_container_width=True)

    if not submitted:
        return None
    if not text.strip():
        st.warning("请先描述你的旅行需求。")
        return None

    with st.spinner("正在理解你的需求..."):
        prefs = get_agent().parse_preferences(text)

    with st.expander("已解析的偏好（可切换到「参数选择」微调后重新生成）", expanded=True):
        st.json(prefs.model_dump())
    if not prefs.destination:
        st.warning("没能识别出明确的目的地，建议补充城市名，或改用「参数选择」模式。")

    return prefs


ensure_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🧭 TravelAgent")
st.caption("ReAct 工具调用 · 本地 Chroma RAG · 高德路线地图 · 流式方案生成")

with st.sidebar:
    st.header("⚙️ 知识库")
    st.write("把 .md / .txt 攻略放到 `data/knowledge`，点击入库。")
    if st.button("构建 / 更新本地文档知识库", use_container_width=True):
        rag = get_rag()
        count = rag.ingest_folder()
        if rag.available:
            st.success(f"已写入 / 更新 {count} 个文本块")
        else:
            st.warning(f"Chroma 初始化失败：{rag.error}")
    st.caption("联网内容 RAG 请运行 scripts/build_rag.py 或 scripts/build_rag_bulk.py。")

    st.divider()
    st.header("🔍 调试")
    show_trace = st.toggle("显示 ReAct Trace", value=True)
    show_json = st.toggle("显示 JSON", value=False)
    if st.button("清空当前结果", use_container_width=True):
        st.session_state["plan"] = None
        st.session_state["steps"] = []
        st.session_state["last_prefs"] = None

mode = st.radio(
    "输入方式",
    ["✍️ 自由描述", "🎛️ 参数选择"],
    horizontal=True,
    label_visibility="collapsed",
)

prefs = freeform_input() if mode == "✍️ 自由描述" else structured_form()

if prefs is not None:
    generate_plan(prefs)

st.divider()

if st.session_state["plan"] is None:
    st.info("选择一种输入方式填写需求，点击生成。模型输出会流式显示，结果会保留在页面状态中。")
else:
    render_result(
        st.session_state["plan"],
        st.session_state["steps"],
        show_trace=show_trace,
        show_json=show_json,
    )
