from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Iterator

from .config import settings
from .llm import ChatMessage, DashScopeLLM
from .rag import LocalRAG
from .schemas import AgentStep, ItineraryItem, TravelPreferences, TripDay, TripPlan
from .tools import ToolRegistry, build_default_registry


SYSTEM_PROMPT = """你是一个旅行规划 ReAct Agent。
你会通过"思考-行动-观察"的循环，自主决定调用哪些工具来收集规划所需的信息，
再根据本地 RAG 知识和工具观察制定行程。
输出要务实，避免编造具体价格、开放时间和坐标；不确定时写入 warnings。
"""


# 单次 ReAct 决策的提示模板。模型每一步只输出一个 Thought + 一个 Action。
REACT_INSTRUCTIONS = """你正在一个 ReAct 循环中，每次只走一步。

可用工具：
{tools}
- finish: 当你已经收集到足够信息，可以开始撰写行程时调用。Action Input 写 {{}} 即可。

严格按如下格式输出（不要输出多余内容、不要使用代码块围栏）：
Thought: <你的推理>
Action: <工具名，必须是上面列出的之一>
Action Input: <一个 JSON 对象，作为工具参数>

用户偏好：
{prefs}

本地 RAG 检索结果（已提供，无需再检索）：
{rag}

到目前为止的推理与观察记录：
{scratchpad}

注意：每日路线地图会在最后自动用 maps_schema_personal_map 生成，你无需调用它。
请输出下一步。如果信息已足够，直接 Action: finish。
"""


PLAN_JSON_INSTRUCTIONS = """请基于以下材料，一次性直接输出结构化旅行方案 JSON。只输出 JSON 本身，不要解释或代码块围栏。

JSON 结构（字段缺失用 null 或空数组，禁止编造坐标/票价）：
{{
  "summary": "一两句话的总体说明",
  "days": [
    {{"date": "YYYY-MM-DD", "theme": "当天主题", "items": [
      {{"time": "09:30", "name": "地点名", "type": "attraction|restaurant|hotel|transport",
        "address": "地址，未知留空", "lat": 30.25 或 null, "lng": 120.15 或 null,
        "duration_min": 120 或 null, "estimated_cost": 80 或 null, "reason": "为什么安排"}}
    ], "notes": "当天提示"}}
  ],
  "hotels": [{{"area": "区域", "reason": "原因", "note": "备注"}}],
  "restaurants": [{{"type": "类型/店名", "reason": "原因", "note": "备注"}}],
  "budget_breakdown": {{"total": 数字, "hotel": 数字, "food": 数字, "transport": 数字, "tickets_and_activities": 数字, "buffer": 数字}},
  "warnings": ["风险提醒"]
}}

要求：
- 行程必须覆盖 {days} 天，日期从 {start_date} 开始按日递增。
- lat/lng 只有当「工具观察」里明确出现该地点坐标时才填，否则必须为 null。
- 不要编造票价和开放时间；不确定的写进 warnings。

用户偏好：
{prefs}

本地 RAG 检索结果：
{rag}

工具观察（坐标/地址的唯一可信来源）：
{observations}
"""


# 从自然语言需求里抽取结构化旅行偏好。
PREF_PARSE_INSTRUCTIONS = """今天是 {today}。请从用户的自然语言旅行需求中抽取结构化偏好，只输出 JSON，不要解释或代码块围栏。

字段（信息缺失填 null）：
{{
  "destination": "目的地城市",
  "origin": "出发地，未提及填 null",
  "start_date": "YYYY-MM-DD，把「下周」「7月初」等相对时间按今天解析",
  "end_date": "YYYY-MM-DD 或 null",
  "days": 整数天数 或 null,
  "people": 整数人数 或 null,
  "budget": 预算数字 或 null,
  "budget_type": "total"（总预算）或 "per_person"（人均）,
  "travel_style": "旅行风格 或 null",
  "transport_preference": "walking|transit|driving|bicycling|mixed",
  "hotel_preference": "住宿偏好 或 null",
  "food_preference": "餐饮偏好 或 null",
  "constraints": "特殊约束，如带老人/忌口/亲子 等，或 null"
}}

用户需求：
{text}
"""


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    return int(f) if f is not None else None


class TravelReActAgent:
    def __init__(
        self,
        llm: DashScopeLLM | None = None,
        rag: LocalRAG | None = None,
        tools: ToolRegistry | None = None,
        max_react_steps: int = 6,
    ):
        self.llm = llm or DashScopeLLM()
        self.rag = rag or LocalRAG()
        self.tools = tools or build_default_registry()
        self.max_react_steps = max_react_steps

    # ---- public API -------------------------------------------------------

    def parse_preferences(self, text: str, today: str | None = None) -> TravelPreferences:
        """把用户自然语言需求解析成结构化 TravelPreferences；解析失败时退回安全默认值。"""
        today = today or datetime.today().date().isoformat()
        raw = self.llm.chat(
            [
                ChatMessage(role="system", content="你是旅行需求解析器，只输出 JSON。"),
                ChatMessage(
                    role="user", content=PREF_PARSE_INSTRUCTIONS.format(today=today, text=text)
                ),
            ]
        )
        data = self._parse_json_object(raw) or {}

        def val(key: str, default: Any) -> Any:
            v = data.get(key)
            return default if v in (None, "") else v

        try:
            start_d = datetime.fromisoformat(str(val("start_date", today))).date()
        except ValueError:
            start_d = datetime.fromisoformat(today).date()

        end_raw = data.get("end_date")
        days = data.get("days")
        end_d = None
        if end_raw:
            try:
                end_d = datetime.fromisoformat(str(end_raw)).date()
            except ValueError:
                end_d = None
        if end_d is not None:
            days = max((end_d - start_d).days + 1, 1)
        else:
            days = int(days) if isinstance(days, (int, float)) and days else 1
            end_d = start_d + timedelta(days=days - 1)

        transport = val("transport_preference", "mixed")
        if transport not in {"walking", "transit", "driving", "bicycling", "mixed"}:
            transport = "mixed"
        budget_type = val("budget_type", "total")
        if budget_type not in {"total", "per_person"}:
            budget_type = "total"

        return TravelPreferences(
            destination=str(val("destination", "")),
            origin=str(val("origin", "")),
            start_date=start_d.isoformat(),
            end_date=end_d.isoformat(),
            days=int(days),
            people=max(int(data.get("people") or 1), 1),
            budget=max(float(data.get("budget") or 0), 0.0),
            budget_type=budget_type,
            travel_style=str(val("travel_style", "舒适、少绕路、经典景点优先")),
            transport_preference=transport,
            hotel_preference=str(val("hotel_preference", "交通便利，性价比优先")),
            food_preference=str(val("food_preference", "当地特色，评分较高")),
            constraints=str(val("constraints", text)),
            raw_query=text,
        )

    def plan(self, prefs: TravelPreferences) -> tuple[TripPlan, list[AgentStep]]:
        steps, rag_hits, scratchpad = self._gather(prefs)
        llm_text = self.llm.chat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=self._plan_prompt(prefs, rag_hits, scratchpad)),
            ]
        )
        return self._finalize(prefs, llm_text, steps, rag_hits)

    def stream_plan_text(
        self, prefs: TravelPreferences
    ) -> tuple[Iterator[str], list[AgentStep], list[dict]]:
        steps, rag_hits, scratchpad = self._gather(prefs)
        chunks = self.llm.stream_chat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=self._plan_prompt(prefs, rag_hits, scratchpad)),
            ]
        )
        return chunks, steps, rag_hits

    def finalize_streamed_plan(
        self, prefs: TravelPreferences, llm_text: str, steps: list[AgentStep], rag_hits: list[dict]
    ) -> tuple[TripPlan, list[AgentStep]]:
        return self._finalize(prefs, llm_text, steps, rag_hits)

    def _finalize(
        self, prefs: TravelPreferences, llm_text: str, steps: list[AgentStep], rag_hits: list[dict]
    ) -> tuple[TripPlan, list[AgentStep]]:
        """单次结构化生成的收尾：记录生成步骤、解析 JSON、补高德地图。"""
        steps.append(self._generation_step(llm_text))
        plan = self.build_plan(prefs, llm_text, rag_hits)
        map_step = self._attach_personal_map(plan, prefs, steps)
        if map_step is not None:
            steps.append(map_step)
        return plan, steps

    def build_plan(self, prefs: TravelPreferences, llm_text: str, rag_hits: list[dict]) -> TripPlan:
        """直接解析模型这一次输出的 JSON；解析失败（如 mock 模式）回退占位模板。"""
        structured = self._plan_from_json(prefs, llm_text, rag_hits)
        if structured is not None:
            return structured
        return self._fallback_structured_plan(prefs, llm_text, rag_hits)

    # ---- ReAct loop -------------------------------------------------------

    def _gather(self, prefs: TravelPreferences) -> tuple[list[AgentStep], list[dict], str]:
        """RAG 检索 + 由 LLM 驱动的 ReAct 工具循环。"""
        steps: list[AgentStep] = []

        query = self._build_query(prefs)
        rag_hits = self.rag.search(query, k=5)
        steps.append(
            AgentStep(
                thought="检索本地 Chroma 知识库，获取目的地背景、区域和旅行注意事项。",
                action="local_chroma_search",
                action_input={"query": query, "k": 5},
                observation=json.dumps(rag_hits, ensure_ascii=False)[:2000] or "未检索到本地知识。",
            )
        )

        react_steps, scratchpad = self._run_react_loop(prefs, rag_hits)
        steps.extend(react_steps)
        return steps, rag_hits, scratchpad

    def _run_react_loop(
        self, prefs: TravelPreferences, rag_hits: list[dict]
    ) -> tuple[list[AgentStep], str]:
        steps: list[AgentStep] = []
        scratchpad = ""
        seen: set[str] = set()
        valid_tools = set(self.tools.names) | {"finish"}

        for _ in range(self.max_react_steps):
            prompt = REACT_INSTRUCTIONS.format(
                tools=self.tools.descriptions(),
                prefs=prefs.model_dump_json(indent=2),
                rag=json.dumps(rag_hits, ensure_ascii=False, indent=2)[:2000],
                scratchpad=scratchpad or "（尚无记录）",
            )
            text = self.llm.chat(
                [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ]
            )
            thought, action, action_input = self._parse_react(text)

            # 模型未给出可解析的 Action（含 mock 模式）：终止循环，进入最终合成。
            if not action:
                steps.append(
                    AgentStep(
                        thought=thought or "模型未返回结构化 Action，结束工具收集，直接合成方案。",
                        action="finish",
                        action_input={},
                        observation="（无可解析的工具调用，提前结束循环）",
                    )
                )
                break

            if action == "finish":
                steps.append(
                    AgentStep(
                        thought=thought or "信息已足够，结束工具收集。",
                        action="finish",
                        action_input={},
                        observation="（结束 ReAct 循环，开始撰写最终方案）",
                    )
                )
                break

            if action not in valid_tools:
                observation_text = f"未知工具：{action}。请从可用工具中选择，或 Action: finish。"
            else:
                signature = f"{action}:{json.dumps(action_input, ensure_ascii=False, sort_keys=True)}"
                if signature in seen:
                    observation_text = "该工具已用相同参数调用过，请换参数或 Action: finish。"
                else:
                    seen.add(signature)
                    result = self.tools.run(action, action_input)
                    observation_text = json.dumps(result, ensure_ascii=False)[:1500]

            steps.append(
                AgentStep(
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation_text,
                )
            )
            scratchpad += (
                f"Thought: {thought}\n"
                f"Action: {action}\n"
                f"Action Input: {json.dumps(action_input, ensure_ascii=False)}\n"
                f"Observation: {observation_text}\n\n"
            )

        return steps, scratchpad

    @staticmethod
    def _parse_react(text: str) -> tuple[str, str, dict[str, Any]]:
        """从模型输出中解析 Thought / Action / Action Input。"""
        thought = ""
        action = ""
        action_input: dict[str, Any] = {}

        thought_match = re.search(r"Thought\s*[:：]\s*(.+?)(?=\n\s*Action\s*[:：]|\Z)", text, re.S)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action\s*[:：]\s*([A-Za-z_][\w]*)", text)
        if action_match:
            action = action_match.group(1).strip()

        input_match = re.search(r"Action\s*Input\s*[:：]\s*(.+)", text, re.S)
        if input_match:
            raw = input_match.group(1).strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            brace = re.search(r"\{.*\}", raw, re.S)
            if brace:
                try:
                    action_input = json.loads(brace.group(0))
                except json.JSONDecodeError:
                    action_input = {}

        return thought, action, action_input

    # ---- prompts / plan ---------------------------------------------------

    def _plan_prompt(self, prefs: TravelPreferences, rag_hits: list[dict], scratchpad: str) -> str:
        return PLAN_JSON_INSTRUCTIONS.format(
            days=prefs.days,
            start_date=prefs.start_date,
            prefs=prefs.model_dump_json(indent=2),
            rag=json.dumps(rag_hits, ensure_ascii=False, indent=2)[:3000],
            observations=scratchpad[:4000] or "（本轮未调用工具）",
        )

    def _plan_from_json(
        self, prefs: TravelPreferences, llm_text: str, rag_hits: list[dict]
    ) -> TripPlan | None:
        """直接解析模型这一次生成的 JSON 为 TripPlan。解析失败返回 None。"""
        data = self._parse_json_object(llm_text)
        if not data:
            return None

        days = self._coerce_days(data.get("days"))
        if not days:
            return None

        sources = sorted(
            {src for hit in rag_hits if (src := hit.get("metadata", {}).get("source"))}
        )
        return TripPlan(
            destination=prefs.destination,
            summary=str(data.get("summary", "") or ""),
            days=days,
            hotels=self._coerce_dicts(data.get("hotels")),
            restaurants=self._coerce_dicts(data.get("restaurants")),
            budget_breakdown=self._coerce_budget(data.get("budget_breakdown"), prefs),
            warnings=[str(w) for w in data.get("warnings", []) if w],
            sources=sources,
        )

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _coerce_days(raw: Any) -> list[TripDay]:
        if not isinstance(raw, list):
            return []
        days: list[TripDay] = []
        for day in raw:
            if not isinstance(day, dict):
                continue
            items = []
            for item in day.get("items", []) or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                try:
                    items.append(
                        ItineraryItem(
                            time=str(item.get("time", "")),
                            name=str(item["name"]),
                            type=str(item.get("type", "attraction")),
                            address=str(item.get("address", "") or ""),
                            lat=_as_float(item.get("lat")),
                            lng=_as_float(item.get("lng")),
                            duration_min=_as_int(item.get("duration_min")),
                            estimated_cost=_as_float(item.get("estimated_cost")),
                            reason=str(item.get("reason", "") or ""),
                        )
                    )
                except Exception:
                    continue
            if not items:
                continue
            days.append(
                TripDay(
                    date=str(day.get("date", "")),
                    theme=str(day.get("theme", "") or ""),
                    items=items,
                    notes=str(day.get("notes", "") or ""),
                )
            )
        return days

    @staticmethod
    def _coerce_dicts(raw: Any) -> list[dict]:
        return [d for d in raw if isinstance(d, dict)] if isinstance(raw, list) else []

    @staticmethod
    def _coerce_budget(raw: Any, prefs: TravelPreferences) -> dict:
        if isinstance(raw, dict) and raw:
            return raw
        total = prefs.budget * prefs.people if prefs.budget_type == "per_person" else prefs.budget
        if not total:
            return {}
        return {
            "total": total,
            "hotel": round(total * 0.4, 2),
            "food": round(total * 0.25, 2),
            "transport": round(total * 0.15, 2),
            "tickets_and_activities": round(total * 0.15, 2),
            "buffer": round(total * 0.05, 2),
        }

    # ---- 高德每日路线地图 -------------------------------------------------

    def _attach_personal_map(
        self, plan: TripPlan, prefs: TravelPreferences, steps: list[AgentStep]
    ) -> AgentStep | None:
        """把每日行程点位按顺序填入高德 maps_schema_personal_map，生成可打开的路线地图 URI。

        没有任何带坐标的点位时返回 None，不调用工具，并在 plan.map_note 里说明原因，
        以便 UI 区分「未接入(mock)」与「已接入但高德调用失败/配额超限」。
        """
        line_list = self._build_line_list(plan)
        if not line_list:
            plan.map_note = self._diagnose_missing_map(steps)
            return None

        params = {"orgName": f"{prefs.destination}行程规划", "lineList": line_list}
        result = self.tools.run("maps_schema_personal_map", params)
        uri = self._extract_uri(result)
        plan.map_uri = uri

        return AgentStep(
            thought="把每日行程点位按行程顺序填入高德 maps_schema_personal_map，生成可在高德打开的每日路线地图。",
            action="maps_schema_personal_map",
            action_input=params,
            observation=(uri or json.dumps(result, ensure_ascii=False))[:1500],
        )

    @staticmethod
    def _diagnose_missing_map(steps: list[AgentStep]) -> str:
        """没拿到坐标时给出可读原因：mock 未接入 / 高德报错 / 仅是没查到坐标。"""
        if settings.mcp_mode.lower() != "real":
            return (
                "当前 MCP_MODE=mock，高德工具返回的是占位数据，没有真实坐标。"
                "把 MCP_MODE 设为 real 并配置 MCP_AMAP_ENDPOINT 后即可生成每日路线地图。"
            )

        amap_obs = " ".join(
            s.observation for s in steps if s.action.startswith("maps_") or "amap" in s.observation.lower()
        )
        if "USER_DAILY_QUERY_OVER_LIMIT" in amap_obs:
            return (
                "高德 MCP 返回 USER_DAILY_QUERY_OVER_LIMIT：该高德 key 当日调用配额已用尽，"
                "因此拿不到坐标、无法生成路线地图。请等次日配额重置，或在高德开放平台为该 key 提升配额。"
            )
        if '"isError": true' in amap_obs or "isError': True" in amap_obs or "调用失败" in amap_obs:
            return (
                "高德 MCP 已接入但工具调用返回错误，未能获得坐标。"
                "请到「调试」标签查看各 maps_* 工具的 Observation 了解具体原因。"
            )
        return (
            "本次未从高德工具获得任何坐标（可能是模型未调用定位类工具，或目的地未匹配到 POI）。"
            "可在「调试」标签查看 ReAct 调用记录。"
        )

    @staticmethod
    def _build_line_list(plan: TripPlan) -> list[dict]:
        """每天一条 line，pointInfoList 为当天按顺序、带坐标的点位。"""
        lines: list[dict] = []
        for idx, day in enumerate(plan.days, start=1):
            points = [
                {"name": item.name, "lon": str(item.lng), "lat": str(item.lat)}
                for item in day.items
                if item.lat is not None and item.lng is not None
            ]
            if not points:
                continue
            lines.append({"title": day.theme or f"第 {idx} 天", "pointInfoList": points})
        return lines

    @staticmethod
    def _extract_uri(value: Any) -> str:
        """从（可能多层嵌套的）工具返回里抠出第一个地图链接。"""
        found: list[str] = []

        def walk(v: Any) -> None:
            if found:
                return
            if isinstance(v, str):
                s = v.strip()
                if s.startswith(("http://", "https://", "amapuri", "androidamap", "iosamap")):
                    found.append(s)
            elif isinstance(v, dict):
                for vv in v.values():
                    walk(vv)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    walk(vv)

        walk(value)
        return found[0] if found else ""

    def _generation_step(self, llm_text: str) -> AgentStep:
        return AgentStep(
            thought="综合偏好、RAG 和工具观察，一次性生成结构化行程方案。",
            action="dashscope_chat_completion",
            action_input={"model": self.llm.model, "stream": True},
            observation=llm_text[:3000],
        )

    def _build_query(self, prefs: TravelPreferences) -> str:
        """构造 RAG 检索 query。

        只保留对「在攻略库里检索」有语义价值的字段，剔除日期、预算、天数等数字噪声——
        这些 token 在攻略文档里几乎不出现，只会稀释向量信号。自由描述模式下并入用户
        原文（raw_query），因为口语化的原始需求往往比解析后的字段更丰富。
        """
        semantic_fields = [
            prefs.travel_style,
            prefs.hotel_preference,
            prefs.food_preference,
            prefs.constraints,
            prefs.raw_query,
        ]
        seen: set[str] = set()
        extras: list[str] = []
        for field in semantic_fields:
            field = (field or "").strip()
            if field and field not in seen:
                seen.add(field)
                extras.append(field)
        return f"{prefs.destination} 旅游攻略 景点 美食 住宿 行程 {' '.join(extras)}".strip()

    def _fallback_structured_plan(self, prefs: TravelPreferences, llm_text: str, rag_hits: list[dict]) -> TripPlan:
        try:
            start = datetime.fromisoformat(prefs.start_date)
        except ValueError:
            start = datetime.today()

        days: list[TripDay] = []
        for idx in range(prefs.days):
            date = (start + timedelta(days=idx)).date().isoformat()
            days.append(
                TripDay(
                    date=date,
                    theme=f"{prefs.destination} 第 {idx + 1} 天探索",
                    items=[
                        ItineraryItem(
                            time="09:30",
                            name=f"{prefs.destination} 核心景点区",
                            type="attraction",
                            duration_min=150,
                            reason="先用结构化占位展示；接入高德 MCP 解析后可替换为真实 POI。",
                        ),
                        ItineraryItem(
                            time="12:30",
                            name="当地特色午餐",
                            type="restaurant",
                            duration_min=75,
                            reason=prefs.food_preference,
                        ),
                        ItineraryItem(
                            time="14:30",
                            name=f"{prefs.destination} 城市漫游/博物馆/街区",
                            type="attraction",
                            duration_min=180,
                            reason=prefs.travel_style,
                        ),
                    ],
                    notes="真实路线、坐标和交通时间将在高德 MCP 接入后自动补齐。",
                )
            )

        total_budget = prefs.budget * prefs.people if prefs.budget_type == "per_person" else prefs.budget
        sources = [hit.get("metadata", {}).get("source", "local_chroma") for hit in rag_hits]
        return TripPlan(
            destination=prefs.destination,
            summary=llm_text,
            days=days,
            hotels=[
                {
                    "area": "交通枢纽或核心商圈附近",
                    "reason": prefs.hotel_preference,
                    "note": "后续可用 maps_text_search + 酒店 API 替换为真实候选。",
                }
            ],
            restaurants=[
                {
                    "type": "当地特色餐厅",
                    "reason": prefs.food_preference,
                    "note": "建议接入合法餐饮数据源后排序。",
                }
            ],
            budget_breakdown={
                "total": total_budget,
                "hotel": round(total_budget * 0.4, 2) if total_budget else None,
                "food": round(total_budget * 0.25, 2) if total_budget else None,
                "transport": round(total_budget * 0.15, 2) if total_budget else None,
                "tickets_and_activities": round(total_budget * 0.15, 2) if total_budget else None,
                "buffer": round(total_budget * 0.05, 2) if total_budget else None,
            },
            warnings=[
                "开放时间、票价、酒店库存和交通管制需要接入真实工具后再校验。",
            ],
            sources=sorted(set(s for s in sources if s)),
        )
