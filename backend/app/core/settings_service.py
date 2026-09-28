"""运行时配置服务。

设计：**环境变量是默认值，数据库是覆盖层**。
  * 界面上的「设置」页写 `app_settings` 表；
  * 业务代码在每次请求开始时调 `effective()` 拿一份合并后的快照；
  * 这样改配置无需重启进程，同时保留 .env 作为「出厂设置」可一键还原。

对外还暴露 `SPECS`（每个可调项的元信息），前端据此自动渲染表单，
避免前后端各写一份配置项清单、改一处忘一处。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import AppSetting


# ----------------------------------------------------------------------
# 动态选项来源
#
# 有些选项的候选值无法在代码里写死 —— 比如「有哪些模型可用」取决于用户
# 实际 pull 了哪些模型。这些列表在**请求时**向 Ollama 查询，
# 好处是用户装了新模型后界面立刻就能选到，不需要改代码。
# ----------------------------------------------------------------------
SOURCE_OLLAMA_CHAT_MODELS = "ollama_chat_models"
SOURCE_OLLAMA_EMBED_MODELS = "ollama_embedding_models"
DYNAMIC_CHOICE_SOURCES: tuple[str, ...] = (
    SOURCE_OLLAMA_CHAT_MODELS,
    SOURCE_OLLAMA_EMBED_MODELS,
)


@dataclass(frozen=True)
class SettingSpec:
    """一个可调配置项的元信息。

    界面完全由这些元信息驱动渲染，所以**层级关系也必须在这里声明**，
    而不是前端写死归类 —— 否则后端加一项、前端忘了归类，界面就长出孤儿字段。
    """

    key: str
    type: type  # bool | int | float | str
    label: str
    group: str = "通用"
    # 二级分组：同一个 group 内再按语义聚类。
    # 空字符串表示「本组主控项」，界面会把它置顶并突出显示。
    section: str = ""
    description: str = ""
    secret: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None
    # 选项的显示名，让下拉框显示「纯本地 · Ollama」而不是裸的 `ollama`。
    # 用元组而不是 dict：dataclass 是 frozen 的，dict 字段不可哈希。
    choice_labels: tuple[tuple[str, str], ...] = ()
    # 动态选项来源（取值为上面的 SOURCE_* 常量）。
    #
    # 与 choices 的关键区别：choices 是代码里写死的枚举，会做强校验；
    # choices_source 是运行时查出来的**建议列表**，界面渲染成
    # 「可自由输入 + 下拉建议」。为什么要自由输入：Ollama 没启动时列表必然是空的，
    # 此时若做强校验，用户连保存设置都做不到 —— 一个连不上 Ollama 的界面
    # 还锁死了「改模型名」这个唯一的自救手段，就太荒唐了。
    choices_source: str | None = None
    restart_hint: str = ""
    # 依赖关系：(开关键, 让本项生效的取值集合)。
    # 依赖不满足时界面置灰并提示原因，避免用户改了「云端模型名」
    # 却发现实际走的是本地通道 —— 这种「改了没反应」是最难排查的一类困惑。
    depends_on: tuple[str, tuple[str, ...]] | None = None
    # 生效范围提示，例如「仅对新建知识库生效」
    scope: str = ""


@dataclass(frozen=True)
class GroupSpec:
    """一级分组：给出展示顺序与一句话定位。

    没有 summary 的分组标题对用户是没有信息量的 ——「检索」两个字
    并不能告诉他这一组改动会影响什么。
    """

    name: str
    summary: str
    order: int


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        name="生成通道",
        summary="决定用哪个模型来组织回答",
        order=1,
    ),
    GroupSpec(
        name="生成参数",
        summary="控制回答的风格、长度与可靠性",
        order=2,
    ),
    GroupSpec(
        name="检索",
        summary="决定从知识库里捞出哪些片段送给模型",
        order=3,
    ),
    GroupSpec(
        name="文档入库",
        summary="决定文档如何被向量化与切分（影响之后新建的知识库）",
        order=4,
    ),
)

GROUP_BY_NAME: dict[str, GroupSpec] = {g.name: g for g in GROUPS}
# 未在 GROUPS 里登记的分组排到最后，保证新增分组不会凭空消失
UNKNOWN_GROUP_ORDER = 999


SPECS: tuple[SettingSpec, ...] = (
    # ================= 生成通道 =================
    SettingSpec(
        key="llm_provider",
        type=str,
        label="生成通道",
        group="生成通道",
        # 主控项：section 留空，界面会置顶并渲染成醒目的选择卡片
        description="ollama = 纯本地推理；openai = 任意 OpenAI 兼容接口（云端或本地 vLLM / LM Studio）",
        choices=("ollama", "openai"),
        choice_labels=(
            ("ollama", "纯本地 · Ollama"),
            ("openai", "云端 · OpenAI 兼容接口"),
        ),
    ),
    SettingSpec(
        key="ollama_chat_model",
        type=str,
        label="本地对话模型",
        group="生成通道",
        section="本地 Ollama",
        description="Ollama 中的模型名，例如 qwen2.5:7b-instruct",
        choices_source=SOURCE_OLLAMA_CHAT_MODELS,
        depends_on=("llm_provider", ("ollama",)),
    ),
    SettingSpec(
        key="openai_base_url",
        type=str,
        label="接口地址 Base URL",
        group="生成通道",
        section="云端 / OpenAI 兼容接口",
        description="需包含版本路径，例如 https://api.deepseek.com/v1",
        depends_on=("llm_provider", ("openai",)),
    ),
    SettingSpec(
        key="openai_api_key",
        type=str,
        label="API Key",
        group="生成通道",
        section="云端 / OpenAI 兼容接口",
        description="仅保存在本机 SQLite，不会外传",
        secret=True,
        depends_on=("llm_provider", ("openai",)),
    ),
    SettingSpec(
        key="openai_model",
        type=str,
        label="模型名",
        group="生成通道",
        section="云端 / OpenAI 兼容接口",
        description="例如 deepseek-chat、qwen-plus",
        depends_on=("llm_provider", ("openai",)),
    ),
    # ================= 生成参数 =================
    SettingSpec(
        key="temperature",
        type=float,
        label="温度",
        group="生成参数",
        section="采样控制",
        description="越低越保守。RAG 场景建议 0.1~0.3",
        minimum=0.0,
        maximum=2.0,
    ),
    SettingSpec(
        key="max_tokens",
        type=int,
        label="最大生成长度",
        group="生成参数",
        section="采样控制",
        description="单次回答的 token 上限，超出会被截断",
        minimum=64,
        maximum=8192,
    ),
    SettingSpec(
        key="num_ctx",
        type=int,
        label="上下文窗口",
        group="生成参数",
        section="上下文与显存",
        description="Ollama 的 num_ctx，是显存占用的大头",
        minimum=2048,
        maximum=32768,
        depends_on=("llm_provider", ("ollama",)),
    ),
    SettingSpec(
        key="strict_grounding",
        type=bool,
        label="严格依据资料作答",
        group="生成参数",
        section="回答准则",
        description="开启后模型被要求：资料不足时明确说明，不得编造",
    ),
    # ================= 检索 =================
    SettingSpec(
        key="enable_hybrid",
        type=bool,
        label="启用混合检索",
        group="检索",
        description="开启 = BM25 稀疏检索 + 向量稠密检索，用 RRF 融合；关闭 = 纯向量检索",
    ),
    SettingSpec(
        key="dense_top_k",
        type=int,
        label="向量召回数",
        group="检索",
        section="双路召回",
        description="稠密检索这一路各自召回多少条候选",
        minimum=1,
        maximum=100,
        depends_on=("enable_hybrid", ("true",)),
    ),
    SettingSpec(
        key="sparse_top_k",
        type=int,
        label="BM25 召回数",
        group="检索",
        section="双路召回",
        description="稀疏检索这一路各自召回多少条候选",
        minimum=1,
        maximum=100,
        depends_on=("enable_hybrid", ("true",)),
    ),
    SettingSpec(
        key="rrf_k",
        type=int,
        label="RRF 平滑常数 k",
        group="检索",
        section="融合与筛选",
        description="倒数排名融合公式中的 k，论文常用 60。越小越突出头部结果",
        minimum=1,
        maximum=200,
        depends_on=("enable_hybrid", ("true",)),
    ),
    SettingSpec(
        key="final_top_k",
        type=int,
        label="最终送入模型片段数",
        group="检索",
        section="融合与筛选",
        description="越大上下文越全，但更慢、更贵，也更容易引入噪声",
        minimum=1,
        maximum=30,
    ),
    SettingSpec(
        key="enable_query_rewrite",
        type=bool,
        label="多轮查询改写",
        group="检索",
        section="多轮对话",
        description="把「那它呢」这类依赖上下文的追问改写成完整独立问题",
    ),
    # ================= 文档入库 =================
    SettingSpec(
        key="ollama_embed_model",
        type=str,
        label="向量模型",
        group="文档入库",
        section="向量模型",
        description=(
            "把切片编码成向量的模型。它决定了整个知识库的「语义坐标系」，"
            "不同模型产出的向量互不兼容，因此不匹配时报错而不是静默出错"
        ),
        choices_source=SOURCE_OLLAMA_EMBED_MODELS,
        scope="仅对新建知识库生效",
    ),
    SettingSpec(
        key="chunk_size",
        type=int,
        label="切片长度（字符）",
        group="文档入库",
        section="切片尺寸",
        description="递归切分的目标长度，优先在段落/句子边界断开",
        minimum=100,
        maximum=4000,
        scope="仅对新建知识库生效",
    ),
    SettingSpec(
        key="chunk_overlap",
        type=int,
        label="切片重叠（字符）",
        group="文档入库",
        section="切片尺寸",
        description="缓解跨片语义被切断的问题，必须小于切片长度",
        minimum=0,
        maximum=1000,
        scope="仅对新建知识库生效",
    ),
)

SPEC_BY_KEY: dict[str, SettingSpec] = {s.key: s for s in SPECS}
SECRET_KEYS = frozenset(s.key for s in SPECS if s.secret)

# 非覆盖项：只读展示用的环境信息。
# 注意 ollama_embed_model 已移出这里 —— 它现在是可调项（见 SPECS 的「文档入库」组）。
READONLY_KEYS = (
    "ollama_base_url",
    "embedding_provider",
    "embed_batch_size",
    "llm_timeout",
    "app_version",
)


class SettingsValidationError(ValueError):
    """配置值不合法。"""


def _normalize_model_name(name: str) -> str:
    """补全省略的 tag。

    Ollama 把 `bge-m3` 和 `bge-m3:latest` 视为同一个模型，
    但 `/api/tags` 只返回带 tag 的完整名，而 .env 里通常写的是省略形式。
    不做归一化的话，界面上会同时出现 `bge-m3` 与 `bge-m3:latest` 两个选项，
    看起来像两个不同的模型，用户会以为配错了。
    """
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def _coerce(spec: SettingSpec, value: Any) -> Any:
    """把前端可能传进来的字符串/数字统一成目标类型，并做范围校验。"""
    if spec.type is bool:
        if isinstance(value, bool):
            out = value
        elif isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                out = True
            elif low in ("false", "0", "no", "off"):
                out = False
            else:
                raise SettingsValidationError(f"{spec.label}：无法把 `{value}` 解释为布尔值")
        elif isinstance(value, (int, float)):
            out = bool(value)
        else:
            raise SettingsValidationError(f"{spec.label}：类型应为布尔值")

    elif spec.type is int:
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise SettingsValidationError(f"{spec.label}：`{value}` 不是合法整数") from exc
        # 注意 bool 是 int 子类，这里已被上面分支拦掉，不会混淆

    elif spec.type is float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise SettingsValidationError(f"{spec.label}：`{value}` 不是合法数字") from exc

    elif spec.type is str:
        out = "" if value is None else str(value).strip()
        # 带动态选项来源的字符串（模型名）不能为空：
        # 空串会一路传到建库流程，虽然下游有兜底默认值，但让用户存一个空模型名
        # 本身就是个不该被接受的输入。
        if spec.choices_source and not out:
            raise SettingsValidationError(f"{spec.label}：不能为空")

    else:  # pragma: no cover - SPECS 里不会出现
        out = value

    if spec.choices and out not in spec.choices:
        raise SettingsValidationError(
            f"{spec.label}：只允许 {'、'.join(spec.choices)}，收到 `{out}`"
        )
    if spec.minimum is not None and isinstance(out, (int, float)) and out < spec.minimum:
        raise SettingsValidationError(f"{spec.label}：不能小于 {spec.minimum}")
    if spec.maximum is not None and isinstance(out, (int, float)) and out > spec.maximum:
        raise SettingsValidationError(f"{spec.label}：不能大于 {spec.maximum}")
    return out


class SettingsService:
    """某一次请求范围内的有效配置视图。"""

    def __init__(self, db: Session, base: Settings | None = None) -> None:
        self.db = db
        self.base = base or get_settings()
        self._overrides: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    @property
    def overrides(self) -> dict[str, Any]:
        if self._overrides is None:
            self._overrides = self._load_overrides()
        return self._overrides

    def _load_overrides(self) -> dict[str, Any]:
        rows = self.db.execute(select(AppSetting)).scalars().all()
        out: dict[str, Any] = {}
        for row in rows:
            spec = SPEC_BY_KEY.get(row.key)
            if spec is None:
                continue  # 已废弃的键，忽略
            raw = row.value.get("v") if isinstance(row.value, dict) else row.value
            try:
                out[row.key] = _coerce(spec, raw)
            except SettingsValidationError:
                continue  # 脏数据不应该让整个系统起不来
        return out

    def effective(self) -> dict[str, Any]:
        """合并后的完整配置快照，供检索 / 生成链路使用。"""
        data: dict[str, Any] = {
            spec.key: getattr(self.base, spec.key) for spec in SPECS
        }
        data.update(self.overrides)  # 数据库覆盖优先
        for key in READONLY_KEYS:
            data[key] = getattr(self.base, key, None)
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self.effective().get(key, default)

    def public_snapshot(
        self, suggestions: dict[str, list[str]] | None = None
    ) -> dict[str, Any]:
        """给前端的版本：密钥脱敏，并带上完整层级与依赖元信息。

        `suggestions` 是调用方（API 层）在请求时向 Ollama 查出来的模型列表，
        按 `choices_source` 归类。之所以由外部传进来而不是在这里查：
        `public_snapshot` 是同步方法，而查 Ollama 是异步 IO，
        混在一起会把同步 HTTP 调用埋进一个看起来纯读取的方法里。
        """
        suggestions = suggestions or {}
        values = self.effective()
        items = []
        for spec in SPECS:
            raw = values.get(spec.key)
            if spec.secret:
                shown = "********" if raw else ""
            else:
                shown = raw

            # 建议列表里若不含当前值，把它补进去 ——
            # 否则界面上的输入框会看起来「当前值不在可选项里」，像是配置坏了。
            # 典型场景：用户手填了一个还没 pull 的模型名。
            options: list[str] | None = None
            loaded = True
            if spec.choices_source:
                # 来源没出现在 suggestions 里 = 根本没查成功（Ollama 未启动）。
                # 这个区分很重要：界面要能分辨「Ollama 里没有这个模型」
                # （真问题，该报警）和「压根没连上 Ollama」（不该把每个库都标红）。
                loaded = spec.choices_source in suggestions
                available = list(suggestions.get(spec.choices_source) or [])
                options = []
                matched = False
                if isinstance(shown, str) and shown.strip():
                    target = _normalize_model_name(shown)
                    for name in available:
                        if _normalize_model_name(name) == target:
                            # 用当前配置里的写法展示，避免同一模型出现两种拼法
                            options.append(shown)
                            matched = True
                        else:
                            options.append(name)
                    if not matched:
                        options.insert(0, shown)
                else:
                    options = available
            elif spec.choices:
                options = list(spec.choices)

            items.append(
                {
                    "key": spec.key,
                    "value": shown,
                    "configured": bool(raw) if spec.secret else True,
                    "overridden": spec.key in self.overrides,
                    "label": spec.label,
                    "group": spec.group,
                    "section": spec.section,
                    "description": spec.description,
                    "type": spec.type.__name__,
                    "secret": spec.secret,
                    "choices": options,
                    "choice_labels": dict(spec.choice_labels) or None,
                    "choices_source": spec.choices_source,
                    "choices_loaded": loaded,
                    "minimum": spec.minimum,
                    "maximum": spec.maximum,
                    "scope": spec.scope,
                    "depends_on": (
                        {"key": spec.depends_on[0], "values": list(spec.depends_on[1])}
                        if spec.depends_on
                        else None
                    ),
                }
            )

        # 分组顺序与一句话定位由后端给出，前端只负责渲染
        used = {spec.group for spec in SPECS}
        groups = [
            {"name": g.name, "summary": g.summary, "order": g.order}
            for g in sorted(GROUPS, key=lambda g: g.order)
            if g.name in used
        ]
        # 兜底：SPECS 里出现了未在 GROUPS 登记的分组，也要显示出来而不是静默丢弃
        for name in sorted(used - set(GROUP_BY_NAME)):
            groups.append({"name": name, "summary": "", "order": UNKNOWN_GROUP_ORDER})

        return {
            "items": items,
            "groups": groups,
            "readonly": {k: values.get(k) for k in READONLY_KEYS},
            "overridden_keys": sorted(self.overrides.keys()),
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def set_many(self, values: dict[str, Any]) -> dict[str, Any]:
        if not values:
            return self.effective()

        unknown = [k for k in values if k not in SPEC_BY_KEY]
        if unknown:
            raise SettingsValidationError(
                f"不支持的配置项：{'、'.join(unknown)}"
            )

        # 先全量校验再落库，避免「改了一半失败」
        staged: dict[str, Any] = {}
        for key, raw in values.items():
            spec = SPEC_BY_KEY[key]
            # 密钥留空表示「不修改」，避免前端回显脱敏串把真 key 覆盖掉
            if spec.secret and (raw is None or str(raw).strip() in ("", "********")):
                continue
            staged[key] = _coerce(spec, raw)

        # 跨字段校验：overlap 必须小于 size（可能只有其中一个本次被修改）
        merged = self.effective()
        merged.update(staged)
        if merged["chunk_overlap"] >= merged["chunk_size"]:
            raise SettingsValidationError(
                f"切片重叠（{merged['chunk_overlap']}）必须小于切片长度（{merged['chunk_size']}）"
            )

        for key, value in staged.items():
            row = self.db.get(AppSetting, key)
            if row is None:
                self.db.add(AppSetting(key=key, value={"v": value}))
            else:
                row.value = {"v": value}
        self.db.commit()
        self._overrides = None  # 失效缓存
        return self.effective()

    def reset(self, keys: list[str] | None = None) -> dict[str, Any]:
        """还原为 .env 默认值。keys 为空表示全部还原。"""
        stmt = delete(AppSetting)
        if keys:
            stmt = stmt.where(AppSetting.key.in_(keys))
        self.db.execute(stmt)
        self.db.commit()
        self._overrides = None
        return self.effective()


__all__ = [
    "DYNAMIC_CHOICE_SOURCES",
    "GROUP_BY_NAME",
    "GROUPS",
    "READONLY_KEYS",
    "SECRET_KEYS",
    "SOURCE_OLLAMA_CHAT_MODELS",
    "SOURCE_OLLAMA_EMBED_MODELS",
    "SPECS",
    "SPEC_BY_KEY",
    "GroupSpec",
    "SettingSpec",
    "SettingsService",
    "SettingsValidationError",
]
