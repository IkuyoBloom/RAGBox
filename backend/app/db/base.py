"""数据库引擎与会话。

SQLite + WAL：
  * WAL 让「后台索引线程写」与「接口线程读」不互相阻塞，这在文档上传后
    异步建索引的场景里很重要。
  * 打开 foreign_keys pragma，否则 SQLite 默认不校验外键，删库会留孤儿行。
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()
settings.ensure_dirs()

engine: Engine = create_engine(
    settings.sqlalchemy_url,
    echo=False,
    future=True,
    # FastAPI 的线程池会跨线程复用连接
    connect_args={"check_same_thread": False},
    # ---- 连接池 ----
    # 默认是 5 + 10 溢出 = 15。对本地单用户应用本应够用，但批量入库时
    # 「后台入库任务」与「界面每 2 秒轮询 2 个接口」会同时抢连接，
    # 15 个用完之后，后续请求全部排队 —— 实测批量导入 118 份文档时
    # 连接池被打满，报 QueuePool limit ... timeout 30.00，
    # 界面表现为整个后端卡死。
    pool_size=20,
    max_overflow=40,
    # 拿不到连接就尽快失败，**不要**让请求堆在那里等 30 秒。
    # 堆等会形成恶性螺旋：前端轮询越积越多 -> 连接越占越满 -> 更慢。
    # 快速失败至少能让界面立刻显示错误，而不是无限转圈。
    pool_timeout=8,
    # SQLite 连接放久了可能被系统回收，定期重建
    pool_recycle=1800,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - 基础设施
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入用的会话生成器。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """脚本 / 后台任务用的会话上下文，自动提交或回滚。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """建表（幂等）。"""
    from app.db import models  # noqa: F401  确保模型已注册到 metadata

    Base.metadata.create_all(bind=engine)
