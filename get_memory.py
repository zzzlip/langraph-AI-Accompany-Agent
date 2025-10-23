import sqlite3
from datetime import datetime
from langchain_core.messages import BaseMessage
from langchain_core.load import dumps, loads


class DatabaseManager:
    """
    一个用于管理人物简介和聊天记忆的数据库操作类。
    【V4版 - Sync】支持同步操作，并直接存储和检索 LangChain 消息对象列表。
    """
    def __init__(self, db_path="memory_data.db"):
        """
        初始化数据库路径。连接将在每个同步方法中按需创建。
        :param db_path: SQLite数据库文件的路径。
        """
        self.db_path = db_path


    def initialize(self):
        """
        ### CHANGE: New sync method to set up tables.
        This must be called once after creating an instance of the class.
        """
        with sqlite3.connect(self.db_path) as db:
            db.execute('''
                CREATE TABLE IF NOT EXISTS character_profiles (
                    uuid TEXT PRIMARY KEY,
                    profile_content TEXT,
                    updated_at TIMESTAMP
                )
            ''')
            db.execute('''
                CREATE TABLE IF NOT EXISTS chat_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL,
                    event_tag TEXT NOT NULL,
                    memory_content TEXT, -- 将存储序列化后的消息列表
                    updated_at TIMESTAMP,
                    UNIQUE(uuid, event_tag)
                )
            ''')
            db.commit()

    def add_or_update_profile(self, user_uuid: str, content: str):
        """
        ### CHANGE: Converted to sync.
        添加或更新用户简介。
        """
        print(f"[*] (Sync) 正在为 UUID: {user_uuid} 添加/更新简介...")
        with sqlite3.connect(self.db_path) as db:
            db.execute('''
                INSERT OR REPLACE INTO character_profiles (uuid, profile_content, updated_at)
                VALUES (?, ?, ?)
            ''', (user_uuid, content, datetime.now()))
            db.commit()
        print(f"[+] (Sync) 简介操作完成。")

    def get_profile(self, user_uuid: str) -> str | None:
        """
        ### CHANGE: Converted to sync.
        根据uuid查询人物简介。
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute("SELECT profile_content FROM character_profiles WHERE uuid = ?",
                                  (user_uuid,))
            result = cursor.fetchone()
        return result[0] if result else None

    def add_memory(self, user_uuid: str, event_tags: list[str], new_messages: list[BaseMessage]):
        """
        ### CHANGE: Converted to sync.
        【核心功能修改】存储或叠加 LangChain 消息对象列表到多个标签。
        """
        if not event_tags or not new_messages:
            print("[!] 警告：传入的标签或消息为空，操作已跳过。")
            return

        print(f"[*] (Sync) 正在为 UUID: {user_uuid} 的事件标签 {event_tags} 添加/叠加 {len(new_messages)} 条消息...")

        with sqlite3.connect(self.db_path) as db:
            for tag in event_tags:
                cursor = db.execute(
                        "SELECT memory_content FROM chat_memories WHERE uuid = ? AND event_tag = ?",
                        (user_uuid, tag)
                )
                result = cursor.fetchone()

                if result and result[0]:
                    print(f"    - 标签 '{tag}': 发现已有记忆，进行叠加...")
                    existing_messages = loads(result[0])
                    combined_messages = existing_messages + new_messages
                    updated_serialized_memory = dumps(combined_messages)
                    db.execute('''
                        UPDATE chat_memories
                        SET memory_content = ?, updated_at = ?
                        WHERE uuid = ? AND event_tag = ?
                    ''', (updated_serialized_memory, datetime.now(), user_uuid, tag))
                else:
                    print(f"    - 标签 '{tag}': 未发现记忆，创建新条目...")
                    serialized_new_messages = dumps(new_messages)
                    db.execute('''
                        INSERT INTO chat_memories (uuid, event_tag, memory_content, updated_at)
                        VALUES (?, ?, ?, ?)
                    ''', (user_uuid, tag, serialized_new_messages, datetime.now()))

            db.commit()
        print(f"[+] (Sync) 记忆操作完成。")

    def get_memory(self, user_uuid: str, event_tag: str) -> list[BaseMessage] | None:
        """
        ### CHANGE: Converted to sync.
        【核心功能修改】根据uuid和事件标签查询聊天记忆。
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                    "SELECT memory_content FROM chat_memories WHERE uuid = ? AND event_tag = ?",
                    (user_uuid, event_tag)
            )
            result = cursor.fetchone()

        if result and result[0]:
            return loads(result[0])
        return None

    def get_all_tags(self, user_uuid: str) -> list[str]:
        """
        ### CHANGE: Converted to sync.
        根据uuid查询该用户拥有的所有事件标签。
        """
        print(f"[*] (Sync) 正在查询 UUID: {user_uuid} 的所有标签...")
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                    "SELECT DISTINCT event_tag FROM chat_memories WHERE uuid = ?",
                    (user_uuid,)
            )
            results = cursor.fetchall()

        tags = [row[0] for row in results]
        print(f"[+] (Sync) 查询到标签: {tags}")
        return tags


    def close(self):
        """(Sync) 关闭数据库连接。在使用 'with' 模式下，此方法不是必需的。"""
        print("[*] (Sync) 数据库操作模式不需要手动关闭连接。")
        pass


