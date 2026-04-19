
import os, json, sqlite3, logging, uuid, time
from datetime import datetime, date
from typing import Optional, Dict, List, Any

log     = logging.getLogger("gymid.db")
project_id = os.environ.get("RAILWAY_PROJECT_ID", "562acf8c-c0a6-41e8-b4d7-f37249b91679")
volume_name = os.environ.get("RAILWAY_VOLUME_NAME", "vol_3cygu5nc1sov070u")
mount_path = os.path.join("/var/lib/containers/railwayapp/bind-mounts", project_id, volume_name)
DEFAULT_DB_PATH = os.path.join(mount_path, "gymid.db")
DB_PATH = os.environ.get("DB_PATH", "")
if not DB_PATH or not DB_PATH.strip():
    DB_PATH = DEFAULT_DB_PATH

class Database:
    def __init__(self, path=DB_PATH):
        self.path = path or DEFAULT_DB_PATH
        db_dir = os.path.dirname(self.path)
        # Wait for volume to be mounted
        for i in range(60):
            if os.path.exists(db_dir):
                break
            if i % 10 == 0:
                log.info(f"Waiting for volume mount at {db_dir}... ({i+1}/60)")
            time.sleep(1)
        else:
            raise RuntimeError(f"Volume not mounted after 60s: {db_dir}")
        os.makedirs(db_dir, exist_ok=True)
        self._init_schema()
        log.info(f"DB ready: {path}")

    def _conn(self):
        c = sqlite3.connect(self.path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as e:
            log.warning(f"WAL journal mode failed ({e}), falling back to default")
            try:
                c.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError as e2:
                log.warning(f"DELETE journal mode also failed ({e2}), proceeding without journal")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS members(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE, phone TEXT NOT NULL,
                    plan TEXT NOT NULL, photo_count INTEGER DEFAULT 0,
                    embedding_count INTEGER DEFAULT 0,
                    registered_at TEXT NOT NULL, active INTEGER DEFAULT 1);
                CREATE TABLE IF NOT EXISTS face_embeddings(
                    id TEXT PRIMARY KEY, member_id TEXT NOT NULL,
                    embedding TEXT NOT NULL, is_mean INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(id) ON DELETE CASCADE);
                CREATE TABLE IF NOT EXISTS checkins(
                    id TEXT PRIMARY KEY, member_id TEXT NOT NULL,
                    confidence REAL NOT NULL, checkin_at TEXT NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(id));
                CREATE INDEX IF NOT EXISTS idx_emb ON face_embeddings(member_id);
                CREATE INDEX IF NOT EXISTS idx_ci  ON checkins(checkin_at);
            """)

    def insert_member(self, m):
        with self._conn() as c:
            c.execute(
                "INSERT INTO members(id,name,email,phone,plan,photo_count,embedding_count,registered_at) VALUES(?,?,?,?,?,?,?,?)",
                (m["id"],m["name"],m["email"],m["phone"],m["plan"],
                 m.get("photo_count",0),m.get("embedding_count",0),datetime.now().isoformat()))

    def member_exists_by_email(self, email):
        with self._conn() as c:
            return c.execute("SELECT id FROM members WHERE email=?",(email,)).fetchone() is not None

    def get_member(self, mid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM members WHERE id=? AND active=1",(mid,)).fetchone()
            return dict(r) if r else None

    def list_members(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM members WHERE active=1 ORDER BY registered_at DESC").fetchall()]

    def count_members(self):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM members WHERE active=1").fetchone()[0]

    def delete_member(self, mid):
        with self._conn() as c:
            c.execute("UPDATE members SET active=0 WHERE id=?",(mid,))

    def insert_embedding(self, member_id, embedding, is_mean=False):
        with self._conn() as c:
            c.execute(
                "INSERT INTO face_embeddings(id,member_id,embedding,is_mean,created_at) VALUES(?,?,?,?,?)",
                (str(uuid.uuid4()),member_id,json.dumps(embedding),1 if is_mean else 0,datetime.now().isoformat()))

    def get_all_embeddings(self):
        """Returns individual embeddings (not means) for accurate matching."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT fe.member_id,fe.embedding FROM face_embeddings fe "
                "JOIN members m ON m.id=fe.member_id WHERE m.active=1 AND fe.is_mean=0"
            ).fetchall()
            if rows:
                result = {}
                for row in rows: result.setdefault(row["member_id"],[]).append(json.loads(row["embedding"]))
                return result
            # Fallback: all embeddings (legacy)
            rows = c.execute(
                "SELECT fe.member_id,fe.embedding FROM face_embeddings fe "
                "JOIN members m ON m.id=fe.member_id WHERE m.active=1"
            ).fetchall()
            result = {}
            for row in rows: result.setdefault(row["member_id"],[]).append(json.loads(row["embedding"]))
            return result

    def get_embeddings_for_member(self, mid):
        with self._conn() as c:
            return [json.loads(r["embedding"]) for r in
                    c.execute("SELECT embedding FROM face_embeddings WHERE member_id=?",(mid,)).fetchall()]

    def log_checkin(self, member_id, confidence):
        cid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute("INSERT INTO checkins(id,member_id,confidence,checkin_at) VALUES(?,?,?,?)",
                      (cid,member_id,confidence,datetime.now().isoformat()))
        return cid

    def get_checkins_for_date(self, day):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT c.id,c.member_id,m.name,m.email,m.phone,m.plan,c.confidence,c.checkin_at "
                "FROM checkins c JOIN members m ON m.id=c.member_id "
                "WHERE DATE(c.checkin_at)=? ORDER BY c.checkin_at DESC",(day,)).fetchall()]

    def get_all_checkins(self, limit=100):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT c.id,c.member_id,m.name,m.plan,c.confidence,c.checkin_at "
                "FROM checkins c JOIN members m ON m.id=c.member_id "
                "ORDER BY c.checkin_at DESC LIMIT ?",(limit,)).fetchall()]
