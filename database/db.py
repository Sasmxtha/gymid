
import os, json, sqlite3, logging, uuid, time
from datetime import datetime, date
from typing import Optional, Dict, List, Any

log = logging.getLogger("gymid.db")

def _get_db_path():
    """Detect Railway volume mount or use fallback."""
    # Try explicit env var first
    db_path = os.environ.get("DB_PATH", "").strip()
    if db_path:
        return db_path
    
    # Try RAILWAY_VOLUME_MOUNT_PATH (set by some Railway versions)
    volume_mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if volume_mount and os.path.exists(volume_mount):
        if _test_volume_writeability(volume_mount):
            return os.path.join(volume_mount, "gymid.db")
        else:
            log.warning(f"Volume {volume_mount} is not reliably writable, falling back to /tmp")
    
    # Try to find actual mounted volume in Railway's standard location
    try:
        base = "/var/lib/containers/railwayapp/bind-mounts"
        if os.path.exists(base):
            for project_dir in os.listdir(base):
                project_path = os.path.join(base, project_dir)
                if os.path.isdir(project_path):
                    for vol_dir in os.listdir(project_path):
                        vol_path = os.path.join(project_path, vol_dir)
                        if os.path.isdir(vol_path):
                            if _test_volume_writeability(vol_path):
                                db_path = os.path.join(vol_path, "gymid.db")
                                log.info(f"Found writable Railway volume mount at {vol_path}")
                                return db_path
                            else:
                                log.warning(f"Volume {vol_path} has I/O issues, skipping")
    except Exception as e:
        log.debug(f"Failed to scan for Railway mount: {e}")
    
    # Fallback to /tmp (ephemeral but reliable on Railway)
    log.warning("Using /tmp/gymid.db - database will not persist across container restarts")
    return "/tmp/gymid.db"

def _test_volume_writeability(path):
    """Test if a volume is writable and doesn't have I/O errors."""
    try:
        test_file = os.path.join(path, ".write_test_" + str(uuid.uuid4())[:8])
        with open(test_file, "w") as f:
            f.write("test")
        os.fsync(test_file)  # Force to disk
        os.remove(test_file)
        return True
    except (IOError, OSError) as e:
        log.debug(f"Volume {path} writability test failed: {e}")
        return False

DB_PATH = _get_db_path()
log.info(f"Using database path: {DB_PATH}")

class Database:
    def __init__(self, path=DB_PATH):
        self.path = path or DB_PATH
        db_dir = os.path.dirname(self.path)
        if not db_dir:
            db_dir = os.getcwd()
            self.path = os.path.join(db_dir, os.path.basename(self.path) or "gymid.db")
        os.makedirs(db_dir, exist_ok=True)
        self._init_schema()
        log.info(f"DB ready: {self.path}")

    def _conn(self):
        try:
            c = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0)
            c.row_factory = sqlite3.Row
            # Try to set pragmas, but don't fail if they don't work
            try:
                c.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                try:
                    c.execute("PRAGMA journal_mode=DELETE")
                except sqlite3.OperationalError:
                    log.debug("Journal modes unavailable, continuing without")
            try:
                c.execute("PRAGMA foreign_keys=ON")
            except sqlite3.OperationalError:
                log.debug("Foreign keys pragma unavailable")
            return c
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Cannot connect to database at {self.path}: {e}. Check volume mount.")

    def _init_schema(self):
        statements = [
            """CREATE TABLE IF NOT EXISTS members(
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE, phone TEXT NOT NULL,
                plan TEXT NOT NULL, photo_count INTEGER DEFAULT 0,
                embedding_count INTEGER DEFAULT 0,
                registered_at TEXT NOT NULL, active INTEGER DEFAULT 1)""",
            """CREATE TABLE IF NOT EXISTS face_embeddings(
                id TEXT PRIMARY KEY, member_id TEXT NOT NULL,
                embedding TEXT NOT NULL, is_mean INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(member_id) REFERENCES members(id) ON DELETE CASCADE)""",
            """CREATE TABLE IF NOT EXISTS checkins(
                id TEXT PRIMARY KEY, member_id TEXT NOT NULL,
                confidence REAL NOT NULL, checkin_at TEXT NOT NULL,
                FOREIGN KEY(member_id) REFERENCES members(id))""",
            "CREATE INDEX IF NOT EXISTS idx_emb ON face_embeddings(member_id)",
            "CREATE INDEX IF NOT EXISTS idx_ci ON checkins(checkin_at)",
        ]
        
        io_errors = 0
        for i, stmt in enumerate(statements):
            for attempt in range(2):
                try:
                    with self._conn() as c:
                        c.execute(stmt)
                        c.commit()
                    io_errors = 0  # Reset on success
                    break
                except sqlite3.OperationalError as e:
                    if "disk I/O error" in str(e):
                        io_errors += 1
                        if io_errors >= 2:
                            log.error(f"Persistent disk I/O errors on volume, switching to /tmp")
                            self._switch_to_tmp()
                            # Retry with /tmp
                            try:
                                with self._conn() as c:
                                    c.execute(stmt)
                                    c.commit()
                                break
                            except Exception as e2:
                                log.error(f"Schema statement {i} failed on /tmp: {e2}")
                                raise
                    
                    if attempt < 1:
                        log.warning(f"Schema statement {i} failed (attempt {attempt+1}/2): {e}, retrying...")
                        time.sleep(0.5)
                    else:
                        log.error(f"Schema statement {i} failed after 2 attempts: {e}")
                        raise
    
    def _switch_to_tmp(self):
        """Switch to /tmp database."""
        global DB_PATH
        new_path = "/tmp/gymid.db"
        log.warning(f"Switching database from {self.path} to {new_path}")
        self.path = new_path
        DB_PATH = new_path
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

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
