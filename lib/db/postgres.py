"""
PostgreSQL connection management for Polymath v4.

Uses psycopg2 with a simple connection pool pattern.
All database access should go through these functions.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Optional, Any, Generator, List, Dict
from queue import Queue, Empty

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from lib.config import config

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Simple thread-safe connection pool for psycopg2.

    Usage:
        pool = ConnectionPool(dsn, min_size=2, max_size=10)
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents")
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
    ):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Queue = Queue(maxsize=max_size)
        self._size = 0
        self._lock = threading.Lock()

        # Pre-create minimum connections
        for _ in range(min_size):
            self._add_connection()

    def _add_connection(self) -> None:
        """Create a new connection and add to pool."""
        with self._lock:
            if self._size >= self.max_size:
                return
            try:
                conn = psycopg2.connect(self.dsn)
                conn.autocommit = False
                self._pool.put_nowait(conn)
                self._size += 1
                logger.debug(f"Created connection {self._size}/{self.max_size}")
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
                raise

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get a connection from the pool."""
        try:
            conn = self._pool.get_nowait()
            # Test if connection is still alive
            try:
                conn.cursor().execute("SELECT 1")
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                logger.warning("Dead connection detected, creating new one")
                with self._lock:
                    self._size -= 1
                conn = self._create_new_connection()
            return conn
        except Empty:
            # Pool empty, try to create new connection
            if self._size < self.max_size:
                return self._create_new_connection()
            # Wait for available connection
            return self._pool.get(timeout=30)

    def _create_new_connection(self) -> psycopg2.extensions.connection:
        """Create a new connection (called when pool is empty)."""
        with self._lock:
            if self._size >= self.max_size:
                # Race condition - another thread created one
                return self._pool.get(timeout=30)
            conn = psycopg2.connect(self.dsn)
            conn.autocommit = False
            self._size += 1
            logger.debug(f"Created connection {self._size}/{self.max_size}")
            return conn

    def _return_connection(self, conn: psycopg2.extensions.connection) -> None:
        """Return a connection to the pool."""
        try:
            # Reset connection state
            if not conn.closed:
                conn.rollback()
                self._pool.put_nowait(conn)
            else:
                with self._lock:
                    self._size -= 1
        except Exception as e:
            logger.warning(f"Error returning connection: {e}")
            with self._lock:
                self._size -= 1

    @contextmanager
    def connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a connection from the pool (context manager)."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._return_connection(conn)

    def close(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
        with self._lock:
            self._size = 0
        logger.info("Connection pool closed")


# Global connection pool
_pool: Optional[ConnectionPool] = None


def get_pool(min_size: int = None, max_size: int = None) -> ConnectionPool:
    """
    Get or create the global connection pool.

    Args:
        min_size: Override minimum connections (default: PG_POOL_MIN)
        max_size: Override maximum connections (default: PG_POOL_MAX)

    Returns:
        ConnectionPool instance
    """
    global _pool

    if _pool is None:
        min_size = min_size if min_size is not None else config.PG_POOL_MIN
        max_size = max_size if max_size is not None else config.PG_POOL_MAX

        logger.info(f"Creating Postgres connection pool (min={min_size}, max={max_size})")
        _pool = ConnectionPool(
            config.POSTGRES_DSN,
            min_size=min_size,
            max_size=max_size,
        )

    return _pool


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Get a connection from the pool.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents LIMIT 10")
                results = cur.fetchall()
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def get_db_connection() -> psycopg2.extensions.connection:
    """
    Get a standalone connection (not pooled).

    Use this for scripts that need their own connection.
    Caller is responsible for closing the connection.

    Returns:
        psycopg2 connection
    """
    return psycopg2.connect(config.POSTGRES_DSN)


def execute_query(
    query: str,
    params: Optional[tuple] = None,
    fetch: bool = True
) -> Optional[List[Dict]]:
    """
    Execute a query and optionally fetch results.

    Args:
        query: SQL query string
        params: Query parameters
        fetch: Whether to fetch and return results

    Returns:
        List of result dicts if fetch=True, else None
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if fetch:
                return [dict(row) for row in cur.fetchall()]
            conn.commit()
            return None


def execute_many(query: str, params_list: List[tuple]) -> int:
    """
    Execute a query with multiple parameter sets.

    Args:
        query: SQL query string with placeholders
        params_list: List of parameter tuples

    Returns:
        Number of rows affected
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, params_list)
            conn.commit()
            return cur.rowcount


def get_stats() -> Dict[str, int]:
    """Get system statistics from the database."""
    stats = {}

    queries = {
        "documents": "SELECT COUNT(*) FROM documents",
        "passages": "SELECT COUNT(*) FROM passages",
        "passages_with_embeddings": "SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL",
        "concepts": "SELECT COUNT(*) FROM passage_concepts",
        "code_files": "SELECT COUNT(*) FROM code_files",
        "code_chunks": "SELECT COUNT(*) FROM code_chunks",
    }

    with get_connection() as conn:
        with conn.cursor() as cur:
            for name, query in queries.items():
                try:
                    cur.execute(query)
                    result = cur.fetchone()
                    stats[name] = result[0] if result else 0
                except Exception as e:
                    logger.warning(f"Failed to get stat {name}: {e}")
                    stats[name] = 0

    return stats


def check_health() -> Dict[str, Any]:
    """Check database health and return status."""
    result = {
        "status": "unknown",
        "connection": False,
        "pgvector": False,
        "tables": [],
    }

    try:
        with get_connection() as conn:
            result["connection"] = True

            with conn.cursor() as cur:
                # Check pgvector extension
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
                if row:
                    result["pgvector"] = True
                    result["pgvector_version"] = row[0]

                # Check tables exist
                cur.execute("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename IN ('documents', 'passages', 'passage_concepts',
                                     'code_files', 'code_chunks')
                """)
                result["tables"] = [row[0] for row in cur.fetchall()]

        result["status"] = "healthy" if result["pgvector"] else "degraded"

    except Exception as e:
        result["status"] = "unhealthy"
        result["error"] = str(e)

    return result


def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("Postgres connection pool closed")


def configure_for_batch_worker() -> None:
    """
    Configure minimal connection pool for batch worker processes.

    IMPORTANT: Call this at the start of batch scripts to prevent
    connection exhaustion when running multiple workers.
    """
    global _pool

    if _pool is not None:
        logger.warning("Pool already initialized, skipping batch configuration")
        return

    # Force minimal pool for batch workers
    logger.info("Configuring minimal connection pool for batch worker (1 connection)")
    _pool = ConnectionPool(
        config.POSTGRES_DSN,
        min_size=1,
        max_size=1,
    )
