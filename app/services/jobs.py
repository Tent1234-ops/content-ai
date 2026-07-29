import threading
import uuid
import time
from typing import Dict, Callable

from app.runtime import get as runtime_get

_jobs: Dict[str, Dict] = {}
_queue = []
_queue_lock = threading.Lock()
_worker_thread = None
_worker_stop = threading.Event()

# Optional RQ support
try:
    import redis
    from rq import Queue, Connection
    _rq_available = True
except Exception:
    _rq_available = False


def _worker_loop():
    while not _worker_stop.is_set():
        job = None
        with _queue_lock:
            if _queue:
                job = _queue.pop(0)
        if not job:
            time.sleep(0.2)
            continue
        job_id = job["job_id"]
        func = job["func"]
        args = job.get("args", [])
        kwargs = job.get("kwargs", {})
        try:
            _jobs[job_id]["status"] = "running"
            result = func(*args, **kwargs)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)


def start_worker():
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _worker_stop.clear()
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    _worker_thread.start()


def stop_worker():
    _worker_stop.set()
    global _worker_thread
    if _worker_thread:
        _worker_thread.join(timeout=2)


def _enqueue_inprocess(func: Callable, *args, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None}
    with _queue_lock:
        _queue.append({"job_id": job_id, "func": func, "args": args, "kwargs": kwargs})
    start_worker()
    return job_id


def _enqueue_rq(func: Callable, *args, redis_url: str = None, **kwargs) -> str:
    if not _rq_available:
        raise RuntimeError("RQ/redis not available in this environment. Install rq and redis package to enable this backend.")
    if not redis_url:
        raise RuntimeError("redis_url must be provided in runtime configuration to use RQ backend")

    # Connect and enqueue using RQ. We serialize by referencing a module-level callable via import path.
    # For safety in this environment, we only support enqueueing callables that are module-level functions (picklable by rq).
    q = None
    try:
        from redis import Redis
        redis_conn = Redis.from_url(redis_url)
        q = Queue(connection=redis_conn)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Redis at {redis_url}: {e}")

    # RQ returns job id
    job = q.enqueue(func, *args, **kwargs)
    return job.get_id()


def enqueue(func: Callable, *args, **kwargs) -> str:
    """Enqueue a callable for background execution and return job_id

    Backends supported: inprocess, rq
    """
    backend = runtime_get("job_backend") or "inprocess"
    if backend == "inprocess":
        return _enqueue_inprocess(func, *args, **kwargs)
    elif backend == "rq":
        redis_url = runtime_get("redis_url")
        return _enqueue_rq(func, *args, redis_url=redis_url, **kwargs)
    else:
        raise RuntimeError(f"Background backend '{backend}' not implemented")


def get_status(job_id: str) -> Dict:
    backend = runtime_get("job_backend") or "inprocess"
    if backend == "inprocess":
        return _jobs.get(job_id, {"status": "not_found"})
    elif backend == "rq":
        if not _rq_available:
            return {"status": "error", "error": "rq/redis not available"}
        try:
            from redis import Redis
            from rq.job import Job
            redis_url = runtime_get("redis_url")
            conn = Redis.from_url(redis_url)
            job = Job.fetch(job_id, connection=conn)
            return {"status": job.get_status(), "result": job.result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    else:
        return {"status": "not_supported"}


# Ensure worker is started when module imported and backend is inprocess
if runtime_get("job_backend") == "inprocess":
    start_worker()
