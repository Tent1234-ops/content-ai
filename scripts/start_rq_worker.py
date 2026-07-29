"
Start an RQ worker connected to the Redis URL defined in environment or admin settings.
Usage:
  python scripts/start_rq_worker.py --redis redis://localhost:6379/0 --queue default

This script is a lightweight helper for local development. In production, run q worker directly
or use supervisor/systemd to manage the worker process.
"
