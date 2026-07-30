# Runtime configuration store for in-memory settings that mirror admin DB settings
_runtime = {
    "asr_model": "small",  # default runtime model
    "enable_model_toggle": True,
    "job_backend": "inprocess",  # inprocess | rq | celery
    "redis_url": None,
    # Feature flags
    "dashboard_live": True,
}


def get(key, default=None):
    return _runtime.get(key, default)


def set(key, value):
    _runtime[key] = value


def dump():
    return dict(_runtime)
