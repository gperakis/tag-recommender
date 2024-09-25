import contextvars

launch_date = contextvars.ContextVar("launch_date", default=None)
