from .base import *  # noqa
from .base import env

SECRET_KEY = "xyzabcdefghu"

INSTALLED_APPS += [  # noqa F405
    "whitenoise.runserver_nostatic",
    "debug_toolbar",
    "django_extensions",
]

ALLOWED_HOSTS = ["*"]

INTERNAL_IPS = [
    "127.0.0.1",
]

MIDDLEWARE = (
    MIDDLEWARE[:1]  # noqa
    + [
        "debug_toolbar.middleware.DebugToolbarMiddleware",
    ]
    + MIDDLEWARE[1:]  # noqa
)

# Celery config
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis"

OTP_EMAIL_TOKEN_VALIDITY = 600
ALLOWED_HOSTS = ["dev.inspection.trackdechets.beta.gouv.fr", "127.0.0.1"]


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


SKIP_ROAD_CONTROL_SIRET_CHECK = env("SKIP_ROAD_CONTROL_SIRET_CHECK")

GDAL_LIBRARY_PATH = env("GDAL_LIBRARY_PATH")
GEOS_LIBRARY_PATH = env("GEOS_LIBRARY_PATH")


DJANGO_VITE = {"default": {"dev_mode": True}}
