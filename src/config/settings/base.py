"""
Generated by 'django-admin startproject' using Django 4.1.5.

"""

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parents[2]
env = environ.Env(DEBUG=(bool, False))
environ.Env.read_env(BASE_DIR / ".env")

DEBUG = env("DEBUG")

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!

# Application definition

INSTALLED_APPS = [
    "grappelli",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "defender",
    "request",  # webstats module
    "accounts",
    "content",
    "sheets",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "defender.middleware.FailedLoginMiddleware",
    "request.middleware.RequestMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES = {"default": env.db()}
WAREHOUSE_URL = env("WAREHOUSE_URL")

# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = "fr-FR"

TIME_ZONE = "Europe/Paris"

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_FINDERS = (
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
)
STATICFILES_DIR = BASE_DIR / "static"
STATICFILES_DIRS = [
    STATICFILES_DIR,
]

# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

ADMIN_SLUG = env("ADMIN_SLUG")

AUTH_USER_MODEL = "accounts.User"

LOGIN_REDIRECT_URL = "/"

CSV_FILES_DIR = BASE_DIR / "csv"

MEDIA_ROOT = BASE_DIR.parent / "public" / "medias"
MEDIA_URL = "/medias/"

SITE_ID = env.int("SITE_ID", 1)

# path to ignore for requests stats
REQUEST_IGNORE_PATHS = (
    rf"^{ADMIN_SLUG}/",
    r"/sheets/compute-fragment/",
    r"/static/",
    r"favicon.ico",
)

# defender
DEFENDER_REDIS_URL = env.str("DEFENDER_REDIS_URL", "redis://localhost:6379/0")
DEFENDER_LOGIN_FAILURE_LIMIT_USERNAME = 3
DEFENDER_LOGIN_FAILURE_LIMIT_IP = 3
DEFENDER_LOCKOUT_TEMPLATE = "accounts/lockout.html"
DEFENDER_LOCKOUT_COOLOFF_TIME = 5 * 60  # seconds

BASE_URL = env.str("BASE_URL", "http://127.0.0.1:8000")
LOGIN_URL = "login"
LOGOUT_REDIRECT_URL = "home"

PASSWORD_RESET_TIMEOUT = 3600 * 12  # 12 hours

GRAPPELLI_ADMIN_TITLE = "Trackdéchets - Inspection"

DEFAULT_FROM_EMAIL = "sender@test.fr"
MESSAGE_RECIPIENTS = env.list("MESSAGE_RECIPIENTS", [])
