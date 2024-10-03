"""
Generated by 'django-admin startproject' using Django 4.1.5.

"""

from pathlib import Path

import environ
from django.urls import reverse_lazy

BASE_DIR = Path(__file__).resolve().parents[2]
env = environ.Env(DEBUG=(bool, False))
environ.Env.read_env(BASE_DIR / ".env")

DEBUG = env("DEBUG")

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!

# Application definition

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    "grappelli.dashboard",
    "grappelli",
    "django.contrib.admin",
    "template_partials",
    "anymail",
    "defender",
    "django_otp",
    "django_otp.plugins.otp_email",
    "request",  # webstats module
    "simple_menu",
    "rest_framework",
    "rest_framework.authtoken",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.openid_connect",
    "aiot_provider",
    "accounts",
    "content",
    "roadcontrol",
    "sheets",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django_otp.middleware.OTPMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "defender.middleware.FailedLoginMiddleware",
    "request.middleware.RequestMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    "aiot_provider.middleware.MonaiotMiddleware",
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

AUTHENTICATION_BACKENDS = [
    # Needed to login by username in Django admin, regardless of `allauth`
    "django.contrib.auth.backends.ModelBackend",
    # `allauth` specific authentication methods, such as login by email
    "allauth.account.auth_backends.AuthenticationBackend",
]

ADMIN_SLUG = env("ADMIN_SLUG")
API_SLUG = env("API_SLUG")

AUTH_USER_MODEL = "accounts.User"

LOGIN_REDIRECT_URL = reverse_lazy("second_factor")

CSV_FILES_DIR = BASE_DIR / "csv"

MEDIA_ROOT = BASE_DIR.parent / "public" / "medias"
MEDIA_URL = "/medias/"

SITE_ID = env.int("SITE_ID", 1)

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework.authentication.BasicAuthentication"],
}

# defender
REDIS_URL = env.str("REDIS_URL", "redis://localhost:6379/0")
DEFENDER_REDIS_URL = REDIS_URL
DEFENDER_LOGIN_FAILURE_LIMIT_USERNAME = 3
DEFENDER_LOGIN_FAILURE_LIMIT_IP = 3
DEFENDER_LOCKOUT_TEMPLATE = "accounts/lockout.html"
DEFENDER_LOCKOUT_COOLOFF_TIME = 5 * 60  # seconds

BASE_URL = env.str("BASE_URL", "http://127.0.0.1:8000")
LOGIN_URL = "login"
LOGOUT_REDIRECT_URL = "private_home"

PASSWORD_RESET_TIMEOUT = 3600 * 12  # 12 hours

GRAPPELLI_ADMIN_TITLE = "Trackdéchets - Inspection"
GRAPPELLI_INDEX_DASHBOARD = "config.dashboard.CustomIndexDashboard"

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
DEFAULT_FROM_EMAIL = "sender@test.fr"
MESSAGE_RECIPIENTS = env.list("MESSAGE_RECIPIENTS", [])

OTP_EMAIL_SUBJECT = "Votre code pour Trackdéchets fiche établissement"
OTP_EMAIL_BODY_TEMPLATE_PATH = "emails/second_factor/second_factor.txt"
OTP_EMAIL_BODY_HTML_TEMPLATE_PATH = "emails/second_factor/second_factor.html"
OTP_EMAIL_THROTTLE_DELAY = 300  # s

# allauth monaiot
SOCIALACCOUNT_ONLY = True
ACCOUNT_ADAPTER = "aiot_provider.account_adapter.MonaiotAccountAdapter"
SOCIALACCOUNT_ADAPTER = "aiot_provider.account_adapter.MonaiotSocialAccountAdapter"
SOCIALACCOUNT_ENABLED = True
SOCIALACCOUNT_STORE_TOKENS = True
SOCIALACCOUNT_EMAIL_AUTHENTICATION = True
SOCIALACCOUNT_AUTO_SIGNUP = True
AUTO_SIGNUP = True
ACCOUNT_EMAIL_VERIFICATION = "none"
MONAIOT_SERVER_URL = env("MONAIOT_SERVER_URL")
MONAIOT_REALM = env("MONAIOT_REALM")
MONAIOT_CLIENT_ID = env("MONAIOT_CLIENT_ID")
MONAIOT_SECRET = env("MONAIOT_SECRET")
MONAIOT_SCOPES = env.list("MONAIOT_SCOPES", default=[])
SOCIALACCOUNT_FORMS = {"signup": "aiot_provider.forms.MonAiotSignupForm"}

WELL_KNOWN_URL = f"{MONAIOT_SERVER_URL}/auth/realms/{MONAIOT_REALM}/.well-known/openid-configuration"
SOCIALACCOUNT_PROVIDERS = {
    "monaiot": {
        "APPS": [
            {
                "provider_id": "monaiot",
                "name": "monaiot",
                "client_id": MONAIOT_CLIENT_ID,
                "secret": MONAIOT_SECRET,
                "settings": {"server_url": WELL_KNOWN_URL, "token_auth_method": "client_secret_jwt"},
            }
        ],
        "SCOPE": MONAIOT_SCOPES,
    }
}

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
    ],
}
CSRF_TRUSTED_ORIGINS = env.list("CSRF_TRUSTED_ORIGINS", default=[])

WEB_QUEUE = "web-queue"
API_QUEUE = "api-queue"

# API
TD_API_URL = env("TD_API_URL")
TD_API_TOKEN = env("TD_API_TOKEN")

# WEBHOOKS

TD_WEBHOOK_URL = env("TD_WEBHOOK_URL")
TD_WEBHOOK_TOKEN = env("TD_WEBHOOK_TOKEN")

# Web  stats:  path to ignore
REQUEST_IGNORE_PATHS = (
    rf"^{ADMIN_SLUG}/",
    r"/sheets/compute-fragment/",
    r"/static/",
    r"favicon.ico",
)

# Storages

AWS_S3_ENDPOINT_URL = env("AWS_S3_ENDPOINT_URL")
AWS_BUCKET_NAME = env("AWS_BUCKET_NAME")
AWS_ACCESS_KEY_ID = env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env("AWS_SECRET_ACCESS_KEY")
AWS_S3_REGION_NAME = env("AWS_S3_REGION_NAME")
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "private_s3": {
        "BACKEND": "roadcontrol.storage_backends.PrivateMediaStorage",
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}
