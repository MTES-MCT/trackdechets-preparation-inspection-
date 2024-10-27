import uuid

from django.conf import settings
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.translation import gettext_lazy as _
from import_export import resources
from import_export.admin import ImportExportMixin

from .forms import AdminCustomUserChangeForm, AdminCustomUserCreationForm
from .models import User


class UserResource(resources.ModelResource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emails = User.objects.values_list("email", flat=True)

    def before_import_row(self, row, **kwargs):
        row["username"] = f"{row['first_name']} {row['last_name']}"

    def skip_row(self, instance, original, row, import_validation_errors=None):
        skip = super().skip_row(instance, original, row, import_validation_errors)
        # skip existing accounts
        if row["email"] in self.emails:
            return True
        return skip

    def before_save_instance(self, instance, row, **kwargs):
        instance.id = uuid.uuid4()
        instance.is_active = True
        instance.is_staff = False
        instance.is_superuser = False
        instance.user_type = User.UserTypeChoice.HUMAN
        instance.password = str(uuid.uuid4())
        instance.email = instance.email.lower()

    class Meta:
        model = User
        fields = ["id", "username", "email", "user_category"]


@admin.action(description="Send invitation email")
def send_invitation_email(_, request, queryset):
    for user in queryset:
        if user.last_login:
            continue
        current_site = get_current_site(request)

        domain = current_site.domain
        body = render_to_string("emails/invitation/welcome.html", {"domain": domain})

        message = EmailMultiAlternatives(
            subject="Trackdéchets - outil de préparation de fiche d'inspection",
            body=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email],
        )

        message.send(fail_silently=False)


@admin.register(User)
class CustomUserAdmin(ImportExportMixin, UserAdmin):
    add_form = AdminCustomUserCreationForm
    form = AdminCustomUserChangeForm
    model = User
    actions = [send_invitation_email]
    search_fields = ("username", "email")
    list_display = [
        "email",
        "user_type",
        "user_category",
        "username",
        "is_active",
        "is_staff",
        "is_superuser",
        "last_login",
        "date_joined",
        "monaiot_connexion",
        "monaiot_signup",
    ]
    list_filter = ("is_staff", "is_superuser", "is_active", "user_type", "user_category", "date_joined")
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        (_("Personal info"), {"fields": ("email",)}),
        (
            _("Segmentation"),
            {
                "fields": (
                    "user_type",
                    "user_category",
                )
            },
        ),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                ),
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": ("username", "email", "password1", "password2"),
            },
        ),
    )
    resource_classes = [
        UserResource,
    ]
