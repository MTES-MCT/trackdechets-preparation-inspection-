from django import template
from django.contrib.sites.models import Site
from django.urls import reverse_lazy

register = template.Library()


@register.simple_tag(takes_context=True)
def second_factor_url(context):
    current_site = Site.objects.get_current()

    domain = current_site.domain
    return f"https://{domain}{reverse_lazy('second_factor')}"
