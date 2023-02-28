from django.core.management.base import BaseCommand

from ...models import ComputedInspectionData
from ...task import render_pdf


class Command(BaseCommand):
    def handle(self, verbosity=0, **kwargs):
        obj = ComputedInspectionData.objects.first()
        render_pdf(obj.pk)
