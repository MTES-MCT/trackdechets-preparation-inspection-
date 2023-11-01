# Generated by Django 4.2.2 on 2023-09-25 10:12

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0029_computedinspectiondata_linked_companies_data"),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="company_created_at",
            field=models.DateTimeField(default=django.utils.timezone.now, verbose_name="Company created at"),
        ),
    ]
