# Generated by Django 4.1.7 on 2023-03-23 09:30

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0009_computedinspectiondata_traceability_interruptions_data"),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="created_by",
            field=models.EmailField(
                blank=True, max_length=254, verbose_name="Created by"
            ),
        ),
    ]