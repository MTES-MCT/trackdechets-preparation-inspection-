# Generated by Django 4.1.7 on 2023-02-28 15:15

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "sheets",
            "0005_rename_waste_origin_map_computedinspectiondata_waste_origin_map_data",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="pdf",
            field=models.ImageField(blank=True, upload_to="", verbose_name="Pdf"),
        ),
    ]
