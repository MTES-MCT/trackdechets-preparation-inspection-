# Generated by Django 4.1.7 on 2023-02-27 16:02

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0004_computedinspectiondata_state"),
    ]

    operations = [
        migrations.RenameField(
            model_name="computedinspectiondata",
            old_name="waste_origin_map",
            new_name="waste_origin_map_data",
        ),
    ]
