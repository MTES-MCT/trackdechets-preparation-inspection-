# Generated by Django 5.0.1 on 2024-01-19 08:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("sheets", "0040_remove_computedinspectiondata_outliers_data"),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="followed_with_pnttd_data",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="gistrid_stats_data",
            field=models.JSONField(default=dict),
        ),
    ]
