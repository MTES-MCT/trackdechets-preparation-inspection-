# Generated by Django 4.2.6 on 2023-11-24 10:16

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0031_computedinspectiondata_icpe_2760_graph_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="bsda_worker_stats_data",
            field=models.JSONField(default=dict),
        ),
    ]
