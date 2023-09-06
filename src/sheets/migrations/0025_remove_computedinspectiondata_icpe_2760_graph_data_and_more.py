# Generated by Django 4.2.2 on 2023-08-16 14:17

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0024_computedinspectiondata_icpe_2790_graph_data"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="computedinspectiondata",
            name="icpe_2760_graph_data",
        ),
        migrations.RemoveField(
            model_name="computedinspectiondata",
            name="icpe_2770_graph_data",
        ),
        migrations.RemoveField(
            model_name="computedinspectiondata",
            name="icpe_2790_graph_data",
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2760_data",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2760_graph",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2770_data",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2770_graph",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2790_data",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2790_graph",
            field=models.TextField(blank=True),
        ),
    ]
