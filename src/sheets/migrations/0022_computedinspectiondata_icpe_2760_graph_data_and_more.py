# Generated by Django 4.2.2 on 2023-08-10 15:31

import datetime

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0021_computedinspectiondata_icpe_2770_graph_data_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2760_graph_data",
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name="computedinspectiondata",
            name="data_start_date",
            field=models.DateTimeField(
                default=datetime.datetime(2022, 8, 10, 15, 31, 40, 561589, tzinfo=datetime.timezone.utc),
                verbose_name="Data Start Date",
            ),
        ),
    ]
