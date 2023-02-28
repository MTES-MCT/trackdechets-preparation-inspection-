# Generated by Django 4.1.7 on 2023-02-27 15:06

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0003_computedinspectiondata_bsda_created_rectified_graph_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="state",
            field=models.CharField(
                choices=[
                    ("INITIAL", "Initial"),
                    ("COMPUTED", "Computed"),
                    ("GRAPH_RENDERED", "Graph rendered"),
                ],
                default="INITIAL",
                max_length=20,
                verbose_name="State",
            ),
        ),
    ]
