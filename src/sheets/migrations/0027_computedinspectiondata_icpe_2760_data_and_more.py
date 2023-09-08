# Generated by Django 4.2.2 on 2023-09-05 12:48

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("sheets", "0026_remove_computedinspectiondata_icpe_2760_data_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2760_data",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2770_data",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="computedinspectiondata",
            name="icpe_2790_data",
            field=models.JSONField(default=dict),
        ),
    ]