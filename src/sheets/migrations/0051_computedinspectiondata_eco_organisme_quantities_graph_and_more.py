# Generated by Django 5.0.6 on 2024-07-01 12:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sheets', '0050_computedinspectiondata_eco_organisme_bordereaux_graph_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='computedinspectiondata',
            name='eco_organisme_quantities_graph',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='computedinspectiondata',
            name='eco_organisme_quantities_graph_data',
            field=models.JSONField(default=dict),
        ),
    ]