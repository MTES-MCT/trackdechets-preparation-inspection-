# Generated by Django 5.0.4 on 2024-05-24 12:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sheets', '0042_registrydownload_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='computedinspectiondata',
            name='non_dangerous_waste_quantities_graph',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='computedinspectiondata',
            name='non_dangerous_waste_quantities_graph_data',
            field=models.JSONField(default=dict),
        ),
    ]