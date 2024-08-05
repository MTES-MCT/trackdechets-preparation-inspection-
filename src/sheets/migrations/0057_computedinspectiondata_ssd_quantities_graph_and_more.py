# Generated by Django 5.0.6 on 2024-08-05 13:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sheets', '0056_computedinspectiondata_company_collector_profiles_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='computedinspectiondata',
            name='ssd_quantities_graph',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='computedinspectiondata',
            name='ssd_quantities_graph_data',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='computedinspectiondata',
            name='ssd_statements_graph',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='computedinspectiondata',
            name='ssd_statements_graph_data',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='computedinspectiondata',
            name='ssd_stats_data',
            field=models.JSONField(default=dict),
        ),
    ]
