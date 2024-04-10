# Generated by Django 5.0.3 on 2024-03-27 10:43

import datetime

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sheets', '0041_computedinspectiondata_followed_with_pnttd_data_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='RegistryDownload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('org_id', models.CharField(max_length=20, verbose_name='Organization ID')),
                ('data_start_date', models.DateTimeField(default=datetime.datetime(2022, 1, 1, 0, 0), verbose_name='Data Start Date')),
                ('data_end_date', models.DateTimeField(default=django.utils.timezone.now, verbose_name='Data End Date')),
                ('created', models.DateTimeField(default=django.utils.timezone.now, verbose_name='Created')),
                ('created_by', models.EmailField(blank=True, max_length=254, verbose_name='Created by')),
            ],
            options={
                'verbose_name': 'Téléchargement de registre',
                'verbose_name_plural': 'Téléchargements de registre',
                'ordering': ('-created',),
            },
        ),
        migrations.AlterModelOptions(
            name='computedinspectiondata',
            options={'ordering': ('-created',), 'verbose_name': "Fiche d'inspection", 'verbose_name_plural': "Fiches d'inspection"},
        ),
    ]