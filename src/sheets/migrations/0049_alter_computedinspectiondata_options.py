# Generated by Django 5.0.6 on 2024-07-04 13:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sheets', '0048_computedinspectiondata_pdf_rendering_end_and_more'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='computedinspectiondata',
            options={'ordering': ('-created',), 'verbose_name': 'Fiche établissement', 'verbose_name_plural': 'Fiches établissement'},
        ),
    ]