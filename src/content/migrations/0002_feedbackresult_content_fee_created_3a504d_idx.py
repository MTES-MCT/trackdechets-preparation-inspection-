# Generated by Django 4.1.7 on 2023-05-11 09:07

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("content", "0001_initial"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="feedbackresult",
            index=models.Index(
                fields=["created", "author"], name="content_fee_created_3a504d_idx"
            ),
        ),
    ]