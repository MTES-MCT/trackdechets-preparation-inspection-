# Generated by Django 5.0.4 on 2024-04-17 09:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_user_monaiot_connexion_user_monaiot_signup'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='monaiot_connexion',
            field=models.BooleanField(default=False, help_text='Did this user already log in with MonAIOT ?', verbose_name='MonAIOT connexion'),
        ),
        migrations.AlterField(
            model_name='user',
            name='monaiot_signup',
            field=models.BooleanField(default=False, help_text='Did this user sign up in with MonAIOT ?', verbose_name='MonAIOT inscription'),
        ),
    ]