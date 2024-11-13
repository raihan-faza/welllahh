# Generated by Django 5.1 on 2024-11-12 13:29

import django.db.models.deletion
import django.utils.timezone
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="CustomUser",
            fields=[
                (
                    "customer_id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("age", models.IntegerField(blank=True, null=True)),
                ("secret_phrase", models.CharField(max_length=100)),
                (
                    "status",
                    models.CharField(
                        choices=[("any", "anonymous"), ("knw", "known")], max_length=9
                    ),
                ),
                (
                    "parent",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="ChatSession",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("session_title", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "message_from",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="BloodCodition",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("blood_sugar", models.DecimalField(decimal_places=2, max_digits=10)),
                ("uric_acid", models.DecimalField(decimal_places=2, max_digits=10)),
                ("cholesterol", models.DecimalField(decimal_places=2, max_digits=10)),
                (
                    "blood_pressure",
                    models.DecimalField(decimal_places=2, max_digits=10),
                ),
                ("check_time", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Message",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("prompt_content", models.TextField()),
                ("chatbot_content", models.TextField()),
                ("context", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "chat_session",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.chatsession",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="NutritionProgress",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("check_time", models.DateTimeField(default=django.utils.timezone.now)),
                ("nutrition_name", models.CharField(default=None, max_length=255)),
                ("calorie", models.DecimalField(decimal_places=2, max_digits=10)),
                ("carbs", models.DecimalField(decimal_places=2, max_digits=10)),
                ("protein", models.DecimalField(decimal_places=2, max_digits=10)),
                ("fat", models.DecimalField(decimal_places=2, max_digits=10)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="RiwayatPenyakit",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("nama_penyakit", models.CharField(max_length=255)),
                ("deskripsi_penyakit", models.CharField(max_length=255)),
                ("check_time", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="TargetPlan",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("target_calorie", models.IntegerField()),
                ("target_carbs", models.IntegerField()),
                ("target_protein", models.IntegerField()),
                ("target_fat", models.IntegerField()),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Token",
            fields=[
                (
                    "token_id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("token", models.CharField(max_length=255)),
                (
                    "custom_user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="UserBodyInfo",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("weight", models.IntegerField()),
                ("muscle_mass", models.IntegerField(blank=True, null=True)),
                ("height", models.IntegerField()),
                ("fat_mass", models.IntegerField(blank=True, null=True)),
                ("check_time", models.DateField()),
                (
                    "custom_user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="my_app.customuser",
                    ),
                ),
            ],
        ),
    ]
