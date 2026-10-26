from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone


class DiaryEntry(models.Model):
    MOOD_CHOICES = [
        ("happy", "Happy"),
        ("sad", "Sad"),
        ("angry", "Angry"),
        ("anxious", "Anxious"),
        ("calm", "Calm"),
    ]
    MOOD_EMOJI = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "anxious": "😰",
        "calm": "😌",
    }

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="entries"
    )
    title = models.CharField(max_length=200)
    content = models.TextField()
    entry_date = models.DateField(default=timezone.localdate)
    # Final mood shown to the user. Either picked by the user, or predicted by
    # the Naive Bayes model (then mood_is_predicted is True).
    mood = models.CharField(max_length=10, choices=MOOD_CHOICES, blank=True)
    mood_is_predicted = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-entry_date", "-created_at"]
        verbose_name_plural = "diary entries"

    def __str__(self):
        return f"{self.title} ({self.entry_date})"

    def get_absolute_url(self):
        return reverse("entry_detail", args=[self.pk])

    @property
    def mood_emoji(self):
        return self.MOOD_EMOJI.get(self.mood, "")

    @property
    def text_for_model(self):
        return f"{self.title}. {self.content}"
