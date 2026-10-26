from django.contrib import admin

from .models import DiaryEntry


@admin.register(DiaryEntry)
class DiaryEntryAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "entry_date", "mood", "mood_is_predicted")
    list_filter = ("mood", "mood_is_predicted", "entry_date")
    search_fields = ("title", "content", "user__username")
