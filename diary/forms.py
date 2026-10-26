from django import forms
from django.contrib.auth.forms import UserCreationForm

from .models import DiaryEntry


class EntryForm(forms.ModelForm):
    class Meta:
        model = DiaryEntry
        fields = ["title", "entry_date", "content", "mood"]
        widgets = {
            "title": forms.TextInput(attrs={"placeholder": "Give today a title"}),
            "entry_date": forms.DateInput(attrs={"type": "date"}, format="%Y-%m-%d"),
            "content": forms.Textarea(
                attrs={"rows": 12, "placeholder": "Dear diary, today I felt..."}
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["mood"].required = False
        self.fields["mood"].label = "Mood"
        self.fields["mood"].choices = [
            ("", "Auto-detect from my text")
        ] + list(DiaryEntry.MOOD_CHOICES)
        # If the saved mood was only a prediction, show "Auto-detect" again so
        # editing the text re-runs the prediction instead of freezing it.
        if self.instance.pk and self.instance.mood_is_predicted:
            self.initial["mood"] = ""


class SignUpForm(UserCreationForm):
    pass
