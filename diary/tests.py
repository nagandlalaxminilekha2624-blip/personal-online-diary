from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .ml.mood_model import build_classifier
from .models import DiaryEntry


class MoodModelTests(TestCase):
    def test_predicts_obvious_moods(self):
        clf = build_classifier()
        cases = {
            "I am so happy and excited, the party was amazing": "happy",
            "I cried all night, I feel lonely and heartbroken": "sad",
            "He lied and I am furious, I yelled at him": "angry",
            "I am nervous and worried, cannot stop overthinking": "anxious",
            "A peaceful quiet evening, I felt relaxed with tea": "calm",
        }
        for text, expected in cases.items():
            self.assertEqual(clf.predict(text).label, expected, text)

    def test_unknown_text_returns_none(self):
        self.assertIsNone(build_classifier().predict("xyzzy plugh"))


class EntryViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("asha", password="pass12345!")
        self.other = User.objects.create_user("ravi", password="pass12345!")
        self.client.login(username="asha", password="pass12345!")

    def test_login_required(self):
        self.client.logout()
        response = self.client.get(reverse("entry_list"))
        self.assertEqual(response.status_code, 302)

    def test_create_with_auto_mood(self):
        response = self.client.post(reverse("entry_create"), {
            "title": "Great day",
            "entry_date": "2026-10-26",
            "content": "I am so happy and grateful, we celebrated with friends",
            "mood": "",
        })
        self.assertEqual(response.status_code, 302)
        entry = DiaryEntry.objects.get()
        self.assertEqual(entry.user, self.user)
        self.assertEqual(entry.mood, "happy")
        self.assertTrue(entry.mood_is_predicted)

    def test_manual_mood_is_kept(self):
        self.client.post(reverse("entry_create"), {
            "title": "Odd day", "entry_date": "2026-10-26",
            "content": "I am so happy", "mood": "calm",
        })
        entry = DiaryEntry.objects.get()
        self.assertEqual(entry.mood, "calm")
        self.assertFalse(entry.mood_is_predicted)

    def test_edit_and_delete(self):
        entry = DiaryEntry.objects.create(
            user=self.user, title="T", content="plain text", mood="calm")
        self.client.post(reverse("entry_edit", args=[entry.pk]), {
            "title": "New title", "entry_date": "2026-10-26",
            "content": "changed", "mood": "sad",
        })
        entry.refresh_from_db()
        self.assertEqual(entry.title, "New title")
        self.client.post(reverse("entry_delete", args=[entry.pk]))
        self.assertFalse(DiaryEntry.objects.filter(pk=entry.pk).exists())

    def test_cannot_access_other_users_entry(self):
        entry = DiaryEntry.objects.create(user=self.other, title="Secret", content="x")
        for name in ("entry_detail", "entry_edit", "entry_delete"):
            response = self.client.get(reverse(name, args=[entry.pk]))
            self.assertEqual(response.status_code, 404)

    def test_predict_endpoint(self):
        response = self.client.post(reverse("predict_mood"), {"text": "I am furious and mad"})
        self.assertEqual(response.json()["mood"], "angry")

    def test_dashboard_and_list_render(self):
        DiaryEntry.objects.create(user=self.user, title="A", content="b", mood="happy")
        self.assertEqual(self.client.get(reverse("dashboard")).status_code, 200)
        self.assertEqual(self.client.get(reverse("entry_list") + "?q=A&mood=happy").status_code, 200)
