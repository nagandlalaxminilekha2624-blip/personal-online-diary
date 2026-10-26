from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import EntryForm, SignUpForm
from .ml.mood_model import build_classifier
from .models import DiaryEntry


# ---------------------------------------------------------------- ML helpers
def _classifier_for(user):
    """Seed data + this user's own mood-labelled entries (not auto-predicted)."""
    own = DiaryEntry.objects.filter(user=user, mood_is_predicted=False).exclude(mood="")
    texts = [e.text_for_model for e in own]
    labels = [e.mood for e in own]
    return build_classifier(texts, labels)


def _apply_mood(entry):
    """Fill in the mood with a prediction when the user left it on auto-detect."""
    if entry.mood:
        entry.mood_is_predicted = False
        return
    prediction = _classifier_for(entry.user).predict(entry.text_for_model)
    if prediction:
        entry.mood = prediction.label
        entry.mood_is_predicted = True
    else:
        entry.mood_is_predicted = False


# ------------------------------------------------------------------ accounts
def signup(request):
    if request.user.is_authenticated:
        return redirect("entry_list")
    form = SignUpForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Welcome! Your diary is ready.")
        return redirect("entry_list")
    return render(request, "registration/signup.html", {"form": form})


# ------------------------------------------------------------------- entries
@login_required
def entry_list(request):
    entries = DiaryEntry.objects.filter(user=request.user)
    query = request.GET.get("q", "").strip()
    mood = request.GET.get("mood", "").strip()
    if query:
        entries = entries.filter(Q(title__icontains=query) | Q(content__icontains=query))
    if mood in dict(DiaryEntry.MOOD_CHOICES):
        entries = entries.filter(mood=mood)

    page = Paginator(entries, 6).get_page(request.GET.get("page"))
    params = request.GET.copy()
    params.pop("page", None)
    moods = [
        {"key": k, "label": v, "emoji": DiaryEntry.MOOD_EMOJI[k]}
        for k, v in DiaryEntry.MOOD_CHOICES
    ]
    return render(request, "diary/entry_list.html", {
        "page": page,
        "query": query,
        "selected_mood": mood,
        "moods": moods,
        "querystring": params.urlencode(),
    })


@login_required
def entry_detail(request, pk):
    entry = get_object_or_404(DiaryEntry, pk=pk, user=request.user)
    return render(request, "diary/entry_detail.html", {"entry": entry})


@login_required
def entry_create(request):
    form = EntryForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        entry = form.save(commit=False)
        entry.user = request.user
        _apply_mood(entry)
        entry.save()
        messages.success(request, "Entry saved.")
        return redirect(entry)
    return render(request, "diary/entry_form.html", {"form": form, "is_new": True})


@login_required
def entry_edit(request, pk):
    entry = get_object_or_404(DiaryEntry, pk=pk, user=request.user)
    form = EntryForm(request.POST or None, instance=entry)
    if request.method == "POST" and form.is_valid():
        entry = form.save(commit=False)
        _apply_mood(entry)
        entry.save()
        messages.success(request, "Entry updated.")
        return redirect(entry)
    return render(request, "diary/entry_form.html", {"form": form, "entry": entry, "is_new": False})


@login_required
def entry_delete(request, pk):
    entry = get_object_or_404(DiaryEntry, pk=pk, user=request.user)
    if request.method == "POST":
        entry.delete()
        messages.success(request, "Entry deleted.")
        return redirect("entry_list")
    return render(request, "diary/entry_confirm_delete.html", {"entry": entry})


# ----------------------------------------------------------------- dashboard
@login_required
def dashboard(request):
    entries = DiaryEntry.objects.filter(user=request.user)
    total = entries.count()
    counts = {row["mood"]: row["n"] for row in entries.values("mood").annotate(n=Count("id"))}
    mood_stats = []
    for key, label in DiaryEntry.MOOD_CHOICES:
        n = counts.get(key, 0)
        mood_stats.append({
            "key": key,
            "label": label,
            "emoji": DiaryEntry.MOOD_EMOJI[key],
            "count": n,
            "percent": round(n * 100 / total) if total else 0,
        })
    today = timezone.localdate()
    this_month = entries.filter(entry_date__year=today.year, entry_date__month=today.month).count()
    trained_on = entries.filter(mood_is_predicted=False).exclude(mood="").count()
    return render(request, "diary/dashboard.html", {
        "total": total,
        "this_month": this_month,
        "mood_stats": mood_stats,
        "trained_on": trained_on,
        "latest": entries.first(),
    })


# ---------------------------------------------------------------- prediction
@login_required
@require_POST
def predict_mood(request):
    """JSON endpoint used by the 'Predict mood' button on the entry form."""
    text = request.POST.get("text", "").strip()
    prediction = _classifier_for(request.user).predict(text) if text else None
    if not prediction:
        return JsonResponse({"mood": None})
    return JsonResponse({
        "mood": prediction.label,
        "label": dict(DiaryEntry.MOOD_CHOICES)[prediction.label],
        "emoji": DiaryEntry.MOOD_EMOJI[prediction.label],
        "confidence": round(prediction.confidence * 100),
    })
