# Personal Online Diary

A private diary website built with **Django**, **SQLite**, **HTML & CSS**.
Users can sign up, then add, view, edit and delete diary entries. A
**Naive Bayes** classifier predicts the mood (Happy, Sad, Angry, Anxious,
Calm) from the entry text.

## Features

- Sign up / log in / log out (each user only sees their own entries)
- Create, read, update and delete diary entries
- Search entries and filter by mood, with pagination
- Mood prediction with Naive Bayes
  - "Predict my mood" button on the entry form (AJAX)
  - Auto-detects the mood on save when you leave it on "Auto-detect"
  - Learns from the moods you pick yourself
- Mood dashboard with a breakdown of your entries
- Django admin panel at `/admin/`

## Run it in VS Code

1. Install Python 3.10+ and VS Code (with the *Python* extension).
2. Open this folder in VS Code (`File > Open Folder`), then open the terminal (`Ctrl+` `` ` ``).
3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   ```

4. Install Django and create the database:

   ```bash
   pip install -r requirements.txt
   python manage.py makemigrations diary
   python manage.py migrate
   ```

5. (Optional) create an admin user: `python manage.py createsuperuser`
6. Start the server and open http://127.0.0.1:8000/

   ```bash
   python manage.py runserver
   ```

7. Run the tests: `python manage.py test`

## Project structure

```
personal_diary/
├── manage.py
├── requirements.txt
├── diary_project/          # settings, root urls
└── diary/
    ├── models.py           # DiaryEntry model (SQLite)
    ├── views.py            # CRUD, dashboard, mood prediction endpoint
    ├── forms.py  urls.py  admin.py  tests.py
    ├── ml/
    │   ├── mood_model.py       # Naive Bayes implemented from scratch
    │   └── training_data.py    # seed sentences for each mood
    ├── templates/          # HTML pages
    └── static/diary/style.css
```

## How the Naive Bayes mood predictor works

1. Text is lower-cased, split into words, and common words are removed.
   "not happy" is kept as one token (`not_happy`) so negation is handled.
2. For each mood, the model counts how often each word appears in that
   mood's training sentences.
3. For new text it scores every mood with
   `log P(mood) + Σ log P(word | mood)` (Laplace smoothing), then converts
   the scores to probabilities and picks the highest.
4. Training data = 80 built-in seed sentences + every entry where **you**
   chose the mood yourself. Auto-predicted moods are not fed back in, so the
   model never trains on its own guesses.

To improve accuracy, add more sentences to `diary/ml/training_data.py`.
The model is plain Python, so scikit-learn is not required (you can swap in
`sklearn.naive_bayes.MultinomialNB` later if you want).

## Before deploying publicly

Change `SECRET_KEY`, set `DEBUG = False` and fill `ALLOWED_HOSTS` in
`diary_project/settings.py`.
