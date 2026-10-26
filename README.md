# Music Genre Classifier

Upload a song sample and the app predicts its genre.
**Machine Learning** (Random Forest) + **Deep Learning** (CNN on Mel-spectrograms) + **Cloud** (Streamlit Cloud hosting, AWS S3 storage).

```
music-genre-classifier/
├── data/                  <- put the Kaggle GTZAN files here (not uploaded to GitHub)
├── models/                <- trained models are saved here (commit music_genre_cnn.h5 + labels.json)
├── reports/               <- charts for your presentation (created by the training scripts)
├── .streamlit/secrets.toml.example
├── audio_utils.py         <- shared audio -> spectrogram code
├── ml_baseline.py         <- Part 1: Random Forest on features_30_sec.csv
├── dl_model.py            <- Part 2: CNN on Mel-spectrograms
├── app.py                 <- Part 3: Streamlit web app (+ optional S3 upload)
├── requirements.txt       <- app dependencies (used by Streamlit Cloud)
├── requirements-train.txt <- everything needed for training
└── packages.txt           <- system packages for Streamlit Cloud (ffmpeg)
```

## 1. Setup (Windows PowerShell; Python 3.10 - 3.12)

```powershell
cd music-genre-classifier
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-train.txt
```
(Mac/Linux: `source .venv/bin/activate`)

## 2. Get the dataset

Download "GTZAN Dataset - Music Genre Classification" from Kaggle (about 1.2 GB), unzip it, and arrange it like this:

```
data/features_30_sec.csv
data/genres_original/blues/*.wav   (and classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
```
You only need those two things; ignore the other folders in the download.

## 3. Run Part 1: ML baseline

```powershell
python ml_baseline.py
```
Prints accuracy of a Decision Tree and a Random Forest, saves `models/genre_rf.joblib` and two charts in `reports/`. Takes seconds.

## 4. Run Part 2: CNN

```powershell
python dl_model.py
```
- First run converts all songs to spectrograms (a few minutes) and caches them in `data/mel_cache.npz`; later runs are faster.
- Trains the CNN (10-30 min on CPU; stops early when it stops improving).
- Saves `models/music_genre_cnn.h5`, `models/labels.json`, and charts in `reports/` (training curves + confusion matrix).
- Prints clip-level and whole-song test accuracy. Quote the whole-song number in your presentation.
- Options: `--epochs 60`, `--batch-size 64`, `--rebuild` (redo the cache).
- Too slow? Run it on Google Colab with a free GPU, then download the `models/` folder.

## 5. Run the web app locally

```powershell
streamlit run app.py
```
Opens at http://localhost:8501. Upload a wav/mp3 and you get the spectrogram, the predicted genre and confidence bars.

## 6. Cloud storage (AWS S3, optional but recommended for the project)

1. AWS Console -> S3 -> Create bucket (keep "Block all public access" ON).
2. IAM -> create a user whose policy allows only `s3:PutObject` and `s3:GetObject` on that bucket -> create an access key.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. Never commit the real file.

Once the secrets exist, every upload is also saved to `uploads/` in the bucket. Without them the app works normally.

## 7. Deploy (Streamlit Community Cloud)

```powershell
git init
git add .
git commit -m "Music genre classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/music-genre-classifier.git
git push -u origin main
```
Then at share.streamlit.io: New app -> pick the repo, branch `main`, main file `app.py` -> Advanced settings: choose Python 3.11 or 3.12 and paste your `secrets.toml` contents into Secrets -> Deploy.

Make sure `models/music_genre_cnn.h5` and `models/labels.json` are committed (the CNN file is only a few MB).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot find data\genres_original` | The dataset folder is in the wrong place; see step 2 |
| "skipped jazz.00054.wav" | Normal: that GTZAN file is corrupted, the script skips it |
| App says "Model not found" | Run `python dl_model.py` first, and commit `models/` |
| Streamlit Cloud build fails on TensorFlow | Set Python to 3.11 or 3.12 in Advanced settings |
| mp3 will not load | Make sure `packages.txt` (ffmpeg) is in the repo root |
| Out of memory on Streamlit Cloud | Redeploy on Hugging Face Spaces or GCP Cloud Run |
