# Deploying to Railway (Native Python / Procfile)

This guide walks you through deploying the **Fire & Human Detection** Flask app to [Railway](https://railway.com) using native Python.

---

## 1. Push to GitHub

Open a terminal (PowerShell or Git Bash) inside the project folder:

```bash
cd "Ai-human-detection-in-fire-main"
```

Initialize git and make your first commit:

```bash
git init
git add .
git commit -m "Fresh deployment setup for Railway"
```

Create a new repository on [GitHub](https://github.com/new) (name it e.g. `fire-human-detection`), and link it:

```bash
git branch -M main
git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-REPO-NAME>.git
git push -u origin main
```

> **Note on Model Weights:**
> - `fire_detection_model.h5` (~7.4 MB) is included directly in the repo.
> - `yolov3.weights` (~248 MB) exceeds GitHub's 100 MB limit, so it is ignored by `.gitignore`.
> - When Railway starts your app, `app.py` **automatically downloads and verifies** `yolov3.weights` from the official Darknet mirror. You do not need Git LFS.

---

## 2. Deploy on Railway

1. Go to [railway.com](https://railway.com) and log in with GitHub.
2. Click **"+ New Project"**.
3. Select **"Deploy from GitHub repo"**.
4. Choose your `fire-human-detection` repository.
5. Railway will automatically detect **Nixpacks** and the `Procfile`:
   - Build phase: Installs dependencies from `requirements.txt` with Python 3.10 and `ffmpeg`/`libGL`.
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300`
6. Click **Deploy**.

---

## 3. Generate a Public Domain

Once deployed:
1. In your Railway project, click on your service.
2. Go to the **Settings** tab.
3. Under **Networking**, click **"Generate Domain"**.
4. Open the generated URL (e.g. `https://fire-human-detection-production.up.railway.app`) in your browser to test the app!

---

## Configuration Files Included

- **`Procfile`**: Tells Railway how to run the app with Gunicorn.
- **`nixpacks.toml`**: Configures Nixpacks to include Python 3.10, ffmpeg, and libGL.
- **`runtime.txt`**: Pins the Python runtime version (`python-3.10.14`).
- **`railway.json`**: Railway platform settings.
- **`requirements.txt`**: Optimized dependencies (`tensorflow-cpu`, `opencv-python-headless`, etc.).

