# Feature Explorer — Streamlit Cloud deployment

A self-contained copy of the `feature_explorer.py` Streamlit app, packaged so it can be
hosted **free** on [Streamlit Community Cloud](https://share.streamlit.io). It contains
only what the app needs — none of your trading credentials, broker config, risk limits,
or trade logs.

## What's in here

```
feature-explorer-deploy/
├── app/feature_explorer.py            # the Streamlit app (entry point)
├── src/                               # only the 5 modules the app imports
│   ├── data/     fetch_alpaca_stock_bars.py, fetch_yf_daily.py
│   └── research/ feature_panel.py, lgbm_dash_model.py
├── config/universe.yaml               # equity universe list
├── data/raw/daily/*.parquet           # 146 daily-bar files (~34 MB) the explorer reads
├── models/lgbm_dash/QQQ.pkl, SPY.pkl  # pre-trained LightGBM models (~64 MB)
├── requirements.txt                   # exact packages for the host to install
├── runtime.txt                        # pins Python 3.11
├── .streamlit/config.toml             # headless server config
└── .streamlit/secrets.toml.example    # template for the Alpaca keys (optional)
```

Total size ~98 MB. Largest single file is 31 MB, well under GitHub's 100 MB limit.

---

## Deploy in 6 steps

### 1. Create a new GitHub repo
On github.com click **New repository**. Name it e.g. `feature-explorer`.
It can be **private** — Streamlit Cloud deploys from private repos on the free tier.
Don't add a README/gitignore (this folder already has them).

### 2. Push this folder to it
Open a terminal **inside this folder** and run:

```bash
git init
git add .
git commit -m "Feature Explorer deploy bundle"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/feature-explorer.git
git push -u origin main
```

(The parquet/pkl files are all under 100 MB, so no Git LFS is needed.)

### 3. Go to Streamlit Community Cloud
Open https://share.streamlit.io and sign in **with your GitHub account**.
Authorize it to see your repos (including private ones).

### 4. Create the app
Click **Create app -> Deploy from GitHub** and set:

- **Repository:** `<YOUR_USERNAME>/feature-explorer`
- **Branch:** `main`
- **Main file path:** `app/feature_explorer.py`

Click **Deploy**. First build takes a few minutes while it installs `requirements.txt`
(lightgbm / scipy / scikit-learn are the slow ones).

### 5. (Optional) Add your Alpaca keys for the "Refresh data" button
The explorer, scatter, correlation and LightGBM tabs all work **without any keys**.
Only the **Refresh data (Alpaca)** button needs them. To enable it:

In the app page, click **... -> Settings -> Secrets**, and paste:

```toml
paper_alpaca_key = "YOUR_ALPACA_PAPER_KEY"
paper_alpaca_secret = "YOUR_ALPACA_PAPER_SECRET"
```

Save. Streamlit loads these into `os.environ`, which is exactly what the app reads.
**Use your paper keys only** — never live keys (matches your project rule).

### 6. Open the URL
Streamlit gives you a public `https://<app-name>.streamlit.app` link. That's your hosted app.

---

## Notes & limits

- **Free tier sleeps** after inactivity; first visit after a nap takes ~30 s to wake.
  ~1 GB RAM, which is enough for this app.
- **Retrain / refresh writes are temporary.** Streamlit Cloud's filesystem resets on
  reboot, so retrained models and freshly fetched bars last only until the app restarts.
  The committed parquet + pkl files are always the baseline.
- **To update the app**, just `git push` — Streamlit redeploys automatically.
- **Why not Vercel:** Vercel is serverless (short-lived functions, no persistent
  websocket), which is what Streamlit needs. Streamlit Cloud is purpose-built and free.

## Run it locally first (optional)

```bash
pip install -r requirements.txt
streamlit run app/feature_explorer.py
```

---

## Automated daily refresh + retrain (GitHub Actions)

The hosted Streamlit app is a **read-only viewer** — its disk is wiped on every
reboot, so refreshing/retraining *in the app* does not persist (and on the app's
Python build the retrain comes back empty). **Do not use the in-app
"Refresh data / Auto-retrain" controls on the hosted app.**

Instead, `.github/workflows/daily-update.yml` runs `scripts/daily_update.py`
every weekday at **23:00 UTC** (after the US close) on a clean Python 3.11 runner:

1. Re-fetches all daily bars + macro/vol sources (VIX, VIX3M, VVIX, MOVE, TNX,
   IRX, TLT, HYG, LQD) from yfinance.
2. Retrains every model in `models/lgbm_dash/` (QQQ, SPY).
3. Commits the updated `data/raw/daily/*.parquet` and `models/lgbm_dash/*.pkl`
   back to the repo — which automatically triggers a Streamlit Cloud redeploy.

If a retrain produces **0 horizons**, the job fails on purpose and commits
nothing, so a bad run can never overwrite your good models.

### One-time setup
1. Push these files (`scripts/`, `.github/`) to the repo (see commands below).
2. On GitHub: **Settings -> Actions -> General -> Workflow permissions ->**
   select **"Read and write permissions"** -> Save. (Lets the job push commits.)
3. **Actions** tab -> **Daily data refresh + retrain** -> **Run workflow** to
   test it immediately instead of waiting for the cron.

### Run it manually anytime
Actions tab -> select the workflow -> **Run workflow**. Takes ~15-25 min.

### Note on repo size
Each run commits ~98 MB of updated binaries, so git history grows over time.
That's fine for months; if the repo gets large later, you can squash history or
re-init the repo. Ask me and I'll set up a slimmer scheme (e.g. force-pushing a
single data commit) if you want to avoid the growth entirely.

---

## The in-app "Refresh + retrain permanently" button

Because Streamlit Cloud's disk is ephemeral, the app can't make a refresh stick on
its own. The sidebar button **"♻️ Refresh + retrain permanently (GitHub)"** instead
triggers the `daily-update.yml` workflow on GitHub, which fetches + retrains on
Python 3.11 and commits — so the result is permanent and the app auto-redeploys.

To enable it, add a GitHub token to the app's Secrets:

1. GitHub -> your avatar -> **Settings -> Developer settings -> Personal access
   tokens -> Fine-grained tokens -> Generate new token**.
   - **Repository access:** only `redxia/feature-explorer`.
   - **Permissions -> Repository -> Actions:** **Read and write**.
   - (Contents can stay read-only; the workflow commits with its own token.)
   - Generate and copy the token (starts with `github_pat_`).
2. In the app: **Manage app -> Settings -> Secrets**, paste:
   ```toml
   GH_TOKEN = "github_pat_...."
   GH_REPO  = "redxia/feature-explorer"
   ```
   Save. The app reboots.
3. Also set repo **Settings -> Actions -> General -> Workflow permissions ->
   "Read and write permissions"** so the workflow can push its commit.

Now clicking the button (or the daily 23:00 UTC schedule) rebuilds data + models
permanently. Each run takes ~15-25 min; the app updates automatically when done.
