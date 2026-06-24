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
