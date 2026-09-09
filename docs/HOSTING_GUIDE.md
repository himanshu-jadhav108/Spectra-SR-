# Spectra SR — Production Hosting & Deployment Guide

This guide details how to deploy **Spectra SR** to the cloud so anyone can access your interactive satellite super-resolution console on both desktop and mobile devices.

---

## 🏗️ Architecture Overview

Spectra SR is packaged as a **Unified Full-Stack Service**:
- **Backend**: FastAPI with Uvicorn, scikit-learn Trust Head, and SEN2SRLite adapter.
- **Frontend**: Responsive HTML5, Vanilla CSS, and JavaScript with interactive before/after split comparison sliders and ESAOpenSR Validation Lab.
- **Unified Delivery**: FastAPI automatically serves the frontend root (`/`), brand assets (`/brand`), and static stylesheets/scripts directly. This means **you only need to host one single service**—no complex CORS setups or separate frontend hosting required!

---

## 🚀 Deployment Options (Ranked by Ease)

### Option 1: Render.com (Recommended for Free / 1-Click Git Push)
Render is the easiest way to deploy Docker or Python web services directly from GitHub with automatic HTTPS.

1. **Sign Up**:
   Go to [Render.com](https://render.com) and create a free account (sign in with GitHub).
2. **Create New Web Service**:
   - Click **New +** → **Web Service**.
   - Select **Build and deploy from a Git repository**.
   - Connect your repository: `https://github.com/himanshu-jadhav108/Spectra-SR-`.
3. **Configure Settings**:
   - **Name**: `spectra-sr`
   - **Region**: Choose closest to you (e.g., Singapore, Frankfurt, Oregon).
   - **Branch**: `main`
   - **Environment**: **Docker** (Render will automatically detect the `Dockerfile`).
   - **Instance Type**: Free or Starter (512 MB – 2 GB RAM).
4. **Environment Variables**:
   Add the following under **Advanced → Environment Variables**:
   ```ini
   APP_ENV=production
   DATA_SOURCE=local
   DEVICE=cpu
   ```
5. **Deploy**:
   - Click **Create Web Service**.
   - Render will build the Docker container and provide a live HTTPS URL (e.g., `https://spectra-sr.onrender.com`).
   - Any future `git push origin main` will automatically rebuild and deploy!

---

### Option 2: Hugging Face Spaces (100% Free AI Showcase with GPU Options)
Hugging Face Spaces is popular for academic, SIH, and research demos.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. **Space Name**: `spectra-sr`
3. **License**: `mit` or `cc-by-4.0`
4. **Space SDK**: Select **Docker** → **Blank**.
5. **Clone and Push**:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/spectra-sr
   git push hf main
   ```
6. In `Dockerfile`, Hugging Face automatically routes to port `7860`. Our `Dockerfile` automatically detects `${PORT:-8000}`, so it works out-of-the-box!
7. Your app is live at `https://huggingface.co/spaces/YOUR_USERNAME/spectra-sr`.

---

### Option 3: Railway.app (Instant PaaS)
Railway offers high performance with generous trial credits and fast container builds.

1. Go to [Railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select `Spectra-SR-`.
4. Railway automatically detects `Dockerfile`, builds the container, and assigns a public domain under **Settings → Networking → Generate Domain**.

---

### Option 4: Cloud VPS (DigitalOcean Droplet, AWS EC2, or Hetzner)
For full control, custom domain names (`spectra.yourdomain.com`), and higher concurrency.

#### Step 1: Install Docker & Docker Compose on your server
```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose git
sudo systemctl enable --now docker
```

#### Step 2: Clone the repository and start
```bash
git clone https://github.com/himanshu-jadhav108/Spectra-SR-.git
cd Spectra-SR-

# Launch in background with Docker Compose
sudo docker-compose up -d --build
```
Your app will be running on `http://YOUR_SERVER_IP:8000`.

#### Step 3: Set up Nginx & SSL (Let's Encrypt)
```nginx
# /etc/nginx/sites-available/spectra-sr
server {
    server_name spectra.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Run `sudo certbot --nginx -d spectra.yourdomain.com` for instant free SSL.

---

## ⚙️ Production Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port for Uvicorn to listen on (automatically set by Render/Railway/HF). |
| `APP_ENV` | `production` | Environment mode (`development` or `production`). |
| `DATA_SOURCE` | `local` | `local` for 100% offline cached scenes; `copernicus_cdse` for live API queries. |
| `DEVICE` | `cpu` | Inference device (`cpu` or `cuda`). Defaults to `cpu` for standard cloud instances. |
| `CDSE_CLIENT_ID` | `""` | *(Optional)* Copernicus Data Space client ID for live Sentinel-2 queries. |
| `CDSE_CLIENT_SECRET` | `""` | *(Optional)* Copernicus Data Space secret key. |

---

## 📱 Mobile Responsiveness Verification

Before sharing your public link, test that the mobile enhancements are active:
1. Open your hosted URL on any phone browser (Safari, Chrome, Firefox).
2. **Navigation**: Swipe horizontally across the top tabs (**Mission Console**, **Pipeline Monitor**, **Analysis Workspace**, **Validation Lab**).
3. **Split Comparison Slider**: Touch and drag the handle (`◀ ▶`) horizontally. The touch events are hardware-accelerated with `touch-action: none` and `PointerCapture`.
4. **Validation Lab**: Switch between **3-Way Multi-Pane** and **SR vs HR Split Slider** to inspect high-resolution ground truth directly on mobile.
