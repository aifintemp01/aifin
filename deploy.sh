#!/bin/bash
# =============================================================
# AIFIN — DigitalOcean Ubuntu Setup Script
# Run as root on a fresh droplet:
#   bash deploy.sh
# =============================================================
set -e

REPO_URL="https://github.com/aifintemp01/aifin.git"
APP_DIR="/opt/ai-hedge-fund"
SERVER_IP=$(curl -s ifconfig.me)

echo "========================================"
echo " AIFIN — Server Setup"
echo "========================================"

# ── 1. System update ────────────────────────────────────────
echo "[1/8] Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq

# ── 2. Install Docker ───────────────────────────────────────
echo "[2/8] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
fi
systemctl enable docker
systemctl start docker

# ── 3. Install Docker Compose plugin ────────────────────────
echo "[3/8] Installing Docker Compose..."
apt-get install -y -qq docker-compose-plugin
docker compose version

# ── 4. Install nginx + certbot ──────────────────────────────
echo "[4/8] Installing nginx and certbot..."
apt-get install -y -qq nginx certbot python3-certbot-nginx git
systemctl enable nginx
systemctl start nginx

# ── 5. Clone repository ─────────────────────────────────────
echo "[5/8] Cloning repository..."
mkdir -p /opt
cd /opt

if [ -d "$APP_DIR" ]; then
    echo "  Repo already exists — pulling latest..."
    cd "$APP_DIR"
    git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 6. Create .env file ─────────────────────────────────────
echo "[6/8] Setting up environment..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo ""
    echo "  ⚠️  .env file created from template."
    echo "  Edit it now before continuing:"
    echo ""
    echo "    nano $APP_DIR/.env"
    echo ""
    echo "  Fill in your LLM API keys, TWELVE_DATA_API_KEY, NEWSDATA_API_KEY,"
    echo "  and CORS_ORIGINS=http://$SERVER_IP"
    echo ""
    read -p "  Press Enter when .env is ready..."
else
    echo "  .env already exists — skipping."
    echo "  Reminder: make sure CORS_ORIGINS in .env includes this server's IP."
fi

# ── 7. Build frontend and start backend ─────────────────────
echo "[7/8] Building frontend..."
cd "$APP_DIR"
docker build --target frontend-builder \
    --build-arg VITE_API_URL="http://$SERVER_IP" \
    -t aifin-frontend-builder .

# Extract the built dist/ folder out of the throwaway build container
CONTAINER_ID=$(docker create aifin-frontend-builder)
rm -rf "$APP_DIR/app/frontend/dist"
docker cp "$CONTAINER_ID:/app/app/frontend/dist" "$APP_DIR/app/frontend/dist"
docker rm "$CONTAINER_ID" > /dev/null

echo "  Frontend built at $APP_DIR/app/frontend/dist"

echo "Building and starting backend container..."
docker compose up -d --build

# ── 8. nginx config ──────────────────────────────────────────
echo "[8/8] Setting up nginx reverse proxy..."
cp "$APP_DIR/nginx.conf" /etc/nginx/sites-available/aifin
ln -sf /etc/nginx/sites-available/aifin /etc/nginx/sites-enabled/aifin
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# ── Done ────────────────────────────────────────────────────
IP=$SERVER_IP
echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo " App running at:  http://$IP"
echo " Health check:    http://$IP/health"
echo " API docs:        http://$IP/docs"
echo ""
echo " Next steps:"
echo "   1. Confirm CORS_ORIGINS in .env is set to http://$IP, then:"
echo "        cd $APP_DIR && docker compose restart"
echo "   2. Once you have a domain pointed at $IP, run:"
echo "        certbot --nginx -d your-domain.com"
echo ""