#!/usr/bin/env bash
# Full setup for Tinker Board 2 (pip-only) + VS Code auto-install
# Handles held packages gracefully

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else echo "Please run as root or install sudo."; exit 1; fi
fi

echo "[1/12] Update & upgrade system (respect held packages)"
$SUDO apt-get update -y
HELD=$($SUDO apt-mark showhold || true)
if [ -n "$HELD" ]; then
  echo "⚠️ Held packages detected (won't change them):"
  echo "$HELD"
fi
if ! $SUDO apt-get upgrade -y --allow-change-held-packages; then
  echo "Standard upgrade failed; attempting fix-broken + dist-upgrade"
  $SUDO apt --fix-broken install -y || true
  $SUDO apt-get -o Dpkg::Options::="--force-confnew" dist-upgrade -y --allow-change-held-packages
fi

echo "[2/12] Install Python 3, Tk, pip, dev headers, tools"
$SUDO apt install -y --allow-change-held-packages python3 python3-tk python3-pip python3-dev python3-venv   build-essential cmake git pkg-config libjpeg-dev libtiff5-dev libpng-dev   libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev   libgtk2.0-dev libatlas-base-dev gfortran libhdf5-dev libssl-dev zlib1g-dev libbz2-dev   libreadline-dev libsqlite3-dev libncurses5-dev libffi-dev liblzma-dev   wget curl ca-certificates firefox-esr net-tools iptables-persistent filezilla

echo "[3/12] Download and install Visual Studio Code (ARM64)"
if ! command -v code >/dev/null 2>&1; then
  cd /tmp
  wget -O code_latest_arm64.deb "https://update.code.visualstudio.com/latest/linux-deb-arm64/stable"
  $SUDO apt install -y ./code_latest_arm64.deb
else
  echo "VS Code already installed, skipping download."
fi

echo "[4/12] Ensure GTK and pkg-config present"
$SUDO apt install -y --allow-change-held-packages libgtk2.0-dev pkg-config

echo "[5/12] Upgrade pip and wheel (user)"
python3 -m pip install --user -U pip setuptools wheel

echo "[6/12] Install Python libraries"
python3 -m pip install --user opencv-python scipy pandas matplotlib paho-mqtt
python3 -m pip install --user tensorflow==2.12.0 scikit-learn==1.4.2 scikit-image tflite-runtime || true
python3 -m pip install --user --force-reinstall --no-cache-dir numpy==1.26.4

echo "[7/12] Install Mosquitto broker + clients"
$SUDO apt install -y --allow-change-held-packages mosquitto mosquitto-clients

echo "[8/12] Open firewall ports 1883 and 5901"
$SUDO mkdir -p /etc/iptables
$SUDO iptables -A INPUT -p tcp --dport 1883 -j ACCEPT || true
$SUDO iptables -A INPUT -p tcp --dport 5901 -j ACCEPT || true
$SUDO sh -c 'iptables-save > /etc/iptables/rules.v4' || true

echo "[9/12] Create systemd service for Cycle Time script"
SERVICE_PATH="/etc/systemd/system/cycle_time_script.service"
$SUDO tee "$SERVICE_PATH" >/dev/null <<'EOF'
[Unit]
Description=Cycle Time Python Script
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/linaro/Downloads/CycleTime_DY08_P6_main_02.py
Restart=always
RestartSec=10
User=linaro
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/linaro/.Xauthority
WorkingDirectory=/home/linaro/Downloads

[Install]
WantedBy=multi-user.target
EOF
$SUDO systemctl daemon-reload
$SUDO systemctl enable cycle_time_script.service || true

echo "[10/12] Install TigerVNC + LXDE"
$SUDO apt install -y --allow-change-held-packages tigervnc-viewer tigervnc-standalone-server lxde

echo "[11/12] Create VNC startup script and service"
START_VNC="/home/linaro/start_vnc.sh"
cat > "$START_VNC" <<'EOF'
#!/bin/bash
/usr/bin/vncserver -localhost no
/usr/bin/vncserver -kill :1
/usr/bin/vncserver :1
EOF
chown linaro:linaro "$START_VNC" || true
chmod +x "$START_VNC"

VNC_SERVICE="/etc/systemd/system/vncserver.service"
$SUDO tee "$VNC_SERVICE" >/dev/null <<'EOF'
[Unit]
Description=Start TightVNC server at startup
After=syslog.target network.target

[Service]
Type=forking
User=linaro
WorkingDirectory=/home/linaro
Environment=HOME=/home/linaro
ExecStart=/home/linaro/start_vnc.sh
ExecStop=/usr/bin/vncserver -kill :1

[Install]
WantedBy=multi-user.target
EOF
$SUDO systemctl daemon-reload
$SUDO systemctl enable vncserver.service || true

echo "[12/12] Setup complete."
echo "Start services manually:"
echo "  sudo systemctl start cycle_time_script.service"
echo "  sudo systemctl start vncserver.service"
