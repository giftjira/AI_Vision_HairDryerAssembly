  #!/usr/bin/env bash
# ASUS Tinker Edge R - One-Click Setup (Debian 10/Buster)
# VS Code pinned to 1.85.2 (ARM64) and held

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

log() { printf "\n[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else echo "Please run as root or install sudo."; exit 1; fi
fi

log "0/16 Switch to Debian Buster archive mirrors"
$SUDO cp /etc/apt/sources.list /etc/apt/sources.list.bak || true
$SUDO tee /etc/apt/sources.list >/dev/null <<'EOF'
deb [check-valid-until=no] http://archive.debian.org/debian buster main contrib non-free
deb [check-valid-until=no] http://archive.debian.org/debian buster-updates main contrib non-free
deb [check-valid-until=no] http://archive.debian.org/debian-security buster/updates main contrib non-free
EOF
echo 'Acquire::Check-Valid-Until "false";' | $SUDO tee /etc/apt/apt.conf.d/99ignore-check-valid-until >/dev/null

log "1/16 Clean & update"
$SUDO apt-get clean
$SUDO apt-get update -y

log "2/16 Upgrade (respect holds)"
HELD=$($SUDO apt-mark showhold || true)
if [ -n "$HELD" ]; then echo "Held packages:"; echo "$HELD"; fi
$SUDO apt-get upgrade -y --allow-change-held-packages || {
  echo "Trying fix-broken + dist-upgrade..."
  $SUDO apt --fix-broken install -y || true
  $SUDO apt-get -o Dpkg::Options::="--force-confnew" dist-upgrade -y --allow-change-held-packages || true
}

log "3/16 Install base packages"
$SUDO apt-get install -y python3 python3-tk python3-pip python3-dev build-essential cmake git pkg-config   libjpeg-dev libtiff5-dev libpng-dev   libavcodec-dev libavformat-dev libswscale-dev   libv4l-dev libxvidcore-dev libx264-dev   libgtk2.0-dev libatlas-base-dev gfortran   libssl-dev zlib1g-dev libbz2-dev libreadline-dev libncurses5-dev libffi-dev liblzma-dev   wget curl ca-certificates firefox-esr net-tools iptables-persistent filezilla || true

log "4/16 Resolve Buster version skews"
$SUDO apt-get install -y --allow-downgrades   libhdf5-103=1.10.4+repack-10+deb10u1 libhdf5-dev=1.10.4+repack-10+deb10u1   libsqlite3-0=3.27.2-3+deb10u2 libsqlite3-dev=3.27.2-3+deb10u2   python3.7-venv || true
$SUDO apt-get -f install -y || true

log "5/16 Upgrade pip (user)"
python3 -m pip install --user -U pip setuptools wheel

log "6/16 Python libraries"
python3 -m pip install --user opencv-python scipy pandas matplotlib paho-mqtt || true
python3 -m pip install --user scikit-learn scikit-image tflite-runtime || true

log "7/16 Mosquitto"
$SUDO apt-get install -y mosquitto mosquitto-clients

log "8/16 Firewall"
$SUDO mkdir -p /etc/iptables
$SUDO iptables -A INPUT -p tcp --dport 1883 -j ACCEPT || true
$SUDO iptables -A INPUT -p tcp --dport 5901 -j ACCEPT || true
$SUDO sh -c 'iptables-save > /etc/iptables/rules.v4' || true

log "9/16 VS Code 1.85.2 cleanup and install"
$SUDO apt -f install -y || true
$SUDO apt purge -y code || true
cd /tmp
wget -O code_1.85.2_arm64.deb "https://update.code.visualstudio.com/1.85.2/linux-deb-arm64/stable"
$SUDO apt-get update -y
if $SUDO apt-get install -y ./code_1.85.2_arm64.deb; then
  if code --version 2>/dev/null | head -n1 | grep -q "1.85.2"; then
    echo "VS Code installed: $(code --version | head -n1)"
    $SUDO apt-mark hold code
    echo "VS Code held at 1.85.2"
  else
    echo "VS Code install completed but version check failed."
  fi
else
  echo "VS Code 1.85.2 install failed; continuing without it."
fi

log "10/16 GTK/pkg-config ensure"
$SUDO apt-get install -y libgtk2.0-dev pkg-config

log "11/16 Camera utils"
$SUDO apt-get install -y v4l-utils || true
v4l2-ctl --list-devices || true

log "12/16 Cycle Time systemd service"
SERVICE_PATH="/etc/systemd/system/cycle_time_script.service"
$SUDO tee "$SERVICE_PATH" >/dev/null <<'EOF_SVC'
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
EOF_SVC
$SUDO systemctl daemon-reload
$SUDO systemctl enable cycle_time_script.service || true

log "13/16 TigerVNC + LXDE"
$SUDO apt-get install -y tigervnc-viewer tigervnc-standalone-server lxde || true

log "14/16 VNC helper + service"
START_VNC="/home/linaro/start_vnc.sh"
cat > "$START_VNC" <<'EOF_VNC'
#!/bin/bash
/usr/bin/vncserver -localhost no
/usr/bin/vncserver -kill :1
/usr/bin/vncserver :1
EOF_VNC
chown linaro:linaro "$START_VNC" || true
chmod +x "$START_VNC"
VNC_SERVICE="/etc/systemd/system/vncserver.service"
$SUDO tee "$VNC_SERVICE" >/dev/null <<'EOF_VNCSVC'
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
EOF_VNCSVC
$SUDO systemctl daemon-reload
$SUDO systemctl enable vncserver.service || true

log "15/16 Cleanup"
$SUDO apt-get autoremove -y || true

log "16/16 Done on Tinker Edge R."
echo "Start services:"
echo "  sudo systemctl start cycle_time_script.service"
echo "  sudo systemctl start vncserver.service"
