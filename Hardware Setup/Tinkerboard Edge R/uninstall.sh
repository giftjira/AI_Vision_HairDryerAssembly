#!/usr/bin/env bash
sudo systemctl disable --now cycle_time_script.service vncserver.service || true
sudo rm -f /etc/systemd/system/cycle_time_script.service /etc/systemd/system/vncserver.service || true
sudo systemctl daemon-reload || true
rm -f /home/linaro/start_vnc.sh || true
