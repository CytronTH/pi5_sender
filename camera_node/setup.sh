#!/bin/bash

echo "========================================="
echo "  Camera Sender Setup Script for RPi"
echo "========================================="

# Update package lists
echo "[1/4] Updating package lists..."
sudo apt update

# Install system dependencies (Picamera2 and OpenCV requirements)
echo "[2/4] Installing system dependencies..."
sudo apt install -y python3-picamera2 python3-pip libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
echo "[3/4] Installing Python dependencies..."
pip3 install -r requirements.txt --break-system-packages 2>/dev/null || pip3 install -r requirements.txt

# Install Systemd Template Service
echo "[4/4] Installing Dual Camera Systemd Service..."
sudo systemctl stop camera-sender@0.service 2>/dev/null || true
sudo systemctl stop camera-sender@1.service 2>/dev/null || true
sudo systemctl disable camera-sender@0.service 2>/dev/null || true
sudo systemctl disable camera-sender@1.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/camera-sender@.service

sudo systemctl stop camera-app.service 2>/dev/null || true
sudo systemctl disable camera-app.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/camera-app.service

# Inject current user and full directory path into the service file
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)
sed -e "s|CURRENT_USER_PLACEHOLDER|${CURRENT_USER}|g" \
    -e "s|CURRENT_DIR_PLACEHOLDER|${CURRENT_DIR}|g" \
    camera-sender@.service > /tmp/camera-sender@.service.tmp

sed -e "s|CURRENT_USER_PLACEHOLDER|${CURRENT_USER}|g" \
    -e "s|CURRENT_DIR_PLACEHOLDER|${CURRENT_DIR}|g" \
    camera-app.service > /tmp/camera-app.service.tmp

sudo cp /tmp/camera-sender@.service.tmp /etc/systemd/system/camera-sender@.service
sudo cp /tmp/camera-app.service.tmp /etc/systemd/system/camera-app.service

sudo systemctl daemon-reload
sudo rm -f /tmp/camera-sender@.service.tmp
sudo rm -f /tmp/camera-app.service.tmp

echo "========================================="
echo "Setup complete! The service is now template-based."
echo "To manage Camera 0 (config_cam0.json):"
echo "  sudo systemctl start camera-sender@0"
echo "  sudo systemctl enable camera-sender@0"
echo ""
echo "To manage Camera 1 (config_cam1.json):"
echo "  sudo systemctl start camera-sender@1"
echo "  sudo systemctl enable camera-sender@1"
echo ""
echo "To manage the Web App API:"
echo "  sudo systemctl start camera-app"
echo "  sudo systemctl enable camera-app"
echo "========================================="
