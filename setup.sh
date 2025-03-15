#!/bin/bash

echo "[*] Updating system..."
sudo apt update && sudo apt upgrade -y

echo "[*] Installing required system packages..."
sudo apt install -y python3 python3-pip git curl wireshark tshark docker.io

echo "[*] Creating Python virtual environment..."
python3 -m venv myEenv
source myEenv/bin/activate

echo "[*] Installing required Python libraries..."
pip install -r requirements.txt

echo "[*] Setting up Renode..."
wget https://builds.renode.io/renode-latest.linux-portable.tar.gz
tar -xzf renode-latest.linux-portable.tar.gz
mv renode-* renode/

echo "[✓] Setup completed! To start, activate the virtual environment using:"
echo "source myEenv/bin/activate"
