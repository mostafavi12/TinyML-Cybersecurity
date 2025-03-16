# TinyML Cybersecurity Deployment on Renode

This project demonstrates how to run a trained TensorFlow Lite model on a virtual embedded device using Renode (ESP32 target).

## 📁 Structure
- renode/
  - firmware/
    - main.c: TinyML inference C program
    - cnn_model.tflite: Your TFLite model (copy your own here)
    - firmware.elf: compiled output
  - renode_script.resc: Renode automation script

## 🛠 Build Instructions
1. Install toolchain: `sudo apt install gcc-arm-none-eabi`
2. Compile firmware: `arm-none-eabi-gcc -o firmware.elf main.c -O2`
3. Copy cnn_model.tflite here
4. Run Renode and use `include @renode/renode_script.resc`

Enjoy your TinyML simulation!
