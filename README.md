# ROS2 Speech Recognition Workspace

## Systeminformationen

- **Betriebssystem:** Ubuntu 24.02  
- **ROS2 Distribution:** Jazzy  
- **System Python Version:** 3.12.3  

---

# Installation & Setup

## 1. Workspace ins Home-Verzeichnis kopieren

Der Ordner `ros2_ws` muss sich im Home-Verzeichnis befinden:

```bash
cp -r ros2_ws ~/
```

Danach liegt der Workspace hier:

```bash
~/ros2_ws
```

---

## 2. Python 3.11.9 Virtual Environment erstellen

Falls Python 3.11 noch nicht installiert ist:

```bash
sudo apt install python3.11 python3.11-venv
```

Virtual Environment im Home-Verzeichnis erstellen:

```bash
cd ~
python3.11 -m venv venv
```

---

## 3. Virtual Environment aktivieren

```bash
source ~/venv/bin/activate
```

---

## 4. Benötigte Python-Pakete installieren

```bash
pip install sounddevice librosa tensorflow joblib pyusb rclpy
```

---

## 5. ROS2 Umgebung laden

```bash
source /opt/ros/jazzy/setup.bash
```

---

## 6. Workspace bauen

```bash
cd ~/ros2_ws
colcon build
```

---

## 7. Workspace sourcen

```bash
source install/setup.bash
```

---

## 8. Node starten

```bash
ros2 run speech_recognition speech_recognition_node
```

---

