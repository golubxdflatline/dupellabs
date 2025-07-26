import tkinter as tk
import subprocess
import sys
import os
import threading
import time
import platform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PACKAGES = [
    'torch',
    'openai-whisper',
    'pyttsx3',
    'coqui-tts',
    'googletrans==3.1.0a0',
    'requests',
    'numpy',
    'librosa',
    'soundfile',
    'pydub'
]

class InstallerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("[P.C. CAHTEXHNKN INSTALLER 0.5]")
        self.root.geometry("375x375")
        self.root.resizable(False, False)

        self.setup_essential_ui()


        self.load_gif_with_fallback()

        self.start_installation()

    def setup_essential_ui(self):
        self.main_frame = tk.Frame(self.root, padx=0, pady=0)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.gif_label = tk.Label(self.main_frame)
        self.gif_label.pack(pady=10)

    def load_gif_with_fallback(self):
        gif_filenames = [
            "cryak.gif",
        ]

        found = False
        for gif_file in gif_filenames:
            full_gif_path = os.path.join(SCRIPT_DIR, gif_file)
            if os.path.exists(full_gif_path):
                self.load_gif(full_gif_path)
                found = True
                break

        if not found:
            self.show_error("КТО ГИФКУ УДАЛИЛ")
            self.create_placeholder_animation()

    def create_placeholder_animation(self):
        try:
            self.frames = []
            colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

            for color in colors:
                img = tk.PhotoImage(width=100, height=100)
                img.put(color, to=(25, 25, 75, 75))
                self.frames.append(img)

            self.current_frame = 0
            self.gif_label.configure(image=self.frames[0])
            self.gif_label.image = self.frames[0]

            self.animate_gif()
        except Exception as e:
            self.show_error(f"ДА БЛЯЯЯТЬ ЛГБТ НЕ УДАЛОСЬ: {str(e)}")

    def load_gif(self, gif_path):
        try:
            self.frames = []
            frame_count = 0

            while True:
                try:
                    frame = tk.PhotoImage(file=gif_path, format=f"gif -index {frame_count}")
                    self.frames.append(frame)
                    frame_count += 1
                except tk.TclError:
                    break

            if frame_count == 0:
                try:
                    frame = tk.PhotoImage(file=gif_path)
                    self.frames.append(frame)
                except Exception as e:
                    self.show_error(f"Я НЕ СМОГ ГИФКУ ЗАГРУЗИТЬ: {str(e)}")
                    return

            self.current_frame = 0
            self.gif_label.configure(image=self.frames[0])
            self.gif_label.image = self.frames[0]

            if len(self.frames) > 1:
                self.animate_gif()
        except Exception as e:
            self.show_error(f"Я НЕ СМОГ ГИФКУ ЗАГРУЗИТЬ: {str(e)}")

    def animate_gif(self):
        if hasattr(self, 'frames') and self.frames:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            frame = self.frames[self.current_frame]
            self.gif_label.configure(image=frame)
            self.gif_label.image = frame
            self.root.after(50, self.animate_gif)

    def play_sound(self):
        sound_filenames = [
            "cryak.wav",
        ]

        found_sound = None
        for sound_file in sound_filenames:
            full_sound_path = os.path.join(SCRIPT_DIR, sound_file)
            if os.path.exists(full_sound_path):
                found_sound = full_sound_path
                break

        if not found_sound:
            return

        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.PlaySound(found_sound, winsound.SND_FILENAME | winsound.SND_ASYNC)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["afplay", found_sound])
            elif system == "Linux":
                try:
                    subprocess.Popen(["aplay", found_sound])
                except FileNotFoundError:
                    subprocess.Popen(["ffplay", "-nodisp", "-autoexit", found_sound])
        except Exception as e:
            self.show_error(f"музыка не удалась чювак: {e}")

    def show_error(self, message):
        error_message = f"ERROR: {message}"
        self.root.title(error_message)

    def install_packages(self):
        total = len(PACKAGES)

        for i, package in enumerate(PACKAGES):
            msg = f"ГРУЗИМ {package} ({i+1}/{total})"
            self.root.title(msg)
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                for line in iter(process.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        self.root.title(f"{msg}: {line}")

                process.wait()

                if process.returncode != 0:
                    self.show_error(f"Я НЕ СМОГ УСТАНОВИТЬ {package} (КИНЬ ДЕВЕЛОПЕРУ {process.returncode})")
                    return

            except Exception as e:
                self.show_error(f"Я НЕ СМОГ УСТАНОВИТЬ {package}: {str(e)}")
                return

        self.root.title("УСТАНОВКА ЗАВЕРШЕНА! [P.C. CAHTEXHNKN]")
        time.sleep(3)
        self.root.destroy()

    def start_installation(self):
        threading.Thread(
            target=self.play_sound,
            daemon=True
        ).start()

        threading.Thread(
            target=self.install_packages,
            daemon=True
        ).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = InstallerApp(root)
    root.mainloop()
