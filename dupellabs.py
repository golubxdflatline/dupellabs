import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import shutil
import subprocess
import whisper
import pyttsx3
from pathlib import Path
import tempfile
import json
import requests
from googletrans import Translator
import torch
from TTS.api import TTS
import numpy as np
import librosa
import soundfile as sf
from audio_separator.separator import Separator
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip


class dupellabs:
    def __init__(self, root):
        self.root = root
        self.root.title("dupellabs")
        self.root.geometry("1000x900")
        
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.reference_audio_path = tk.StringVar()
        self.whisper_model = None
        self.tts_engine = None
        self.voice_clone_model = None
        self.translator = Translator()
        self.separator = Separator()
        self.is_processing = False
        
        self.whisper_languages = {
            'Auto-detect': None,
            'Afrikaans': 'af',
            'Albanian': 'sq',
            'Amharic': 'am',
            'Arabic': 'ar',
            'Armenian': 'hy',
            'Assamese': 'as',
            'Azerbaijani': 'az',
            'Bashkir': 'ba',
            'Basque': 'eu',
            'Belarusian': 'be',
            'Bengali': 'bn',
            'Bosnian': 'bs',
            'Breton': 'br',
            'Bulgarian': 'bg',
            'Burmese': 'my',
            'Castilian': 'es',
            'Catalan': 'ca',
            'Chinese': 'zh',
            'Croatian': 'hr',
            'Czech': 'cs',
            'Danish': 'da',
            'Dutch': 'nl',
            'English': 'en',
            'Estonian': 'et',
            'Faroese': 'fo',
            'Finnish': 'fi',
            'Flemish': 'nl',
            'French': 'fr',
            'Galician': 'gl',
            'Georgian': 'ka',
            'German': 'de',
            'Greek': 'el',
            'Gujarati': 'gu',
            'Haitian': 'ht',
            'Hausa': 'ha',
            'Hawaiian': 'haw',
            'Hebrew': 'he',
            'Hindi': 'hi',
            'Hungarian': 'hu',
            'Icelandic': 'is',
            'Indonesian': 'id',
            'Italian': 'it',
            'Japanese': 'ja',
            'Javanese': 'jw',
            'Kannada': 'kn',
            'Kazakh': 'kk',
            'Khmer': 'km',
            'Korean': 'ko',
            'Lao': 'lo',
            'Latin': 'la',
            'Latvian': 'lv',
            'Lingala': 'ln',
            'Lithuanian': 'lt',
            'Luxembourgish': 'lb',
            'Macedonian': 'mk',
            'Malagasy': 'mg',
            'Malay': 'ms',
            'Malayalam': 'ml',
            'Maltese': 'mt',
            'Mandarin': 'zh',
            'Maori': 'mi',
            'Marathi': 'mr',
            'Mongolian': 'mn',
            'Nepali': 'ne',
            'Norwegian': 'no',
            'Nynorsk': 'nn',
            'Occitan': 'oc',
            'Pashto': 'ps',
            'Persian': 'fa',
            'Polish': 'pl',
            'Portuguese': 'pt',
            'Punjabi': 'pa',
            'Romanian': 'ro',
            'Russian': 'ru',
            'Sanskrit': 'sa',
            'Serbian': 'sr',
            'Shona': 'sn',
            'Sindhi': 'sd',
            'Sinhala': 'si',
            'Slovak': 'sk',
            'Slovenian': 'sl',
            'Somali': 'so',
            'Spanish': 'es',
            'Sundanese': 'su',
            'Swahili': 'sw',
            'Swedish': 'sv',
            'Tagalog': 'tl',
            'Tajik': 'tg',
            'Tamil': 'ta',
            'Tatar': 'tt',
            'Telugu': 'te',
            'Thai': 'th',
            'Tibetan': 'bo',
            'Turkish': 'tr',
            'Turkmen': 'tk',
            'Ukrainian': 'uk',
            'Urdu': 'ur',
            'Uzbek': 'uz',
            'Vietnamese': 'vi',
            'Welsh': 'cy',
            'Yiddish': 'yi',
            'Yoruba': 'yo'
        }
        
        self.translation_languages = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Chinese (Simplified)': 'zh-cn',
            'Arabic': 'ar',
            'Hindi': 'hi',
            'Dutch': 'nl',
            'Polish': 'pl',
            'Turkish': 'tr',
            'Swedish': 'sv',
            'Norwegian': 'no',
            'Danish': 'da',
            'Finnish': 'fi',
            'Greek': 'el',
            'Hebrew': 'he',
            'Thai': 'th',
            'Vietnamese': 'vi',
            'Czech': 'cs',
            'Hungarian': 'hu',
            'Romanian': 'ro',
            'Bulgarian': 'bg',
            'Croatian': 'hr',
            'Serbian': 'sr',
            'Slovak': 'sk',
            'Slovenian': 'sl',
            'Estonian': 'et',
            'Latvian': 'lv',
            'Lithuanian': 'lt',
            'Ukrainian': 'uk',
            'Bengali': 'bn',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Gujarati': 'gu',
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'Marathi': 'mr',
            'Punjabi': 'pa',
            'Urdu': 'ur',
            'Indonesian': 'id',
            'Malay': 'ms',
            'Tagalog': 'tl',
            'Swahili': 'sw',
            'Persian': 'fa'
        }
        
        self.init_tts_engine()
        self.init_voice_clone_model()
        
        self.setup_gui()
        
    def init_tts_engine(self):
        """инициализация ттс движка"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            messagebox.showerror("ошибка ттс", f"не получилось инициализировать движок ттс: {str(e)}")
    
    def init_voice_clone_model(self):
        """клон голоса"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.voice_clone_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        except Exception as e:
            print(f"ошибка модели клонирования: {str(e)}")
            self.voice_clone_model = None
    
    def setup_gui(self):
        """сетап интерфейса"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text='главные настройки')

        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text='дополнительные настройки')
        
        self.setup_main_tab(main_tab)
        self.setup_advanced_tab(advanced_tab)
        

        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        ttk.Label(log_frame, text="логи:").pack(anchor='w')
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill='both', expand=True)
    
    def setup_main_tab(self, parent):
        """сетап мейн говнища"""
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill='both', expand=True)
        
   
        title_label = ttk.Label(main_frame, text="dupellabs", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        

        ttk.Label(main_frame, text="видео для перевода:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=60).grid(row=1, column=1, padx=(5, 5), sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="указать", command=self.browse_video).grid(row=1, column=2)
        

        ttk.Label(main_frame, text="выходная директория:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_path, width=60).grid(row=2, column=1, padx=(5, 5), sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="указать", command=self.browse_output).grid(row=2, column=2)
        

        lang_frame = ttk.LabelFrame(main_frame, text="настройки языка", padding="10")
        lang_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(lang_frame, text="язык в видео (whisper):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_lang_var = tk.StringVar(value="Auto-detect")
        source_lang_combo = ttk.Combobox(lang_frame, textvariable=self.source_lang_var, 
                                        values=list(self.whisper_languages.keys()), state="readonly", width=20)
        source_lang_combo.grid(row=0, column=1, padx=(5, 20), sticky=tk.W)
        
        ttk.Label(lang_frame, text="целевой язык (перевод):").grid(row=0, column=2, sticky=tk.W, pady=5)
        self.target_lang_var = tk.StringVar(value="Spanish")
        target_lang_combo = ttk.Combobox(lang_frame, textvariable=self.target_lang_var, 
                                        values=list(self.translation_languages.keys()), state="readonly", width=20)
        target_lang_combo.grid(row=0, column=3, padx=(5, 0), sticky=tk.W)
        

        model_frame = ttk.LabelFrame(main_frame, text="ИИ модели", padding="10")
        model_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(model_frame, text="модель whisper:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="large")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  values=["tiny", "base", "small", "medium", "large"], 
                                  state="readonly")
        model_combo.grid(row=0, column=1, padx=(5, 40), sticky=tk.W)

        ttk.Label(model_frame, text="модель separator (сверху - качество, снизу - скорость)").grid(row=0, column=3, sticky=tk.W, pady=5)
        self.separator_model = tk.StringVar(value="model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt")
        separator_model_combo = ttk.Combobox(model_frame,textvariable=self.separator_model,
                                             values=["model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "UVR-MDX-NET-Inst_HQ_3.onnx", "UVR_MDXNET_KARA_2.onnx"],
                                             state="readonly")
        separator_model_combo.grid(row=0, column=4, padx=(5, 0), sticky=tk.W)
        

        voice_frame = ttk.LabelFrame(main_frame, text="настройки голоса", padding="10")
        voice_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.voice_method_var = tk.StringVar(value="video_clone")
        ttk.Radiobutton(voice_frame, text="дефолтный ттс (для тестов?)", variable=self.voice_method_var, 
                       value="standard").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(voice_frame, text="клонировать голос из видео", variable=self.voice_method_var, 
                       value="video_clone").grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Radiobutton(voice_frame, text="клонировать из файла", variable=self.voice_method_var, 
                       value="file_clone").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        

        ttk.Label(voice_frame, text="референс звук (только для файла):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(voice_frame, textvariable=self.reference_audio_path, width=50).grid(row=1, column=1, columnspan=2, padx=(5, 5), sticky=(tk.W, tk.E))
        ttk.Button(voice_frame, text="указать", command=self.browse_reference_audio).grid(row=1, column=3)

        separator_settings_frame = ttk.LabelFrame(main_frame, text="настройки separator (отделяет голос от музыки)", padding="10")
        separator_settings_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        self.separator_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(separator_settings_frame, text="включить separator?", variable=self.separator_enabled).pack(anchor='w')
        
        tts_settings_frame = ttk.LabelFrame(main_frame, text="ттс настройки", padding="10")
        tts_settings_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(tts_settings_frame, text="скорость:").grid(row=0, column=0, padx=(0, 5))
        self.speed_var = tk.IntVar(value=150)
        speed_scale = ttk.Scale(tts_settings_frame, from_=50, to=300, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=0, column=1, padx=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Label(tts_settings_frame, text="звук:").grid(row=0, column=2, padx=(0, 5))
        self.volume_var = tk.DoubleVar(value=0.9)
        volume_scale = ttk.Scale(tts_settings_frame, from_=0.1, to=1.0, variable=self.volume_var, orient=tk.HORIZONTAL)
        volume_scale.grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        tts_settings_frame.columnconfigure(1, weight=1)
        tts_settings_frame.columnconfigure(3, weight=1)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=20)
        
        self.process_btn = ttk.Button(button_frame, text="подписаться на канал лаласкул", 
                                     command=self.start_dubbing_process)
        self.process_btn.pack(pady=5)
        
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)
        
        self.status_var = tk.StringVar(value="Я ГОТОВ НАХУЙ")
        ttk.Label(button_frame, textvariable=self.status_var).pack()
        
        main_frame.columnconfigure(1, weight=1)
        lang_frame.columnconfigure(1, weight=1)
        lang_frame.columnconfigure(3, weight=1)
        voice_frame.columnconfigure(1, weight=1)
    
    def setup_advanced_tab(self, parent):
        """сетап доп настроек"""
        advanced_frame = ttk.Frame(parent, padding="10")
        advanced_frame.pack(fill='both', expand=True)
        
        trans_frame = ttk.LabelFrame(advanced_frame, text="настройки перевода", padding="10")
        trans_frame.pack(fill='x', pady=10)
        
        self.preserve_timing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(trans_frame, text="сохранить тайминг фраз", 
                       variable=self.preserve_timing_var).pack(anchor='w')
        
        self.chunk_translation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(trans_frame, text="переводить по кускам", 
                       variable=self.chunk_translation_var).pack(anchor='w')
        

        audio_frame = ttk.LabelFrame(advanced_frame, text="обработка звука", padding="10")
        audio_frame.pack(fill='x', pady=10)
        
        self.noise_reduction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(audio_frame, text="шумоподавление", 
                       variable=self.noise_reduction_var).pack(anchor='w')
        
        self.normalize_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(audio_frame, text="нормализировать звук", 
                       variable=self.normalize_audio_var).pack(anchor='w')
        
        clone_frame = ttk.LabelFrame(advanced_frame, text="настройки клона голоса", padding="10")
        clone_frame.pack(fill='x', pady=10)
        
        ttk.Label(clone_frame, text="порог схожести голоса:").pack(anchor='w')
        self.similarity_var = tk.DoubleVar(value=0.8)
        similarity_scale = ttk.Scale(clone_frame, from_=0.5, to=1.0, variable=self.similarity_var, orient=tk.HORIZONTAL)
        similarity_scale.pack(fill='x', padx=(0, 0))
        
        self.enhance_voice_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(clone_frame, text="улучшить качество голоса", 
                       variable=self.enhance_voice_var).pack(anchor='w')
        
        extract_frame = ttk.LabelFrame(advanced_frame, text="настройки извлечения голоса", padding="10")
        extract_frame.pack(fill='x', pady=10)
        
        ttk.Label(extract_frame, text="минимальная длина сэмпла (сек):").pack(anchor='w')
        self.min_sample_length_var = tk.DoubleVar(value=3.0)
        sample_scale = ttk.Scale(extract_frame, from_=1.0, to=10.0, variable=self.min_sample_length_var, 
                                orient=tk.HORIZONTAL)
        sample_scale.pack(fill='x')
        
        self.voice_activity_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(extract_frame, text="детекция голосовой активности", 
                       variable=self.voice_activity_detection_var).pack(anchor='w')
    
    def browse_video(self):
        """найти видео говну"""
        filename = filedialog.askopenfilename(
            title="выбор видео файла",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
    
    def browse_output(self):
        """найти финальную директорию"""
        directory = filedialog.askdirectory(title="выбор директории выгрузки файла")
        if directory:
            self.output_path.set(directory)
    
    def browse_reference_audio(self):
        """найти реф голоса"""
        filename = filedialog.askopenfilename(
            title="выбор референс голоса",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.reference_audio_path.set(filename)
    
    def log(self, message):
        """логируем всякую хуйню"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, status):
        """обнова статуса"""
        self.status_var.set(status)
        self.root.update_idletasks()
    
    def extract_audio(self, video_path, audio_path):
        """экстракт звука через ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '22050', '-ac', '1',
                '-y', audio_path
            ]
            
            if self.noise_reduction_var.get():
                cmd.extend(['-af', 'anlmdn'])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ошибка FFmpeg: {result.stderr}")
            
            return True
        except Exception as e:
            self.log(f"ошибка экстракта файла: {str(e)}")
            return False
        
    def combine_audio(self, audio_1, audio_2, output_path):
        """комбинирование вокала с инструменталом"""
        try:
            cmd = [
                'ffmpeg', '-i', audio_1, 
                '-i', audio_2, '-filter_complex', 
                '[0:a][1:a]amix=inputs=2:duration=longest',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ошибка FFmpeg: {result.stderr}")
            
            return True
        except Exception as e:
            self.log(f"ошибка комбинирования: {str(e)}")
            return False
    
    def extract_voice_sample_from_video(self, video_path, output_audio_path, transcript_result, audio_vocal):
        """извлекаем сэмпл голоса из видео для клонирования"""
        try:
            self.update_status("извлекаем сэмпл голоса из видео...")
            self.log("извлекаем сэмпл голоса...")

            temp_full_audio = output_audio_path + "_full.wav"

            if audio_vocal and os.path.exists(audio_vocal):
                shutil.copyfile(audio_vocal, temp_full_audio)
            elif not self.extract_audio(video_path, temp_full_audio):
                return None

            audio_data, sr = librosa.load(temp_full_audio, sr=22050)
            
            segments = transcript_result.get('segments', [])
            if not segments:
                self.log("используем весь звук")
                sf.write(output_audio_path, audio_data, sr)
                os.remove(temp_full_audio)
                return output_audio_path
            
            best_segments = []
            min_length = self.min_sample_length_var.get()
            
            for segment in segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                duration = end_time - start_time
                
                if duration >= min_length:
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    if self.voice_activity_detection_var.get():
                        rms = np.sqrt(np.mean(segment_audio**2))
                        if rms > 0.01:
                            best_segments.append({
                                'audio': segment_audio,
                                'duration': duration,
                                'rms': rms,
                                'start': start_time,
                                'end': end_time
                            })
            
            if not best_segments:
                self.log("не найдены подходящие сегменты речи, используем первые 10 секунд")

                segment_audio = audio_data[:int(10 * sr)]
                sf.write(output_audio_path, segment_audio, sr)
                os.remove(temp_full_audio)
                return output_audio_path
            
            best_segments.sort(key=lambda x: x['rms'], reverse=True)
            
            combined_audio = []
            total_duration = 0
            max_duration = 45.0
            
            for segment in best_segments:
                if total_duration + segment['duration'] <= max_duration:
                    combined_audio.append(segment['audio'])
                    total_duration += segment['duration']
                    self.log(f"добавлен сегмент: {segment['start']:.1f}s - {segment['end']:.1f}s")
                
            
            if combined_audio:
                final_audio = np.concatenate(combined_audio)
                
                if self.noise_reduction_var.get():
                    final_audio = self.simple_noise_reduction(final_audio, sr)
                
                sf.write(output_audio_path, final_audio, sr)
                self.log(f"создан сэмпл голоса: {total_duration:.1f}s длиной")
            else:
                sf.write(output_audio_path, best_segments[0]['audio'], sr)
                self.log("использован первый доступный сегмент")
            
            os.remove(temp_full_audio)
            return output_audio_path
            
        except Exception as e:
            self.log(f"ошибка извлечения сэмпла голоса: {str(e)}")
            return None
    
    def simple_noise_reduction(self, audio, sr):
        """простое шумоподавление"""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            from scipy import signal
            nyquist = sr / 2
            low_cutoff = 80 / nyquist
            b, a = signal.butter(5, low_cutoff, btype='high')
            cleaned_audio = signal.filtfilt(b, a, audio)
            
            return cleaned_audio
        except Exception as e:
            self.log(f"предупреждение: не удалось применить шумоподавление: {str(e)}")
            return audio
    
    def transcribe_audio(self, audio_path):
        """транскрипция этого говнища"""
        try:
            self.update_status("загружаем модель whisper...")
            self.log(f"загружаем модель whisper: {self.model_var.get()}")
            
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model(self.model_var.get())
            
            self.update_status("транскрипция звука...")
            self.log("начинаем транскрипцию...")
            
            source_lang_code = self.whisper_languages[self.source_lang_var.get()]
            
            result = self.whisper_model.transcribe(
                audio_path, 
                language=source_lang_code,
                word_timestamps=True
            )
            
            return result
        except Exception as e:
            self.log(f"ошибка транскрипции: {str(e)}")
            return None
    
    def translate_text(self, text, source_lang, target_lang):
        """переводим через различные методы"""
        try:
            if source_lang == target_lang:
                return text
            
            self.update_status("переводим текст...")
            self.log(f"переводим с {source_lang} на {target_lang}")
            
            if source_lang in self.whisper_languages:
                source_code = self.whisper_languages[source_lang]
                if source_code is None:
                    source_code = 'auto'
            else:
                source_code = self.translation_languages.get(source_lang, 'en')
            
            target_code = self.translation_languages[target_lang]
            
            return self._try_translation_methods(text, source_code, target_code)
            
        except Exception as e:
            self.log(f"ошибка перевода: {str(e)}")
            return text
    
    def _try_translation_methods(self, text, source_code, target_code):
        """пробуем разные методы перевода"""
        
        try:
            self.log("метод 1: пробуем старую версию googletrans...")
            from googletrans import Translator
            translator = Translator()
            
            if self.chunk_translation_var.get():
                sentences = [s.strip() for s in text.split('. ') if s.strip()]
                translated_sentences = []
                
                for i, sentence in enumerate(sentences):
                    try:
                        self.log(f"переводим предложение {i+1}/{len(sentences)}")
                        result = translator.translate(sentence, src=source_code, dest=target_code)
                        if hasattr(result, 'text'):
                            translated_sentences.append(result.text)
                        else:
                            translated_sentences.append(sentence)
                    except:
                        translated_sentences.append(sentence)
                
                return '. '.join(translated_sentences)
            else:
                result = translator.translate(text, src=source_code, dest=target_code)
                if hasattr(result, 'text'):
                    return result.text
                else:
                    raise Exception("no text attribute")
                    
        except Exception as e1:
            self.log(f"метод 1 не сработал: {str(e1)}")
        
        try:
            self.log("метод 2: прямой запрос к Google Translate API...")
            import urllib.parse
            import requests
            import json
            import re
            
            if self.chunk_translation_var.get():
                sentences = [s.strip() for s in text.split('. ') if s.strip()]
                translated_sentences = []
                
                for sentence in sentences:
                    translated_sentence = self._direct_google_translate(sentence, source_code, target_code)
                    translated_sentences.append(translated_sentence)
                
                return '. '.join(translated_sentences)
            else:
                return self._direct_google_translate(text, source_code, target_code)
                
        except Exception as e2:
            self.log(f"метод 2 не сработал: {str(e2)}")
        
        try:
            self.log("метод 3: пробуем deep_translator...")
            from deep_translator import GoogleTranslator
            
            translator = GoogleTranslator(source=source_code, target=target_code)
            
            if self.chunk_translation_var.get():
                sentences = [s.strip() for s in text.split('. ') if s.strip()]
                translated_sentences = []
                
                for sentence in sentences:
                    try:
                        translated = translator.translate(sentence)
                        translated_sentences.append(translated)
                    except:
                        translated_sentences.append(sentence)
                
                return '. '.join(translated_sentences)
            else:
                return translator.translate(text)
                
        except Exception as e3:
            self.log(f"метод 3 не сработал: {str(e3)}")
            self.log("предупреждение: установите deep_translator: pip install deep-translator")
        
        self.log("все методы перевода не сработали, возвращаем оригинальный текст")
        return text
    
    def _direct_google_translate(self, text, source_code, target_code):
        """прямой запрос к Google Translate"""
        try:
            import urllib.parse
            import requests
            import json
            import re
            
            encoded_text = urllib.parse.quote(text)
            
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_code}&tl={target_code}&dt=t&q={encoded_text}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and result[0]:
                    translated_parts = []
                    for part in result[0]:
                        if part and len(part) > 0:
                            translated_parts.append(part[0])
                    
                    return ''.join(translated_parts) if translated_parts else text
            
            return text
            
        except Exception as e:
            self.log(f"прямой запрос не сработал: {str(e)}")
            return text
    
    def generate_dubbed_audio_standard(self, text, output_audio_path):
        """генерируем звук через ттс"""
        try:
            self.update_status("генерация звука (дефолтный ттс)...")
            self.log("генерация звука ттс...")
            
            self.tts_engine.setProperty('rate', self.speed_var.get())
            self.tts_engine.setProperty('volume', self.volume_var.get())
            
            self.tts_engine.save_to_file(text, output_audio_path)
            self.tts_engine.runAndWait()
            
            if self.normalize_audio_var.get():
                self.normalize_audio(output_audio_path)
            
            return True
        except Exception as e:
            self.log(f"ошибка генерации ттс: {str(e)}")
            return False
    
    def generate_dubbed_audio_clone(self, text, output_audio_path, reference_audio_path):
        """дубляж ии"""
        try:
            if self.voice_clone_model is None:
                self.log("ребят извиняйте клон голоса не пашет, переходим на дефолт ттс")
                return self.generate_dubbed_audio_standard(text, output_audio_path)
            
            self.update_status("генерируем звук (клонирование голоса)...")
            self.log("генерируем звук клонированого голоса...")
            
            if not os.path.exists(reference_audio_path):
                self.log(f"референс аудио не найден: {reference_audio_path}")
                return self.generate_dubbed_audio_standard(text, output_audio_path)
            
            target_lang_code = self.translation_languages[self.target_lang_var.get()]
            
            self.voice_clone_model.tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language=target_lang_code,
                file_path=output_audio_path
            )
            
            if self.normalize_audio_var.get():
                self.normalize_audio(output_audio_path)
            
            return True
        except Exception as e:
            self.log(f"ошибка генерации клонированого голоса: {str(e)}")
            self.log("во избежание проблем переходим на дефолтный ттс...")
            return self.generate_dubbed_audio_standard(text, output_audio_path)
    
    def normalize_audio(self, audio_path):
        """нормализация звука"""
        try:
            temp_path = audio_path + "_temp.wav"
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'loudnorm',
                '-y', temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                os.replace(temp_path, audio_path)
                self.log("звук нормализирован")
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            self.log(f"ПРЕДУПРЕЖДЕНИЕ: не получилось нормализовать звук: {str(e)}")
    
    def combine_video_audio(self, video_path, audio_path, output_path):
        """комбинируем это все"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ошибка FFmpeg: {result.stderr}")
            
            return True
        except Exception as e:
            self.log(f"ошибка комбинирования звука и видео: {str(e)}")
            return False
    
    def dubbing_process(self):
        """мейн процесс"""
        try:
            video_file = self.video_path.get()
            output_dir = self.output_path.get()
            
            if not video_file or not output_dir:
                messagebox.showerror("дурачье", "ты еблан? выбери видео и куда мне файл выкладывать")
                return
            
            if not os.path.exists(video_file):
                messagebox.showerror("ошибка", "файла видео не существует.")
                return
            
            if not os.path.exists(output_dir):
                messagebox.showerror("ошибка", "директории финального файла не существует.")
                return
            
            voice_method = self.voice_method_var.get()
            if (voice_method == "file_clone" and not self.reference_audio_path.get()):
                messagebox.showerror("ошибка", "выберите референс звука для дубляжа файлом.")
                return
            
            with tempfile.TemporaryDirectory() as temp_dir:
                video_name = Path(video_file).stem
                temp_video = os.path.join(temp_dir,"temp_video.mp4")

                temp_audio = os.path.join(temp_dir, "extracted_audio.wav")
                temp_audio_vocal = os.path.join(temp_dir, "extracted_audio_vocal.wav")
                temp_audio_instrumental = os.path.join(temp_dir, "extracted_audio_instrumental.wav")
                temp_audio_names = {
                    "Vocals": "extracted_audio_vocal",
                    "Instrumental": "extracted_audio_instrumental",
                }

                voice_sample_audio = os.path.join(temp_dir, "voice_sample.wav")
                dubbed_audio = os.path.join(temp_dir, "dubbed_audio.wav")
                
                lang_suffix = f"_{self.source_lang_var.get()}_to_{self.target_lang_var.get()}"
                output_video = os.path.join(output_dir, f"{video_name}_dubbed{lang_suffix}.mp4")
                transcript_file = os.path.join(output_dir, f"{video_name}_transcript{lang_suffix}.json")
                voice_sample_file = os.path.join(output_dir, f"{video_name}_voice_sample{lang_suffix}.wav")

                clip = VideoFileClip(str(Path(video_file)))
                watermark = ImageClip("watermark.png").set_duration(clip.duration).set_position(lambda t: (clip.w - watermark.w - 5, clip.h - watermark.h - 10))

                separated_audio = None

                if self.separator_enabled.get():
                    self.separator.output_dir = temp_dir
                    separator_model = self.separator_model.get()
                    self.separator.load_model(separator_model)
                    self.log(f"загружена модель сепаратора: {separator_model}")

                self.log("шаг 0: добавляем вотермарку...")

                video_with_watermark = CompositeVideoClip([clip,watermark])
                video_with_watermark.write_videofile(temp_video)
                
                self.log("шаг 1: берем звук из видео...")
                if not self.extract_audio(video_file, temp_audio):
                    return
                self.log("экстракция звука из файла завершена.")

                if self.separator_enabled.get():
                    self.log("шаг 1.1: отделяем вокал от инструментала...")
                    separated_audio = self.separator.separate(temp_audio,temp_audio_names) 

                audio_source_temp = temp_audio_vocal if separated_audio else temp_audio
                audio_source_dubbed = temp_audio_vocal if separated_audio else dubbed_audio

                self.log("шаг 2: транскрипция звука...")
                transcript_result = self.transcribe_audio(audio_source_temp)

                if transcript_result is None:
                    return
                
                original_text = transcript_result['text']
                self.log(f"транскрипция завершена. текст: {original_text[:100]}...")
                
                reference_audio_for_cloning = None
                if voice_method == "video_clone":
                    self.log("шаг 3: извлекаем сэмпл голоса из видео...")
                    reference_audio_for_cloning = self.extract_voice_sample_from_video(
                        video_file, voice_sample_audio, transcript_result, temp_audio_vocal if separated_audio else None
                    )
                    if reference_audio_for_cloning:
                        shutil.copy2(voice_sample_audio, voice_sample_file)
                        self.log(f"сэмпл голоса сохранен: {voice_sample_file}")
                elif voice_method == "file_clone":
                    reference_audio_for_cloning = self.reference_audio_path.get()
                
                self.log("шаг 4: переводим текст...")
                translated_text = self.translate_text(
                    original_text, 
                    self.source_lang_var.get(), 
                    self.target_lang_var.get()
                )
                self.log(f"перевод закончен. переведенный текст: {translated_text[:100]}...")
                
                enhanced_transcript = {
                    'original': transcript_result,
                    'translated_text': translated_text,
                    'source_language': self.source_lang_var.get(),
                    'target_language': self.target_lang_var.get(),
                    'voice_sample_used': reference_audio_for_cloning is not None,
                    'voice_sample_path': voice_sample_file if voice_method == "video_clone" else None,
                    'settings': {
                        'whisper_model': self.model_var.get(),
                        'voice_method': voice_method,
                        'preserve_timing': self.preserve_timing_var.get(),
                        'chunk_translation': self.chunk_translation_var.get(),
                        'min_sample_length': self.min_sample_length_var.get(),
                        'voice_activity_detection': self.voice_activity_detection_var.get()
                    }
                }
                
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_transcript, f, indent=2, ensure_ascii=False)
                
                self.log("шаг 5: генерируем звук...")
                
                if voice_method in ["video_clone", "file_clone"] and reference_audio_for_cloning and self.voice_clone_model:
                    success = self.generate_dubbed_audio_clone(translated_text, audio_source_dubbed, reference_audio_for_cloning)
                else:
                    success = self.generate_dubbed_audio_standard(translated_text, audio_source_dubbed)

                if not success:
                    return
                
                self.log("генерация звука завершена")

                if separated_audio:
                    self.log("шаг 5.1: комбинируем вокал и инструментал...")
                    success = self.combine_audio(temp_audio_vocal,temp_audio_instrumental,dubbed_audio)

                    if not success:
                        return
                    
                    self.log("комбинирование завершено")
                
                self.log("шаг 6: комбинируем аудио с оригинальным видео...")
                if not self.combine_video_audio(temp_video, dubbed_audio, output_video):
                    return
                
                self.log("комбинирование завершено")
                
                self.log(f"дубляж завершен!")
                self.log(f"финальное видео: {output_video}")
                self.log(f"транскрипт файл: {transcript_file}")
                if voice_method == "video_clone" and reference_audio_for_cloning:
                    self.log(f"сэмпл голоса: {voice_sample_file}")
                
                self.update_status("все прошло успешно!")
                
                success_message = (
                    f"дубляж завершен!\n"
                    f"язык: {self.source_lang_var.get()} → {self.target_lang_var.get()}\n"
                    f"метод дубляжа: {voice_method}\n"
                    f"видео: {output_video}"
                )
                
                if voice_method == "video_clone" and reference_audio_for_cloning:
                    success_message += f"\nсэмпл голоса: {voice_sample_file}"
                
                messagebox.showinfo("успешно", success_message)
                
        except Exception as e:
            self.log(f"ошибка: {str(e)}")
            messagebox.showerror("ошибка", f"ошибка дубляжа: {str(e)}")
        finally:
            self.is_processing = False
            self.process_btn.config(state='normal')
            self.progress.stop()
    
    def start_dubbing_process(self):
        """начинаем перевод на отдельном потоке"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.progress.start()
        self.log_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.dubbing_process)
        thread.daemon = True
        thread.start()

def main():
    try:
        import whisper
        import pyttsx3
        from TTS.api import TTS
        import librosa
        import soundfile as sf
        from scipy import signal
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
        from audio_separator.separator import Separator
    except ImportError as e:
        print(f"отсутствуют пакеты. {e}")
        print("установите слеующие пакеты используя эту команду:")
        print("pip install openai-whisper pyttsx3 coqui-tts torch librosa soundfile scipy onnxruntime audio-separator moviepy==1.0.3")
        return
    
 
    translation_available = False
    try:
        from googletrans import Translator
        translation_available = True
        print("используем googletrans для перевода")
    except ImportError:
        try:
            from deep_translator import GoogleTranslator
            translation_available = True
            print("используем deep_translator для перевода")
        except ImportError:
            print("предупреждение: не найдены библиотеки перевода")
            print("установите одну из:")
            print("pip install googletrans==3.1.0a0")
            print("или")
            print("pip install deep-translator")
            print("программа будет работать без перевода")
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg нужен, но не найден")
        print("установите FFmpeg")
        return
        
    root = tk.Tk()
    app = dupellabs(root)
    root.mainloop()

if __name__ == "__main__":
    main()