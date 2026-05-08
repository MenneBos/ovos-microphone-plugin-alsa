# Copyright 2022 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import audioop
import time
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Optional
import numpy as np
import os   # used to write data to file fo testing audio quality
import wave  # used to write data to file fo testing audio quality
from pyrnnoise import RNNoise

import alsaaudio
from ovos_plugin_manager.templates.microphone import Microphone
from ovos_utils.log import LOG


@dataclass
class AlsaMicrophone(Microphone):
    device: str = "default"
    period_size: int = 1024
    timeout: float = 5.0
    multiplier: float = 1.0
    audio_retries: int = 0
    audio_retry_delay: float = 0.0
    _thread: Optional[Thread] = None
    _queue: "Queue[Optional[bytes]]" = field(default_factory=Queue)
    _is_running: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.denoiser = RNNoise(sample_rate=48000)
        self._prev_sample = 0.0  # Voor high-pass context
        self.sample_width = 2
        self.sample_channels = 1
        self.input_sample_rate = 48000
        self.sample_rate = 16000
        self._remainder = np.array([], dtype=np.float32) # Voor RNNoise context
        self._queue = Queue()

    def start(self):
        assert self._thread is None, "Already started"
        self._is_running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def read_chunk(self) -> Optional[bytes]:
        assert self._is_running, "Not running"
        try:
            return self._queue.get(timeout=self.timeout)
        except: # let the listener handle this, and maybe restart the plugin
            return None

    def _preprocess_audio(self, chunk_bytes):
        audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
        # DC removal
        audio -= np.mean(audio)
        # snelle high-pass (vectorized)
        audio = np.append(
            audio[0],
            audio[1:] - 0.97 * audio[:-1]
        )
        #audio = np.diff(audio, prepend=audio[0]) * 0.97

        # 4. RNNoise & Downsampling
        # denoise_chunk splitst de data automatisch in 480-sample frames
        frames_out = []
        for vad, denoised in self.denoiser.denoise_chunk(audio):
            # Downsample direct naar 16k (elke 3e sample)
            frames_out.append(denoised[::3])
        if not frames_out:
            return None

        audio = np.concatenate(frames_out)
        # TERUGSCHALEN: Van float naar Int16 bereik
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        return audio.tobytes()


    def stop(self):
        self._is_running = False
        while not self._queue.empty():
            self._queue.get()
        self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self):
        # Debug: open een bestand om de bewerkte audio in op te slaan
        debug_file_path = "/tmp/debug_mic.wav"  # used to write data to file fo testing audio quality
        debug_file = wave.open(debug_file_path, "wb")  # used to write data to file fo testing audio quality
        debug_file.setnchannels(self.sample_channels)  # used to write data to file fo testing audio quality
        debug_file.setsampwidth(self.sample_width)  # used to write data to file fo testing audio quality
        debug_file.setframerate(self.sample_rate)  # used to write data to file fo testing audio quality
        
        # -----------------------------------------
        # RNNoise instance
        # -----------------------------------------
        self.denoiser = RNNoise(self.input_sample_rate)
        
        try:
            assert self.sample_width in {
                2,
                4,
            }, "Only 16-bit and 32-bit sample widths are supported"

            for _ in range(self.audio_retries + 1):
                try:
                    LOG.debug(
                        "Opening microphone (device=%s, rate=%s, width=%s, channels=%s)",
                        self.device,
                        self.sample_rate,
                        self.sample_width,
                        self.sample_channels,
                    )

                    mic = alsaaudio.PCM(
                        type=alsaaudio.PCM_CAPTURE,
                        rate=self.input_sample_rate,
                        channels=self.sample_channels,
                        format=alsaaudio.PCM_FORMAT_S16_LE,
                        device=self.device,
                        periodsize=480,
                    )

                    try:
                        full_chunk = bytes()

                        while self._is_running:
                            mic_chunk_length, mic_chunk = mic.read()
                            if mic_chunk_length <= 0:
                                LOG.warning("Bad chunk length: %s", mic_chunk_length)
                                continue
                            
                            # >>> PLAATS DEZE REGEL DIRECT NA mic.read()
                            mic_chunk = self._preprocess_audio(mic_chunk)

                            if mic_chunk is None:
                                continue
                            
                            # Increase loudness of audio
                            if self.multiplier != 1.0:
                                mic_chunk = audioop.mul(
                                    mic_chunk, 2, self.multiplier
                                )
                            
                            # Voeg dit toe in je loop voor debuggen
                            if len(mic_chunk) % 2 != 0:
                                LOG.error("CRITICAL: Oneven aantal bytes! Audio corruptie gegarandeerd.")                    
                            # Schrijf de bewerkte bytes weg naar het debug-bestand
                            debug_file.writeframes(mic_chunk)  # used to write data to file fo testing audio quality
                            
                            full_chunk += mic_chunk
                            while len(full_chunk) >= self.chunk_size:
                                self._queue.put_nowait(full_chunk[: self.chunk_size])
                                full_chunk = full_chunk[self.chunk_size:]

                            time.sleep(0.001)
                    finally:
                        debug_file.close()  # used to write data to file fo testing audio quality
                        LOG.info(f"Debug opname opgeslagen in {debug_file_path}")   # used to write data to file fo testing audio quality
                        mic.close()
                except Exception:
                    LOG.exception("Failed to open microphone")
                    time.sleep(0.001)
        except Exception:
            LOG.exception("Unexpected error in ALSA microphone thread")
