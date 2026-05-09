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
import numpy as np
import wave
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from multiprocessing import Process, Queue as MPQueue
from typing import Optional
from pyrnnoise import RNNoise

import alsaaudio
from ovos_plugin_manager.templates.microphone import Microphone
from ovos_utils.log import LOG

@dataclass
class AlsaMicrophone(Microphone):
    device: str = "default"
    period_size: int = 960  # Optimaal voor RNNoise (2 frames van 480)
    timeout: float = 5.0
    multiplier: float = 1.0
    audio_retries: int = 0
    audio_retry_delay: float = 0.0
    _thread: Optional[Thread] = None
    _worker_process: Optional[Process] = None
    _is_running: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_width = 2
        self.sample_channels = 1
        self.sample_rate = 16000
        self.input_sample_rate = 16000
        
        # Queues voor communicatie tussen processen
        self._input_queue = MPQueue()  # Van Mic naar Denoiser
        self._output_queue = Queue()   # Van Denoiser naar Listener (Queue.get)

    def start(self):
        assert self._thread is None, "Already started"
        self._is_running = True
        
        # 1. Start de Denoiser Worker in een apart PROCES (andere CPU kern)
        self._worker_process = Process(target=self._denoise_worker, daemon=True)
        self._worker_process.start()
        
        # 2. Start de Microphone Reader in een aparte THREAD
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def read_chunk(self) -> Optional[bytes]:
        try:
            return self._output_queue.get(timeout=self.timeout)
        except:
            return None

    def _denoise_worker(self):
        """ Draait op een aparte CPU kern. """
        LOG.info("RNNoise worker process gestart.")
        # Initialiseer RNNoise binnen het proces
        denoiser = RNNoise(sample_rate=16000)
        
        while self._is_running:
            try:
                chunk_bytes = self._input_queue.get(timeout=1.0)
                if chunk_bytes is None:
                    break
                
                # Preprocessing
                audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_input = audio.reshape(1, -1)
                
                denoised_chunks = []
                for vad, frame in denoiser.denoise_chunk(audio_input):
                    # Terugschalen naar int16 bereik en flatten naar 1D
                    denoised_chunks.append((frame.flatten() * 32768.0))
                
                if denoised_chunks:
                    combined = np.concatenate(denoised_chunks)
                    # Clipping om overflow te voorkomen
                    final_bytes = np.clip(combined, -32768, 32767).astype(np.int16).tobytes()
                    self._output_queue.put(final_bytes)
                    
            except Exception:
                continue

    def _run(self):
        """ Microfoon reader thread. """
        try:
            mic = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE,
                rate=self.input_sample_rate,
                channels=self.sample_channels,
                format=alsaaudio.PCM_FORMAT_S16_LE,
                device=self.device,
                periodsize=self.period_size
            )
            
            full_chunk = bytes()
            
            while self._is_running:
                length, mic_chunk = mic.read()
                
                if length > 0:
                    # Pas multiplier toe (indien nodig) op ruwe data
                    if self.multiplier != 1.0:
                        mic_chunk = audioop.mul(mic_chunk, 2, self.multiplier)
                    
                    # Stuur naar de worker process via de input queue
                    self._input_queue.put(mic_chunk)
                    
                time.sleep(0) # Yield naar andere threads
                
        except Exception:
            LOG.exception("Fout in ALSA reader thread")
        finally:
            mic.close()

    def stop(self):
        self._is_running = False
        self._input_queue.put(None) # Stop de worker
        if self._worker_process:
            self._worker_process.join()
        if self._thread:
            self._thread.join()
