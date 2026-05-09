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
import os
import time
import wave
from dataclasses import dataclass, field
from multiprocessing import Process, Queue as MPQueue
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import numpy as np
from pyrnnoise import RNNoise

import alsaaudio
from ovos_plugin_manager.templates.microphone import Microphone
from ovos_utils.log import LOG


def _preprocess_worker(input_queue, output_queue, sample_rate, cpu_core):
    try:
        try:
            os.sched_setaffinity(0, {cpu_core})
            LOG.info("Preprocess worker pinned to CPU core %s", cpu_core)
        except AttributeError:
            LOG.warning("CPU affinity is not supported on this platform")
        except Exception:
            LOG.exception("Failed to pin preprocess worker to CPU core %s", cpu_core)

        denoiser = RNNoise(sample_rate=sample_rate)

        while True:
            chunk_bytes = input_queue.get()
            if chunk_bytes is None:
                output_queue.put(None)
                break

            try:
                audio = (
                    np.frombuffer(chunk_bytes, dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )
                audio = audio.reshape(1, -1)

                denoised_chunks = []

                for speech_prob, denoised_frame in denoiser.denoise_chunk(audio):
                    if denoised_frame.ndim > 1:
                        denoised_chunks.append(denoised_frame.flatten())
                    mixed = (0.5 * denoised_chunks) + (0.5 * audio[:len(denoised_chunks)])
               
                if denoised_chunks:
                    denoised_audio = (
                        np.concatenate(mixed)
                        .astype(np.int16)
                        .tobytes()
                    )
                else:
                    denoised_audio = b""

                output_queue.put(denoised_audio)
            except Exception:
                LOG.exception("Failed to preprocess audio")
                output_queue.put(None)
    except Exception:
        LOG.exception("Unexpected error in preprocess worker")
        try:
            output_queue.put(None)
        except Exception:
            pass


@dataclass
class AlsaMicrophone(Microphone):
    device: str = "default"
    period_size: int = 1024
    timeout: float = 5.0

    audio_retries: int = 0
    audio_retry_delay: float = 0.0
    preprocess_cpu_core: int = 2
    _thread: Optional[Thread] = None
    _preprocess_process: Optional[Process] = None
    _queue: "Queue[Optional[bytes]]" = field(default_factory=Queue)
    _is_running: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_sample = 0.0
        self.sample_width = 2
        self.multiplier: float = 2.0
        self.sample_channels = 1
        self.input_sample_rate = 16000
        self.sample_rate = 16000
        self._remainder = np.array([], dtype=np.float32)

        self._queue = Queue()
        self._preprocess_input_queue = MPQueue()
        self._preprocess_output_queue = MPQueue()

    def start(self):
        assert self._thread is None, "Already started"
        assert self._preprocess_process is None, "Preprocess process already started"

        self._is_running = True

        self._preprocess_process = Process(
            target=_preprocess_worker,
            args=(
                self._preprocess_input_queue,
                self._preprocess_output_queue,
                self.sample_rate,
                self.preprocess_cpu_core,
            ),
            daemon=True,
        )
        self._preprocess_process.start()

        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def read_chunk(self) -> Optional[bytes]:
        assert self._is_running, "Not running"
        try:
            return self._queue.get(timeout=self.timeout)
        except Exception:
            return None

    def stop(self):
        self._is_running = False

        while not self._queue.empty():
            self._queue.get()

        self._queue.put_nowait(None)

        try:
            self._preprocess_input_queue.put(None)
        except Exception:
            pass

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._preprocess_process is not None:
            self._preprocess_process.join(timeout=2)
            if self._preprocess_process.is_alive():
                self._preprocess_process.terminate()
                self._preprocess_process.join()
            self._preprocess_process = None

    def _run(self):
        debug_file_path = "/tmp/debug_mic.wav"
        debug_file = wave.open(debug_file_path, "wb")
        debug_file.setnchannels(self.sample_channels)
        debug_file.setsampwidth(self.sample_width)
        debug_file.setframerate(self.sample_rate)

        try:
            assert self.sample_width in {2, 4}, (
                "Only 16-bit and 32-bit sample widths are supported"
            )

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
                        periodsize=960,
                    )

                    try:
                        full_chunk = bytes()

                        debug_buffer = bytearray()

                        bytes_per_second = (
                            self.sample_rate *
                            self.sample_width *
                            self.sample_channels
                        )

                        max_buffer_size = bytes_per_second * 15
                        debug_saved = False

                        while self._is_running:
                            mic_chunk_length, mic_chunk = mic.read()

                            if mic_chunk_length <= 0:
                                LOG.warning("Bad chunk length: %s", mic_chunk_length)
                                continue

                            if self.multiplier != 1.0:
                                mic_chunk = audioop.mul(
                                    mic_chunk, 2, self.multiplier
                                )
                            
                            self._preprocess_input_queue.put(mic_chunk)

                            try:
                                mic_chunk = self._preprocess_output_queue.get(
                                    timeout=self.timeout
                                )
                            except Empty:
                                LOG.warning("Timed out waiting for preprocessed audio")
                                continue
                            except Exception:
                                LOG.exception(
                                    "Failed to receive preprocessed audio"
                                )
                                continue

                            if mic_chunk is None:
                                continue

                            if self.multiplier != 1.0:
                                mic_chunk = audioop.mul(
                                    mic_chunk, 2, self.multiplier
                                )

                            if not debug_saved:
                                debug_buffer.extend(mic_chunk)

                                if len(debug_buffer) >= max_buffer_size:
                                    debug_file.writeframes(bytes(debug_buffer))
                                    LOG.info(
                                        "Debug opname van 10 seconden opgeslagen: %s",
                                        debug_file_path
                                    )
                                    debug_saved = True
                                    debug_buffer.clear()

                            full_chunk += mic_chunk
                            while len(full_chunk) >= self.chunk_size:
                                self._queue.put_nowait(full_chunk[: self.chunk_size])
                                full_chunk = full_chunk[self.chunk_size:]

                            time.sleep(0.001)
                    finally:
                        debug_file.close()
                        LOG.info(
                            "Debug opname opgeslagen in %s",
                            debug_file_path
                        )
                        mic.close()
                except Exception:
                    LOG.exception("Failed to open microphone")
                    time.sleep(0.001)
        except Exception:
            LOG.exception("Unexpected error in ALSA microphone thread")
