## Description

OpenVoiceOS Microphone plugin with DC Offset and high pass filter for mems microphones


## Configuration

If you need to customize the configuration, add/or merge below to your mycroft.conf

```python
"listener": {
    "microphone": {
        "module": "ovos-microphone-plugin-alsa",
        "device": "pulse",                  # name/alias of the device; default = "default"
        "period_size": 1024,                # frames per period; default = 1024
        "timeout": 5.0,                     # blocks x seconds and raises the Empty exception if no item was available within that time; default = 5.0
        "multiplier": 1.0,                  # Increase/decrease loudness of audio; default = 1.0
        "audio_retries": 0,                 # Number of times to retry listening; default = 0
        "audio_retry_delay": 0.0            # Seconds to wait between retries; default = 0.0
    }
}
```

## Install

`pip install ovos-microphone-plugin-alsa`
