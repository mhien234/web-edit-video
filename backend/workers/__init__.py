from . import beat_sync, denoise, remove_bg, setup_ffmpeg, subtitles, voiceover

HANDLERS = {
    "beat_sync": beat_sync.run,
    "denoise": denoise.run,
    "remove_bg": remove_bg.run,
    "setup_rembg": remove_bg.setup,
    "subtitles": subtitles.run,
    "setup_ai": subtitles.setup,
    "setup_ffmpeg": setup_ffmpeg.run,
    "voiceover": voiceover.run,
    "setup_tts": voiceover.setup,
}
