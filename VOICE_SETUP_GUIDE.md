# 🔊 AetherMind Voice Setup Guide

## Quick Status Check

✅ Voice synthesis is **FULLY IMPLEMENTED** and ready to use!
✅ Uses **Edge TTS** (Microsoft's free neural voices - no API keys needed!)
✅ Already installed in your environment

## How to Enable Voice

### Method 1: In the Chat UI (Main App)

1. **Start the services:**
   ```bash
   # Terminal 1 - Backend
   ./start_backend.sh
   
   # Terminal 2 - Frontend
   ./start_frontend.sh
   ```

2. **Open the app:** http://127.0.0.1:5000

3. **Click the speaker button** 🔊 (bottom-right corner of the input box)
   - Icon changes from 🔇 (muted) to 🔊 (enabled)
   - You'll hear a test message: "Voice enabled! You can hear my responses now."

4. **Send a message** - AetherMind will speak the response!

### Method 2: Voice Test Page (Standalone)

For direct testing without the full app:

1. **Start backend only:**
   ```bash
   ./start_backend.sh
   ```

2. **Visit:** http://127.0.0.1:5000/voice-test

3. **Click any test button** to hear different voice profiles

## Available Voice Profiles

| Profile | Voice ID | Description |
|---------|----------|-------------|
| `default` | en-US-AriaNeural | Default female voice |
| `professional` | en-US-JennyNeural | Professional female (-5% speed) |
| `casual` | en-US-SaraNeural | Casual female (+5% speed) |
| `energetic` | en-US-AnaNeural | Energetic female (+10% speed, +5Hz pitch) |
| `calm` | en-US-AriaNeural | Calm voice (-10% speed, -2Hz pitch) |
| `male_professional` | en-US-GuyNeural | Professional male (-5% speed) |
| `male_casual` | en-US-ChristopherNeural | Casual male |
| `british_female` | en-GB-SoniaNeural | British female accent |
| `british_male` | en-GB-RyanNeural | British male accent |
| `australian_female` | en-AU-NatashaNeural | Australian female accent |
| `australian_male` | en-AU-WilliamNeural | Australian male accent |

## Troubleshooting

### "Voice not playing"

**Browser autoplay policy:**
- Most browsers block autoplay until user interaction
- Click the voice button FIRST, then send a message
- You should hear "Voice enabled!" to confirm it works

**Check browser console (F12):**
```javascript
// You should see these logs when voice is enabled:
🔊 [Voice] Speaking: ...
📡 [Voice] API response: ...
🎵 [Voice] Playing audio, size: ...
✅ [Voice] Playback complete
```

### "No audio heard"

1. **Check system volume** - make sure it's not muted
2. **Check browser audio permissions**
3. **Try the test page:** http://127.0.0.1:5000/voice-test
4. **Open browser console** and look for errors

### "API Key error"

If using the test page directly, you need an API key:

```javascript
// In browser console:
localStorage.setItem('aethermind_api_key', 'your_api_key_here');
```

Get your API key by logging in via GitHub OAuth at http://127.0.0.1:5000

## Backend API Endpoint

Voice synthesis is available via REST API:

```bash
POST /v1/voice/synthesize
Headers:
  x-api-key: your_api_key
  Content-Type: application/json

Body:
{
  "text": "Hello world!",
  "persona": "professional",  // optional
  "voice_id": "en-US-AriaNeural",  // optional, overrides persona
  "rate": "+0%",  // -50% to +100%
  "pitch": "+0Hz"  // pitch adjustment
}

Response:
{
  "success": true,
  "audio": "base64_encoded_mp3_data...",
  "format": "mp3",
  "voice_used": "en-US-AriaNeural"
}
```

## Testing from Command Line

```bash
# Activate virtual environment
source .venv/bin/activate

# Test voice synthesis directly
python -c "
import asyncio
from perception.voice_synthesizer import speak_base64

async def test():
    audio = await speak_base64('Hello from AetherMind!')
    print(f'Generated {len(audio)} bytes of audio')

asyncio.run(test())
"
```

## Architecture

```
Frontend (VoiceManager.js)
    ↓
API Request (/v1/voice/synthesize)
    ↓
Backend (orchestrator/main_api.py)
    ↓
Voice Synthesizer (perception/voice_synthesizer.py)
    ↓
Edge TTS (Microsoft)
    ↓
MP3 Audio (base64 encoded)
    ↓
Browser Playback
```

## Features

- ✅ **No API keys required** - Edge TTS is free
- ✅ **Multiple voices** - 14+ profiles
- ✅ **Persona-specific voices** - Different personas use different voices
- ✅ **Adjustable speed/pitch** - Customize the voice
- ✅ **Clean text processing** - Removes markdown, code blocks, etc.
- ✅ **Queue system** - Handles multiple speech requests
- ✅ **Browser autoplay handling** - Graceful fallback
- ✅ **Persistent settings** - Voice preferences saved in localStorage

## Next Steps

1. **Try it now:** Start the app and click the 🔊 button
2. **Test different voices:** Use the test page at /voice-test
3. **Customize:** Adjust rate/pitch in VoiceManager settings
4. **Integrate:** Voice works automatically with persona system

---

**Need help?** Check browser console (F12) for detailed logs.
