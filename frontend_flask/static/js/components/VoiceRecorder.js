// components/VoiceRecorder.js - Handles voice recording and audio capture

export class VoiceRecorder {
    constructor(buttonId, onRecordingComplete) {
        this.button = document.getElementById(buttonId);
        this.onRecordingComplete = onRecordingComplete;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;

        this.button.addEventListener('click', () => this.toggleRecording());
    }

    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    async startRecording() {
        try {
            console.log('🎤 [VoiceRecorder] Requesting microphone access...');
            
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                } 
            });

            // Determine the best MIME type for audio recording
            const mimeType = this.getSupportedMimeType();
            console.log(`🎤 [VoiceRecorder] Using MIME type: ${mimeType}`);

            this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
            this.audioChunks = [];

            this.mediaRecorder.addEventListener('dataavailable', (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            });

            this.mediaRecorder.addEventListener('stop', () => {
                const audioBlob = new Blob(this.audioChunks, { type: mimeType });
                console.log(`🎤 [VoiceRecorder] Recording complete. Size: ${audioBlob.size} bytes`);
                
                // Create a File object from the Blob
                const audioFile = new File(
                    [audioBlob], 
                    `recording_${Date.now()}.webm`, 
                    { type: mimeType }
                );

                if (this.onRecordingComplete) {
                    this.onRecordingComplete(audioFile);
                }

                // Stop all tracks
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            });

            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateButtonState();
            console.log('🎤 [VoiceRecorder] Recording started');

        } catch (error) {
            console.error('❌ [VoiceRecorder] Failed to start recording:', error);
            alert('Failed to access microphone. Please check your permissions.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            console.log('🎤 [VoiceRecorder] Stopping recording...');
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateButtonState();
        }
    }

    updateButtonState() {
        const icon = this.button.querySelector('i');
        if (this.isRecording) {
            icon.className = 'fas fa-stop-circle';
            this.button.style.color = '#ef4444'; // Red when recording
            this.button.classList.add('recording');
        } else {
            icon.className = 'fas fa-microphone';
            this.button.style.color = '';
            this.button.classList.remove('recording');
        }
    }

    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4',
            'audio/mpeg'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        return 'audio/webm'; // Fallback
    }
}
