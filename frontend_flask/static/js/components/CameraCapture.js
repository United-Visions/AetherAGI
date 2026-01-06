// components/CameraCapture.js - Handles webcam capture for images and video

export class CameraCapture {
    constructor(buttonId, onCaptureComplete) {
        this.button = document.getElementById(buttonId);
        this.onCaptureComplete = onCaptureComplete;
        this.stream = null;
        this.videoElement = null;
        this.modal = null;

        this.button.addEventListener('click', () => this.openCamera());
        this.createModal();
    }

    createModal() {
        // Create camera modal
        this.modal = document.createElement('div');
        this.modal.className = 'camera-modal';
        this.modal.style.display = 'none';
        
        this.modal.innerHTML = `
            <div class="camera-modal-content">
                <div class="camera-header">
                    <h3><i class="fas fa-camera"></i> Camera</h3>
                    <button class="camera-close" id="camera-close-btn">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="camera-preview">
                    <video id="camera-video" autoplay playsinline></video>
                    <canvas id="camera-canvas" style="display: none;"></canvas>
                </div>
                <div class="camera-controls">
                    <button class="camera-btn-secondary" id="camera-switch-btn">
                        <i class="fas fa-sync-alt"></i> Switch Camera
                    </button>
                    <button class="camera-btn-primary" id="camera-capture-btn">
                        <i class="fas fa-camera"></i> Capture Photo
                    </button>
                    <button class="camera-btn-danger" id="camera-record-btn">
                        <i class="fas fa-video"></i> Record Video
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(this.modal);

        // Setup event listeners
        this.modal.querySelector('#camera-close-btn').addEventListener('click', () => this.closeCamera());
        this.modal.querySelector('#camera-capture-btn').addEventListener('click', () => this.capturePhoto());
        this.modal.querySelector('#camera-record-btn').addEventListener('click', () => this.toggleVideoRecording());
        this.modal.querySelector('#camera-switch-btn').addEventListener('click', () => this.switchCamera());
        
        // Click outside to close
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeCamera();
            }
        });

        this.videoElement = this.modal.querySelector('#camera-video');
        this.canvasElement = this.modal.querySelector('#camera-canvas');
        
        // Video recording state
        this.isRecordingVideo = false;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.currentFacingMode = 'user'; // 'user' or 'environment'
    }

    async openCamera() {
        try {
            console.log('📹 [CameraCapture] Opening camera...');
            
            const constraints = {
                video: {
                    facingMode: this.currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: true // Enable audio for video recording
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;
            this.modal.style.display = 'flex';
            
            console.log('✅ [CameraCapture] Camera opened successfully');
        } catch (error) {
            console.error('❌ [CameraCapture] Failed to access camera:', error);
            alert('Failed to access camera. Please check your permissions.');
        }
    }

    async switchCamera() {
        // Toggle between front and back camera
        this.currentFacingMode = this.currentFacingMode === 'user' ? 'environment' : 'user';
        console.log(`📹 [CameraCapture] Switching to ${this.currentFacingMode} camera`);
        
        // Close current stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        // Reopen with new facing mode
        await this.openCamera();
    }

    capturePhoto() {
        if (!this.stream) {
            console.error('❌ [CameraCapture] No camera stream available');
            return;
        }

        console.log('📸 [CameraCapture] Capturing photo...');
        
        const canvas = this.canvasElement;
        const video = this.videoElement;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
            const file = new File(
                [blob], 
                `photo_${Date.now()}.jpg`, 
                { type: 'image/jpeg' }
            );
            
            console.log(`✅ [CameraCapture] Photo captured. Size: ${file.size} bytes`);
            
            if (this.onCaptureComplete) {
                this.onCaptureComplete(file);
            }
            
            this.closeCamera();
        }, 'image/jpeg', 0.95);
    }

    async toggleVideoRecording() {
        if (this.isRecordingVideo) {
            this.stopVideoRecording();
        } else {
            await this.startVideoRecording();
        }
    }

    async startVideoRecording() {
        try {
            console.log('🎥 [CameraCapture] Starting video recording...');
            
            const mimeType = this.getSupportedVideoMimeType();
            this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
            this.recordedChunks = [];

            this.mediaRecorder.addEventListener('dataavailable', (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            });

            this.mediaRecorder.addEventListener('stop', () => {
                const videoBlob = new Blob(this.recordedChunks, { type: mimeType });
                console.log(`✅ [CameraCapture] Video recording complete. Size: ${videoBlob.size} bytes`);
                
                const videoFile = new File(
                    [videoBlob], 
                    `video_${Date.now()}.webm`, 
                    { type: mimeType }
                );

                if (this.onCaptureComplete) {
                    this.onCaptureComplete(videoFile);
                }
            });

            this.mediaRecorder.start();
            this.isRecordingVideo = true;
            this.updateRecordButtonState();
            
            console.log('✅ [CameraCapture] Video recording started');

        } catch (error) {
            console.error('❌ [CameraCapture] Failed to start video recording:', error);
        }
    }

    stopVideoRecording() {
        if (this.mediaRecorder && this.isRecordingVideo) {
            console.log('🎥 [CameraCapture] Stopping video recording...');
            this.mediaRecorder.stop();
            this.isRecordingVideo = false;
            this.updateRecordButtonState();
            this.closeCamera();
        }
    }

    updateRecordButtonState() {
        const recordBtn = this.modal.querySelector('#camera-record-btn');
        if (this.isRecordingVideo) {
            recordBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
            recordBtn.classList.add('recording');
        } else {
            recordBtn.innerHTML = '<i class="fas fa-video"></i> Record Video';
            recordBtn.classList.remove('recording');
        }
    }

    closeCamera() {
        console.log('📹 [CameraCapture] Closing camera...');
        
        // Stop recording if active
        if (this.isRecordingVideo) {
            this.stopVideoRecording();
        }
        
        // Stop all tracks
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.modal.style.display = 'none';
        console.log('✅ [CameraCapture] Camera closed');
    }

    getSupportedVideoMimeType() {
        const types = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        return 'video/webm';
    }
}
