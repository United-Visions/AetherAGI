// components/FileUploader.js - Handles file selection and preview chips

export class FileUploader {
    constructor(inputId, previewContainerId, onFileChange) {
        this.input = document.getElementById(inputId);
        this.previewContainer = document.getElementById(previewContainerId);
        this.onFileChange = onFileChange; // Callback when files change
        this.files = []; // Store File objects

        this.input.addEventListener('change', (e) => this.handleSelection(e));
    }

    trigger() {
        this.input.click();
    }

    handleSelection(event) {
        const newFiles = Array.from(event.target.files);
        this.files = [...this.files, ...newFiles];
        this.renderPreviews();
        this.input.value = ''; // Reset input so same file can be selected again
        if(this.onFileChange) this.onFileChange(this.files);
    }

    removeFile(index) {
        this.files.splice(index, 1);
        this.renderPreviews();
        if(this.onFileChange) this.onFileChange(this.files);
    }

    clear() {
        this.files = [];
        this.renderPreviews();
    }

    getFiles() {
        return this.files;
    }

    renderPreviews() {
        this.previewContainer.innerHTML = '';

        this.files.forEach((file, index) => {
            const chip = document.createElement('div');
            chip.className = 'file-chip';

            let iconClass = 'fa-file';
            if (file.type.startsWith('image/')) iconClass = 'fa-image';
            else if (file.type.startsWith('video/')) iconClass = 'fa-video';
            else if (file.type.startsWith('audio/')) iconClass = 'fa-music';

            chip.innerHTML = `
                <i class="fas ${iconClass}"></i>
                <span>${file.name}</span>
                <i class="fas fa-times remove-file" data-index="${index}"></i>
            `;

            chip.querySelector('.remove-file').addEventListener('click', (e) => {
                e.stopPropagation();
                this.removeFile(index);
            });

            this.previewContainer.appendChild(chip);
        });
    }
}
