// main.js - Application Entry Point

import { api } from './api.js';
import { ChatInterface } from './components/ChatInterface.js';
import { ThinkingVisualizer } from './components/ThinkingVisualizer.js';
import { FileUploader } from './components/FileUploader.js';

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Components
    const chat = new ChatInterface('messages');
    const visualizer = new ThinkingVisualizer('visualizer-container');
    const fileUploader = new FileUploader('file-input', 'file-previews', (files) => {
        // Optional: Enable/Disable send button based on files
    });

    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const cameraBtn = document.getElementById('camera-btn');
    const micBtn = document.getElementById('mic-btn');
    const fileBtn = document.getElementById('file-btn');

    let messageHistory = [];

    // --- Handlers ---

    const handleSend = async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        const files = fileUploader.getFiles();

        if (!text && files.length === 0) return;

        // Clear Inputs
        chatInput.value = '';
        fileUploader.clear();

        // 1. Process Files First
        if (files.length > 0) {
            for (const file of files) {
                // UI: Show file sent
                chat.addFileAttachmentMessage('user', file.name, file.type);

                // Visualizer: Start Upload
                const card = visualizer.createCard(`Uploading ${file.name}`);

                try {
                    const result = await api.uploadFile(file);
                    card.append(`Analysis: ${result.analysis}`);
                    card.append(`Surprise: ${result.surprise}`);
                    card.setSuccess();

                    // Assistant Response regarding file
                    const metadata = { surprise: result.surprise, research: result.surprise > 0.5 };
                    chat.addMessage('assistant', `Analyzed ${file.name}: ${result.analysis}`, metadata);

                } catch (err) {
                    card.append(`Error: ${err.message}`);
                    card.setError();
                    chat.addMessage('assistant', `Failed to process ${file.name}.`);
                }
            }
        }

        // 2. Process Text
        if (text) {
            chat.addMessage('user', text);
            messageHistory.push({ role: 'user', content: text });

            const card = visualizer.createCard("Thinking Process");
            card.append("User input received.");
            card.append("Initiating active inference cycle...");

            try {
                const response = await api.sendMessage(messageHistory);
                const assistantMsg = response.choices[0].message;

                card.append("Response generated.");
                card.setSuccess();

                chat.addMessage('assistant', assistantMsg.content);
                messageHistory.push(assistantMsg);

            } catch (err) {
                card.append(`Error: ${err.message}`);
                card.setError();
                chat.addMessage('assistant', "I encountered an error processing your request.");
            }
        }
    };

    // --- Event Listeners ---

    chatForm.addEventListener('submit', handleSend);

    fileBtn.addEventListener('click', () => fileUploader.trigger());

    cameraBtn.addEventListener('click', () => {
        alert("Camera feature integration pending backend update.");
    });

    micBtn.addEventListener('click', () => {
        alert("Microphone feature integration pending backend update.");
    });
});
