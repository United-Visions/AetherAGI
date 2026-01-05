// ApiKeyModal.js - Modal for API key entry/management

console.log('🔑 [API_KEY_MODAL] Module loaded');

export class ApiKeyModal {
    constructor() {
        console.log('🔑 [API_KEY_MODAL] Initializing...');
        this.modal = null;
        this.onKeySubmit = null;
        this.createModal();
    }

    createModal() {
        console.log('🔑 [API_KEY_MODAL] Creating modal element...');
        
        // Create modal overlay
        this.modal = document.createElement('div');
        this.modal.id = 'api-key-modal';
        this.modal.className = 'api-key-modal-overlay';
        this.modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            animation: fadeIn 0.3s ease;
        `;

        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.className = 'api-key-modal-content';
        modalContent.style.cssText = `
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(31, 41, 55, 0.95) 100%);
            border: 2px solid rgba(16, 185, 129, 0.3);
            border-radius: 1rem;
            padding: 3rem;
            max-width: 600px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.4s ease;
        `;

        modalContent.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <i class="fas fa-key" style="font-size: 4rem; color: #10b981; margin-bottom: 1rem;"></i>
                <h2 style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 0.5rem;">
                    API Key Required
                </h2>
                <p style="color: rgba(156, 163, 175, 1); font-size: 1rem;">
                    Please enter your AetherMind API key to continue
                </p>
            </div>

            <div style="margin-bottom: 2rem;">
                <label style="display: block; color: rgba(209, 213, 219, 1); font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">
                    Your API Key
                </label>
                <input 
                    type="text" 
                    id="api-key-input" 
                    placeholder="am_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                    style="
                        width: 100%;
                        background: rgba(31, 41, 55, 0.9);
                        border: 2px solid rgba(75, 85, 99, 0.5);
                        color: white;
                        padding: 1rem;
                        border-radius: 0.5rem;
                        font-family: 'Courier New', monospace;
                        font-size: 0.95rem;
                        transition: all 0.3s;
                    "
                />
                <p id="key-error" style="color: #ef4444; font-size: 0.85rem; margin-top: 0.5rem; display: none;">
                    <i class="fas fa-exclamation-circle"></i> Invalid API key format
                </p>
            </div>

            <div style="margin-bottom: 2rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 0.5rem;">
                <div style="display: flex; align-items: start;">
                    <i class="fas fa-info-circle" style="color: #10b981; font-size: 1.2rem; margin-right: 0.75rem; margin-top: 0.2rem;"></i>
                    <div>
                        <p style="color: rgba(209, 213, 219, 1); font-size: 0.9rem; margin-bottom: 0.5rem;">
                            <strong>Don't have an API key?</strong>
                        </p>
                        <p style="color: rgba(156, 163, 175, 1); font-size: 0.85rem; margin-bottom: 0.5rem;">
                            You can generate one through the onboarding process or contact your administrator.
                        </p>
                        <a href="/" style="color: #10b981; text-decoration: none; font-size: 0.85rem; font-weight: 600;">
                            <i class="fas fa-arrow-left" style="margin-right: 0.25rem;"></i>
                            Go to Onboarding
                        </a>
                    </div>
                </div>
            </div>

            <div style="display: flex; gap: 1rem; justify-content: flex-end;">
                <button 
                    id="api-key-cancel-btn"
                    style="
                        padding: 0.75rem 1.5rem;
                        background: rgba(75, 85, 99, 0.5);
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s;
                        font-size: 1rem;
                    "
                    onmouseover="this.style.background='rgba(75, 85, 99, 0.7)'"
                    onmouseout="this.style.background='rgba(75, 85, 99, 0.5)'"
                >
                    Cancel
                </button>
                <button 
                    id="api-key-submit-btn"
                    style="
                        padding: 0.75rem 2rem;
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s;
                        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
                        font-size: 1rem;
                    "
                    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(16, 185, 129, 0.4)'"
                    onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(16, 185, 129, 0.3)'"
                >
                    <i class="fas fa-check" style="margin-right: 0.5rem;"></i>
                    Continue
                </button>
            </div>
        `;

        this.modal.appendChild(modalContent);
        document.body.appendChild(this.modal);

        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes slideUp {
                from { 
                    opacity: 0;
                    transform: translateY(30px);
                }
                to { 
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            #api-key-input:focus {
                outline: none;
                border-color: #10b981 !important;
                box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
            }
        `;
        document.head.appendChild(style);

        // Attach event listeners
        this.attachEventListeners();
        
        console.log('✅ [API_KEY_MODAL] Modal created and attached to DOM');
    }

    attachEventListeners() {
        const input = document.getElementById('api-key-input');
        const submitBtn = document.getElementById('api-key-submit-btn');
        const cancelBtn = document.getElementById('api-key-cancel-btn');
        const errorMsg = document.getElementById('key-error');

        // Submit on button click
        submitBtn.addEventListener('click', () => {
            console.log('🔑 [API_KEY_MODAL] Submit button clicked');
            this.handleSubmit();
        });

        // Submit on Enter key
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                console.log('🔑 [API_KEY_MODAL] Enter key pressed');
                this.handleSubmit();
            }
        });

        // Cancel button - Don't allow closing without valid API key
        cancelBtn.addEventListener('click', () => {
            console.log('🔑 [API_KEY_MODAL] Cancel button clicked');
            const hasValidKey = localStorage.getItem('aethermind_api_key');
            if (!hasValidKey) {
                console.warn('⚠️ [API_KEY_MODAL] Cannot close without a valid API key');
                const errorMsg = document.getElementById('key-error');
                errorMsg.textContent = 'An API key is required to use AetherMind. Please enter your key or get one from the onboarding page.';
                errorMsg.style.display = 'block';
                return;
            }
            this.hide();
        });

        // Prevent closing on overlay click if no API key
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                console.log('🔑 [API_KEY_MODAL] Overlay clicked');
                const hasValidKey = localStorage.getItem('aethermind_api_key');
                if (!hasValidKey) {
                    console.warn('⚠️ [API_KEY_MODAL] Cannot close without a valid API key');
                    const errorMsg = document.getElementById('key-error');
                    errorMsg.textContent = 'An API key is required to use AetherMind.';
                    errorMsg.style.display = 'block';
                    return;
                }
                this.hide();
            }
        });

        // Clear error on input
        input.addEventListener('input', () => {
            errorMsg.style.display = 'none';
        });

        console.log('✅ [API_KEY_MODAL] Event listeners attached');
    }

    handleSubmit() {
        const input = document.getElementById('api-key-input');
        const errorMsg = document.getElementById('key-error');
        const apiKey = input.value.trim();

        console.log('🔑 [API_KEY_MODAL] Validating API key...');

        // Validate API key format
        if (!apiKey) {
            console.warn('⚠️ [API_KEY_MODAL] Empty API key');
            errorMsg.textContent = 'Please enter your API key';
            errorMsg.style.display = 'block';
            return;
        }

        if (!apiKey.startsWith('am_live_')) {
            console.warn('⚠️ [API_KEY_MODAL] Invalid API key format');
            errorMsg.textContent = 'Invalid API key format. Key should start with "am_live_"';
            errorMsg.style.display = 'block';
            return;
        }

        if (apiKey.length < 20) {
            console.warn('⚠️ [API_KEY_MODAL] API key too short');
            errorMsg.textContent = 'API key is too short';
            errorMsg.style.display = 'block';
            return;
        }

        console.log('✅ [API_KEY_MODAL] API key validated');

        // Store API key
        localStorage.setItem('aethermind_api_key', apiKey);
        console.log('💾 [API_KEY_MODAL] API key stored in localStorage');

        // Call callback if provided
        if (this.onKeySubmit) {
            console.log('🔄 [API_KEY_MODAL] Calling onKeySubmit callback');
            this.onKeySubmit(apiKey);
        }

        // Hide modal
        this.hide();
        
        // Reload page to reinitialize with new key
        console.log('🔄 [API_KEY_MODAL] Reloading page...');
        window.location.reload();
    }

    show() {
        console.log('🔑 [API_KEY_MODAL] Showing modal');
        this.modal.style.display = 'flex';
        
        // Focus input after animation
        setTimeout(() => {
            document.getElementById('api-key-input').focus();
        }, 400);
    }

    hide() {
        console.log('🔑 [API_KEY_MODAL] Hiding modal');
        this.modal.style.display = 'none';
        
        // Clear input
        document.getElementById('api-key-input').value = '';
        document.getElementById('key-error').style.display = 'none';
    }

    setCallback(callback) {
        console.log('🔑 [API_KEY_MODAL] Setting callback');
        this.onKeySubmit = callback;
    }
}

console.log('✅ [API_KEY_MODAL] Module initialized');
