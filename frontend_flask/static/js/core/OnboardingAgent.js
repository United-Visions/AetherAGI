// OnboardingAgent.js - Interactive user understanding
// Gets to know the user through conversation, not forms

export class OnboardingAgent {
    constructor(shell) {
        this.shell = shell;
        this.state = 'init';
        this.profile = {
            name: null,
            preferences: {},
            domain: null,
            mediaOptIn: false,
            conversationStyle: 'friendly',
            goals: [],
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
        };
        this.askLaterQueue = [];
        this.conversationStage = 0;
        
        // Conversation flow stages
        this.stages = [
            'greeting',
            'name',
            'purpose',
            'domain_interest',
            'work_style',
            'media_request',
            'confirmation'
        ];
    }

    start() {
        console.log('🎯 [ONBOARDING] Starting personalized onboarding...');
        this.state = 'greeting';
        this.sendGreeting();
    }

    sendGreeting() {
        const hour = new Date().getHours();
        let timeGreeting = 'Hello';
        if (hour < 12) timeGreeting = 'Good morning';
        else if (hour < 17) timeGreeting = 'Good afternoon';
        else timeGreeting = 'Good evening';

        const greeting = `${timeGreeting}! 👋 I'm **AetherMind**, your personal AI assistant.

I'm not like other AI tools—I genuinely learn and adapt to you over time. The more we work together, the better I understand your needs.

Before we dive in, I'd love to get to know you a bit. This helps me tailor everything just for you.

**What should I call you?**`;

        this.shell.addMessage('assistant', greeting);
        this.state = 'awaiting_name';
    }

    async handleResponse(text) {
        console.log(`📝 [ONBOARDING] State: ${this.state}, Response: "${text.substring(0, 50)}..."`);

        switch (this.state) {
            case 'awaiting_name':
                await this.handleNameResponse(text);
                break;
            case 'awaiting_purpose':
                await this.handlePurposeResponse(text);
                break;
            case 'awaiting_domain':
                await this.handleDomainResponse(text);
                break;
            case 'awaiting_workstyle':
                await this.handleWorkstyleResponse(text);
                break;
            case 'awaiting_media':
                await this.handleMediaResponse(text);
                break;
            case 'awaiting_confirmation':
                await this.handleConfirmation(text);
                break;
            default:
                // Pass through to normal chat
                this.completeOnboarding();
        }
    }

    async handleNameResponse(text) {
        // Extract name (simple heuristic - first capitalized word or the whole thing)
        const name = this.extractName(text);
        this.profile.name = name;

        const response = `Nice to meet you, **${name}**! 🎉

I work best when I understand what you're trying to accomplish. 

**What brings you here today?** Are you looking to:
- Build something (apps, tools, websites)
- Research or learn about topics
- Get help with work tasks
- Just explore what I can do

Feel free to tell me in your own words!`;

        this.shell.addMessage('assistant', response);
        this.state = 'awaiting_purpose';
    }

    extractName(text) {
        // Clean common phrases
        const cleaned = text
            .replace(/^(i'?m|my name is|call me|it's|i am)\s*/i, '')
            .replace(/[.!?]$/, '')
            .trim();
        
        // Get first word if multiple
        const words = cleaned.split(/\s+/);
        const name = words[0];
        
        // Capitalize first letter
        return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
    }

    async handlePurposeResponse(text) {
        const purpose = text.toLowerCase();
        
        // Analyze intent
        let detectedDomain = 'general';
        let domainResponse = '';

        if (purpose.includes('build') || purpose.includes('app') || purpose.includes('code') || purpose.includes('develop') || purpose.includes('website')) {
            detectedDomain = 'code';
            domainResponse = `Ah, a builder! 🛠️ I love helping create things. I can write code, design architectures, and even deploy applications.`;
        } else if (purpose.includes('research') || purpose.includes('learn') || purpose.includes('study') || purpose.includes('understand')) {
            detectedDomain = 'research';
            domainResponse = `A curious mind! 🔬 I excel at deep research, synthesizing information, and explaining complex topics.`;
        } else if (purpose.includes('work') || purpose.includes('business') || purpose.includes('productivity')) {
            detectedDomain = 'business';
            domainResponse = `Productivity focused! 💼 I can help with documents, analysis, planning, and automating repetitive tasks.`;
        } else if (purpose.includes('legal') || purpose.includes('contract') || purpose.includes('law')) {
            detectedDomain = 'legal';
            domainResponse = `Legal matters! ⚖️ I can help analyze documents, research precedents, and draft content (though always consult a real lawyer for advice).`;
        } else if (purpose.includes('finance') || purpose.includes('money') || purpose.includes('invest') || purpose.includes('trading')) {
            detectedDomain = 'finance';
            domainResponse = `Financial focus! 📊 I can help with analysis, research, and modeling (not financial advice though).`;
        } else if (purpose.includes('explore') || purpose.includes('try') || purpose.includes('see what')) {
            detectedDomain = 'general';
            domainResponse = `Explorer mode! 🚀 I'm a generalist who can help with almost anything. Let's discover together!`;
        }

        this.profile.domain = detectedDomain;
        this.profile.goals.push(text);

        const response = `${domainResponse}

Here's what I can do for you:
- **Background tasks**: I work on things even when you're away
- **Tool creation**: I can build custom tools for your specific needs
- **Continuous learning**: I remember our conversations and get better over time

**How do you like to work?**
- Quick, direct answers
- Detailed explanations  
- Step-by-step guidance
- Let me figure out what's best for each situation`;

        this.shell.addMessage('assistant', response);
        this.state = 'awaiting_workstyle';
    }

    async handleDomainResponse(text) {
        // This stage might be skipped if we detect domain from purpose
        this.handleWorkstyleResponse(text);
    }

    async handleWorkstyleResponse(text) {
        const style = text.toLowerCase();
        
        if (style.includes('quick') || style.includes('direct') || style.includes('brief')) {
            this.profile.conversationStyle = 'concise';
        } else if (style.includes('detail') || style.includes('thorough') || style.includes('explain')) {
            this.profile.conversationStyle = 'detailed';
        } else if (style.includes('step') || style.includes('guide') || style.includes('walk')) {
            this.profile.conversationStyle = 'guided';
        } else {
            this.profile.conversationStyle = 'adaptive';
        }

        const styleAck = {
            'concise': "Got it—I'll keep things snappy and to the point! 🎯",
            'detailed': "Perfect—I'll make sure to explain things thoroughly! 📚",
            'guided': "Great—I'll walk you through things step by step! 🗺️",
            'adaptive': "Smart choice—I'll adapt my style based on what you need! 🧠"
        };

        const response = `${styleAck[this.profile.conversationStyle]}

One optional thing that helps me serve you better: **a quick photo or video of yourself**. 

This helps me:
- Recognize you across sessions
- Personalize visual content I create
- Remember context better

This is completely optional. Would you like to share a photo? Or we can skip this entirely—no pressure at all!`;

        this.shell.addInteractiveMessage('assistant', response, [
            { label: '📸 Take Photo', primary: true, handler: () => this.requestMedia('photo') },
            { label: '🎥 Record Video', handler: () => this.requestMedia('video') },
            { label: 'Skip for now', handler: () => this.skipMedia() }
        ]);

        this.state = 'awaiting_media';
    }

    requestMedia(type) {
        this.shell.ui.requestMedia(type, 'onboarding');
        this.profile.mediaOptIn = true;
    }

    skipMedia() {
        this.profile.mediaOptIn = false;
        this.shell.addMessage('user', 'Skip for now');
        
        const response = `No problem, ${this.profile.name}! I'll ask again sometime if that's okay—just in case you change your mind. 😊

Let me just confirm what I've learned:
- **Name**: ${this.profile.name}
- **Focus area**: ${this.profile.domain}
- **Style preference**: ${this.profile.conversationStyle}

Does this look right? Say "yes" to get started, or tell me what to change!`;

        this.shell.addMessage('assistant', response);
        this.state = 'awaiting_confirmation';
    }

    async handleMediaResponse(text) {
        // If they type something instead of clicking buttons
        const lower = text.toLowerCase();
        
        if (lower.includes('skip') || lower.includes('no') || lower.includes('later') || lower.includes("don't")) {
            this.skipMedia();
        } else if (lower.includes('photo') || lower.includes('picture')) {
            this.requestMedia('photo');
        } else if (lower.includes('video')) {
            this.requestMedia('video');
        } else {
            this.skipMedia();
        }
    }

    async handleConfirmation(text) {
        const lower = text.toLowerCase();
        
        if (lower.includes('yes') || lower.includes('correct') || lower.includes('right') || lower.includes('good') || lower.includes('perfect')) {
            await this.completeOnboarding();
        } else if (lower.includes('change') || lower.includes('no') || lower.includes('wrong')) {
            this.shell.addMessage('assistant', `No worries! What would you like me to change? Just tell me and I'll update it.`);
            this.state = 'awaiting_changes';
        } else {
            // Assume it's good
            await this.completeOnboarding();
        }
    }

    async completeOnboarding() {
        console.log('✅ [ONBOARDING] Completing onboarding with profile:', this.profile);
        
        // Save profile
        try {
            await this.shell.ui.api?.saveUserProfile(this.profile);
            localStorage.setItem('aethermind_profile', JSON.stringify(this.profile));
        } catch (err) {
            console.error('Failed to save profile:', err);
            localStorage.setItem('aethermind_profile', JSON.stringify(this.profile));
        }
        
        this.shell.isOnboarded = true;
        this.shell.userProfile = this.profile;
        
        // Final message
        const finalMessage = `Excellent! 🎉 We're all set, **${this.profile.name}**!

I'm now configured to help you with **${this.getDomainLabel(this.profile.domain)}** in a ${this.profile.conversationStyle} style.

A few things to know:
• I work on tasks in the background—even complex ones that take time
• I'll notify you when things are done
• I'm always learning from our conversations

**What would you like to do first?**`;

        this.shell.addMessage('assistant', finalMessage);
        
        // Show quick actions now that onboarding is complete
        this.shell.elements.quickActions.classList.remove('hidden');
    }

    getDomainLabel(domain) {
        const labels = {
            'code': 'software development',
            'research': 'research & learning',
            'business': 'business & productivity',
            'legal': 'legal matters',
            'finance': 'financial analysis',
            'general': 'general assistance'
        };
        return labels[domain] || 'general assistance';
    }

    // Called later to re-ask skipped questions
    async askLaterQuestions() {
        if (this.askLaterQueue.length === 0) return;
        
        const question = this.askLaterQueue.shift();
        
        if (question === 'media') {
            this.shell.addMessage('assistant', `Hey ${this.profile.name}! Remember when I asked about a photo? Would you be up for that now? It really helps me personalize things for you. No pressure though!`);
            this.shell.addInteractiveMessage('assistant', '', [
                { label: '📸 Sure, take photo', primary: true, handler: () => this.requestMedia('photo') },
                { label: 'Maybe later', handler: () => this.shell.addMessage('assistant', 'No problem! I\'ll check back another time. 😊') }
            ]);
        }
    }

    // Schedule periodic check-ins
    scheduleFollowUp() {
        // Check after 5 conversations or 1 hour
        setTimeout(() => {
            if (!this.profile.mediaOptIn && !this.profile.mediaDeclined) {
                this.askLaterQueue.push('media');
            }
        }, 60 * 60 * 1000); // 1 hour
    }
}
