/**
 * ResponseParser.js
 * Parses agent responses to extract thinking, actions, and clean content
 */

export class ResponseParser {
    constructor() {
        this.thinkingRegex = /<think>([\s\S]*?)<\/think>/gi;
        this.actionRegex = /<aether-(\w+)([^>]*)>([\s\S]*?)<\/aether-\1>/gi;
    }

    /**
     * Parse agent response into structured components
     * @param {string} rawResponse - Raw response from agent
     * @returns {Object} Parsed response with thinking, actions, and clean content
     */
    parse(rawResponse) {
        const result = {
            thinking: [],
            actions: [],
            cleanContent: rawResponse
        };

        // Extract thinking blocks
        let thinkMatch;
        while ((thinkMatch = this.thinkingRegex.exec(rawResponse)) !== null) {
            result.thinking.push({
                content: thinkMatch[1].trim(),
                timestamp: Date.now()
            });
        }

        // Extract action tags
        let actionMatch;
        const actionRegex = /<aether-(\w+)([^>]*)>([\s\S]*?)<\/aether-\1>/gi;
        while ((actionMatch = actionRegex.exec(rawResponse)) !== null) {
            const actionType = actionMatch[1]; // e.g., "sandbox", "write", "forge"
            const attributes = this.parseAttributes(actionMatch[2]);
            const content = actionMatch[3].trim();

            result.actions.push({
                type: actionType,
                attributes,
                content,
                displayName: this.getActionDisplayName(actionType),
                icon: this.getActionIcon(actionType),
                timestamp: Date.now()
            });
        }

        // Remove all tags from clean content
        result.cleanContent = rawResponse
            .replace(/<think>[\s\S]*?<\/think>/gi, '')
            .replace(/<aether-\w+[^>]*>[\s\S]*?<\/aether-\w+>/gi, '')
            .trim();

        return result;
    }

    /**
     * Parse XML-style attributes from tag
     * @param {string} attrString - Attribute string from tag
     * @returns {Object} Key-value pairs
     */
    parseAttributes(attrString) {
        const attrs = {};
        const attrRegex = /(\w+)="([^"]*)"/g;
        let match;

        while ((match = attrRegex.exec(attrString)) !== null) {
            attrs[match[1]] = match[2];
        }

        return attrs;
    }

    /**
     * Get display-friendly name for action type
     * @param {string} actionType - Action type (e.g., "sandbox")
     * @returns {string} Display name
     */
    getActionDisplayName(actionType) {
        const names = {
            'sandbox': 'Running Code',
            'write': 'Writing File',
            'forge': 'Creating Tool',
            'install': 'Installing Package',
            'research': 'Researching',
            'shell': 'Executing Command'
        };
        return names[actionType] || `Action: ${actionType}`;
    }

    /**
     * Get icon for action type
     * @param {string} actionType - Action type
     * @returns {string} Font Awesome icon class
     */
    getActionIcon(actionType) {
        const icons = {
            'sandbox': 'fa-flask',
            'write': 'fa-file-code',
            'forge': 'fa-hammer',
            'install': 'fa-download',
            'research': 'fa-search',
            'shell': 'fa-terminal'
        };
        return icons[actionType] || 'fa-cog';
    }

    /**
     * Stream parse for real-time response processing
     * @param {string} chunk - New chunk of response
     * @param {Object} state - Current parsing state
     * @returns {Object} Updated state and new content to display
     */
    streamParse(chunk, state = { buffer: '', inTag: false, tagStack: [] }) {
        state.buffer += chunk;
        const newContent = [];
        const newActions = [];
        const newThinking = [];

        // Check for complete thinking blocks
        const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
        let match;
        while ((match = thinkRegex.exec(state.buffer)) !== null) {
            newThinking.push({
                content: match[1].trim(),
                timestamp: Date.now()
            });
            // Remove from buffer
            state.buffer = state.buffer.replace(match[0], '');
        }

        // Check for complete action tags
        const actionRegex = /<aether-(\w+)([^>]*)>([\s\S]*?)<\/aether-\1>/g;
        while ((match = actionRegex.exec(state.buffer)) !== null) {
            const actionType = match[1];
            const attributes = this.parseAttributes(match[2]);
            const content = match[3].trim();

            newActions.push({
                type: actionType,
                attributes,
                content,
                displayName: this.getActionDisplayName(actionType),
                icon: this.getActionIcon(actionType),
                timestamp: Date.now()
            });
            // Remove from buffer
            state.buffer = state.buffer.replace(match[0], '');
        }

        // Content to display is what's not in tags
        const displayContent = state.buffer
            .replace(/<think>[\s\S]*$/i, '') // Remove incomplete thinking
            .replace(/<aether-\w+[\s\S]*$/i, '') // Remove incomplete actions
            .trim();

        return {
            state,
            content: displayContent,
            thinking: newThinking,
            actions: newActions
        };
    }
}
