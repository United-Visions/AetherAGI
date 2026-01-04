# Documentation Access Points

This document outlines where users can access AetherMind API documentation throughout the application.

## Primary Documentation Page

**URL:** `/documentation`
**Route:** `@app.route("/documentation")` in `app.py`
**Template:** `templates/documentation.html`

### Features:
- ✅ Sticky sidebar navigation with scroll spy
- ✅ Syntax-highlighted code examples (Python, Next.js, React, Node.js, React Native, Flutter)
- ✅ Comprehensive API authentication guide
- ✅ RBAC permissions matrix
- ✅ Rate limiting tables
- ✅ Security best practices
- ✅ Integration examples for all major platforms
- ✅ Responsive design with Tailwind CSS

## Access Links Throughout Site

### 1. Homepage Navigation (`index_home.html`)
**Location:** Top navigation bar
```html
<a href="{{ url_for('documentation') }}" class="nav-link">
  <i class="fas fa-book mr-1"></i>Docs
</a>
```
**Style:** Matches other nav links with hover effects

### 2. Chat Interface Header (`index.html`)
**Location:** Top-right header next to namespace indicator
```html
<a href="{{ url_for('documentation') }}" target="_blank" 
   style="font-size: 0.8em; color: #10b981; ...">
  <i class="fas fa-book"></i>
  <span>API Docs</span>
</a>
```
**Behavior:** Opens in new tab so users don't lose chat session
**Style:** Green color matching AGI branding, compact design

### 3. Pricing Page Footer (`pricing.html`)
**Location:** Bottom footer with other important links
```html
<a href="{{ url_for('documentation') }}" class="text-green-500 hover:text-green-400">
  <i class="fas fa-book mr-2"></i>API Documentation
</a>
```
**Context:** Grouped with "Try Chat" and "Home" links

### 4. Domain Landing Pages (`domain_legal.html`, etc.)
**Location:** Top navigation bar
```html
<a href="{{ url_for('documentation') }}" class="text-blue-400 hover:text-blue-300">
  <i class="fas fa-book mr-1"></i>Docs
</a>
```
**Context:** Between logo and pricing link

### 5. Documentation Page Self-Navigation
**Location:** Top navigation within docs page itself
```html
<div class="flex justify-between items-center mb-8">
  <a href="{{ url_for('home') }}">Home</a>
  <a href="{{ url_for('pricing') }}">Pricing</a>
  <a href="{{ url_for('documentation') }}">Documentation</a>
  <a href="{{ url_for('index') }}">Chat</a>
</div>
```

## User Journey Examples

### Journey 1: New User → Documentation
1. Land on homepage
2. Click "Docs" in top nav
3. Read Getting Started guide
4. Copy Python integration code
5. Return to homepage to sign up

### Journey 2: Active User → Documentation (from Chat)
1. Using chat interface
2. Need API integration info
3. Click "API Docs" in header (opens new tab)
4. Find React Native example
5. Close docs tab, continue chatting

### Journey 3: Pricing Research → Documentation
1. On pricing page comparing plans
2. Want to understand API details
3. Click "API Documentation" in footer
4. Review rate limiting for each tier
5. Navigate to pricing to upgrade

### Journey 4: Domain Exploration → Documentation
1. On legal domain page
2. Interested in integrating legal AGI into app
3. Click "Docs" in nav
4. Find Next.js integration example
5. Return to domain page, click "Start Free"

## Technical Implementation

### Flask Routes
```python
@app.route("/")
def home():
    return render_template("index_home.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")
```

### Template Structure
- **Base:** All pages use inline CSS or Tailwind CDN
- **Icons:** Font Awesome 6.0.0 for consistent book icon (`fa-book`)
- **Colors:** Green (`#10b981`) for AGI branding, Blue for domain pages
- **Responsive:** Mobile-friendly navigation with collapsible menus (to be implemented)

## Future Enhancements

### Phase 1 (Immediate)
- [ ] Add search functionality to documentation
- [ ] Add "Copy Code" buttons for all examples
- [ ] Mobile hamburger menu for small screens

### Phase 2 (Next Sprint)
- [ ] Version selector (v1.0, v2.0, etc.)
- [ ] Language preference toggle (show only Python examples, etc.)
- [ ] Dark/Light theme toggle
- [ ] Table of contents quick jump links

### Phase 3 (Future)
- [ ] Interactive API playground (try requests in browser)
- [ ] Auto-generated API reference from OpenAPI spec
- [ ] Video tutorials embedded in docs
- [ ] Community examples gallery

## Maintenance Notes

### When Adding New Pages
Always include documentation link in navigation:
```html
<a href="{{ url_for('documentation') }}" class="nav-link">
  <i class="fas fa-book"></i> Docs
</a>
```

### When Updating API
1. Update `docs/integration/API_AUTHENTICATION.md`
2. Sync changes to `documentation.html` code examples
3. Test all integration examples
4. Update version number in docs header

### When Supporting New Platform
1. Add section to `API_AUTHENTICATION.md`
2. Add navigation item to docs sidebar
3. Include syntax-highlighted code example
4. Test with `highlight.js` language support
5. Add to "Integrations" section of homepage

## Analytics Tracking (Future)

Recommended events to track:
- Documentation page views
- Section scroll depth (which sections are read most)
- Code copy button clicks
- External documentation links clicked
- Time spent on documentation
- Documentation → Sign-up conversion rate

## Support Integration

Documentation page includes:
- Discord community link
- Email support (dev@aethermind.ai)
- GitHub repository link
- Status page link (future)

## SEO Optimization

Documentation page includes:
```html
<title>AetherMind API Documentation - Integration Guide</title>
<meta name="description" content="Complete API documentation for integrating AetherMind AGI into Python, Next.js, React, Node.js, and mobile apps. Includes authentication, rate limiting, and RBAC.">
<meta name="keywords" content="AetherMind API, AGI integration, API documentation, Python SDK, Next.js integration">
```

## Accessibility

- ✅ Semantic HTML (nav, section, article tags)
- ✅ ARIA labels for navigation
- ✅ Keyboard navigation support
- ✅ High contrast code syntax highlighting
- ✅ Focus indicators on interactive elements
- ⏳ Screen reader optimization (Phase 2)
- ⏳ Skip to content link (Phase 2)
