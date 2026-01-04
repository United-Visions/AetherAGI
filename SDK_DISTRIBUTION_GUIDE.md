# AetherMind SDK Distribution Guide

Complete guide for making the AetherMind SDK available to developers across all platforms.

## Overview

The AetherMind SDK enables developers to integrate AGI capabilities into their applications with just a few lines of code. We provide official SDKs for:

- **Python** - `pip install aethermind`
- **JavaScript/TypeScript** - `npm install @aethermind/sdk`
- **React Native** - Same as JavaScript SDK
- **Flutter/Dart** - `flutter pub add aethermind`

## Directory Structure

```
aethermind_universal/
├── sdk/
│   ├── python/              # Python SDK
│   │   ├── aethermind/
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   ├── models.py
│   │   │   └── exceptions.py
│   │   ├── setup.py
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   └── PUBLISHING.md
│   │
│   ├── javascript/          # JS/TS SDK
│   │   ├── src/
│   │   │   └── index.ts
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── README.md
│   │
│   ├── flutter/             # Flutter/Dart SDK
│   │   ├── lib/
│   │   │   └── aethermind.dart
│   │   ├── pubspec.yaml
│   │   └── README.md
│   │
│   └── examples/            # Example integrations
│       ├── nextjs-chat/
│       ├── react-native-app/
│       ├── flask-api/
│       ├── fastapi-service/
│       └── django-integration/
```

## Python SDK Distribution

### 1. Build Package

```bash
cd sdk/python
python -m build
```

### 2. Test Locally

```bash
pip install dist/aethermind-1.0.0-py3-none-any.whl
python -c "from aethermind import AetherMindClient; print('✓ Works!')"
```

### 3. Publish to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

### 4. Publish to PyPI

```bash
twine upload dist/*
```

### 5. Verify

```bash
pip install aethermind
python -c "import aethermind; print(aethermind.__version__)"
```

## JavaScript SDK Distribution

### 1. Build Package

```bash
cd sdk/javascript
npm install
npm run build
```

### 2. Test Locally

```bash
npm link
cd /tmp/test-project
npm link @aethermind/sdk
```

### 3. Publish to npm

```bash
npm login
npm publish --access public
```

### 4. Verify

```bash
npm install @aethermind/sdk
node -e "const {AetherMindClient} = require('@aethermind/sdk'); console.log('✓')"
```

## Flutter SDK Distribution

### 1. Publish to pub.dev

```bash
cd sdk/flutter
flutter pub publish
```

### 2. Verify

```bash
flutter pub add aethermind
flutter pub get
```

## CDN Distribution (JavaScript)

For direct browser usage without npm:

### 1. Upload to CDN (jsDelivr, unpkg, or custom)

```html
<!-- Latest version -->
<script src="https://cdn.jsdelivr.net/npm/@aethermind/sdk@latest/dist/index.min.js"></script>

<!-- Specific version -->
<script src="https://cdn.jsdelivr.net/npm/@aethermind/sdk@1.0.0/dist/index.min.js"></script>
```

### 2. Example Usage

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/@aethermind/sdk@latest"></script>
</head>
<body>
  <script>
    const client = new AetherMind.AetherMindClient({
      apiKey: 'am_live_your_key'
    });
    
    client.chat({ message: 'Hello!' })
      .then(res => console.log(res.answer));
  </script>
</body>
</html>
```

## Documentation Distribution

### 1. Update Website

Add SDK documentation to `frontend_flask/templates/documentation.html`

### 2. Create GitHub Wiki

```bash
# Clone wiki
git clone https://github.com/United-Visions/AetherAGI.wiki.git

# Add pages
# - Home
# - Python-SDK
# - JavaScript-SDK
# - Flutter-SDK
# - API-Reference
# - Examples
```

### 3. README Badges

Add to repository README:

```markdown
[![PyPI](https://img.shields.io/pypi/v/aethermind.svg)](https://pypi.org/project/aethermind/)
[![npm](https://img.shields.io/npm/v/@aethermind/sdk.svg)](https://www.npmjs.com/package/@aethermind/sdk)
[![pub.dev](https://img.shields.io/pub/v/aethermind.svg)](https://pub.dev/packages/aethermind)
```

## Example Projects Distribution

### 1. Create Examples Repository

```bash
# Create separate repo for examples
gh repo create United-Visions/aethermind-examples --public
```

### 2. Structure

```
aethermind-examples/
├── python/
│   ├── flask-chatbot/
│   ├── fastapi-service/
│   └── django-integration/
├── javascript/
│   ├── nextjs-app/
│   ├── react-chat/
│   ├── express-api/
│   └── vanilla-js/
├── mobile/
│   ├── react-native/
│   └── flutter/
└── README.md
```

### 3. Deploy Examples

- Next.js → Vercel
- React → Netlify
- Python → Render
- Node.js → Railway

Provide live demo links in documentation.

## Integration Testing

### Automated Testing Workflow

Create `.github/workflows/sdk-test.yml`:

```yaml
name: SDK Integration Tests

on:
  push:
    paths:
      - 'sdk/**'
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install SDK
        run: pip install -e sdk/python
      - name: Run tests
        run: pytest sdk/python/tests/
        env:
          AETHERMIND_API_KEY: ${{ secrets.AETHERMIND_API_KEY }}

  test-javascript:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install SDK
        run: |
          cd sdk/javascript
          npm install
          npm run build
      - name: Run tests
        run: npm test
        env:
          AETHERMIND_API_KEY: ${{ secrets.AETHERMIND_API_KEY }}
```

## Version Management Strategy

### Semantic Versioning

- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes

### Release Process

1. **Update Version**
   - Python: `aethermind/__init__.py` and `pyproject.toml`
   - JavaScript: `package.json`
   - Flutter: `pubspec.yaml`

2. **Update CHANGELOG.md**
   ```markdown
   ## [1.1.0] - 2026-01-05
   ### Added
   - Streaming support for chat responses
   - New search_memory method
   ### Fixed
   - Rate limit handling
   ```

3. **Create Git Tag**
   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0"
   git push origin v1.1.0
   ```

4. **Publish Packages**
   ```bash
   # Python
   cd sdk/python && twine upload dist/*
   
   # JavaScript
   cd sdk/javascript && npm publish
   
   # Flutter
   cd sdk/flutter && flutter pub publish
   ```

5. **Create GitHub Release**
   - Go to Releases → Draft New Release
   - Select tag v1.1.0
   - Add release notes
   - Attach built artifacts

## Marketing & Distribution

### 1. Package Registry Profiles

#### PyPI Profile
- Add logo (512x512 px)
- Complete project description
- Add badges (build status, coverage, downloads)
- Link to documentation

#### npm Profile
- Add README with GIFs/screenshots
- Include example code
- Link to live demos

### 2. Developer Portals

Register on:
- **PyPI** - Python Package Index
- **npm** - Node Package Manager  
- **pub.dev** - Dart/Flutter packages
- **GitHub Packages** - Alternative registry
- **JitPack** - Java/Kotlin (future)

### 3. Community Promotion

- **Dev.to** - "Building with AetherMind SDK" tutorial
- **Medium** - "Integrating AGI into Your App" guide
- **YouTube** - SDK walkthrough video
- **Hacker News** - "Show HN: AetherMind Python SDK"
- **Reddit** - r/Python, r/javascript, r/FlutterDev
- **Discord** - Create #sdk-help channel
- **Twitter** - Announce releases with code examples

### 4. Documentation Sites

- **ReadTheDocs** - Python docs
- **GitHub Pages** - JavaScript docs
- **pub.dev docs** - Flutter docs
- **Postman** - API collection

## Support & Maintenance

### GitHub Issues Templates

Create `.github/ISSUE_TEMPLATE/sdk-bug.md`:

```markdown
---
name: SDK Bug Report
about: Report a bug in Python/JS/Flutter SDK
---

**SDK Version:**
- [ ] Python
- [ ] JavaScript/TypeScript
- [ ] Flutter

**Version Number:** (e.g., 1.0.0)

**Environment:**
- OS: (e.g., macOS 13, Ubuntu 22.04)
- Runtime: (e.g., Python 3.11, Node 18)

**Bug Description:**
[Clear description]

**Code to Reproduce:**
\```python
# Your code here
\```

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]
```

### SDK Support Channels

1. **Discord** - `#sdk-support` channel
2. **GitHub Discussions** - Q&A forum
3. **Stack Overflow** - Tag: `aethermind`
4. **Email** - dev@aethermind.ai (Pro/Enterprise)

## Distribution Checklist

Before releasing new SDK version:

### Pre-Release
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run all tests locally
- [ ] Build packages without errors
- [ ] Test install on clean environment
- [ ] Update documentation
- [ ] Review breaking changes

### Release
- [ ] Publish to TestPyPI/npm test registry
- [ ] Test install from test registry
- [ ] Publish to production registries
- [ ] Create Git tag
- [ ] Create GitHub release
- [ ] Update website documentation

### Post-Release
- [ ] Announce on Discord
- [ ] Tweet about release
- [ ] Update example projects
- [ ] Monitor error tracking
- [ ] Respond to GitHub issues
- [ ] Update integration guides

## Monitoring & Analytics

### Package Download Tracking

- **PyPI** - https://pypistats.org/packages/aethermind
- **npm** - https://www.npmjs.com/package/@aethermind/sdk
- **pub.dev** - Built-in analytics

### Error Tracking

Integrate Sentry or similar:

```python
# In SDK code
import sentry_sdk

sentry_sdk.init(
    dsn="your-dsn",
    environment="production"
)
```

### Usage Analytics

Track (anonymized):
- SDK version distribution
- Most used methods
- Error rates
- Average response times

## Future Enhancements

### Phase 2
- [ ] Ruby SDK
- [ ] Go SDK
- [ ] PHP SDK
- [ ] Rust SDK

### Phase 3
- [ ] GraphQL API option
- [ ] WebSocket streaming
- [ ] gRPC support
- [ ] CLI tool

### Phase 4
- [ ] Browser extension SDK
- [ ] VS Code extension
- [ ] JetBrains plugin
- [ ] Postman collection

---

**Questions?** Email dev@aethermind.ai or join our Discord.
