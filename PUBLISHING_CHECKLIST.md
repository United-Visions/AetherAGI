# 🚀 AetherMind SDK Publishing Checklist

Complete checklist for publishing AetherMind SDKs to PyPI and npm.

## ✅ Pre-Publishing Checklist

### Python SDK
- [x] Package structure created (`sdk/python/aethermind/`)
- [x] All modules implemented (`client.py`, `models.py`, `exceptions.py`)
- [x] `setup.py` and `pyproject.toml` configured
- [x] README.md with examples
- [x] Package built successfully (`dist/aethermind-1.0.0.tar.gz`, `dist/aethermind-1.0.0-py3-none-any.whl`)
- [ ] Create PyPI account at https://pypi.org/account/register/
- [ ] Create TestPyPI account at https://test.pypi.org/account/register/
- [ ] Generate PyPI API token
- [ ] Test package locally

### JavaScript SDK
- [x] Package structure created (`sdk/javascript/src/`)
- [x] TypeScript client implemented (`index.ts`)
- [x] `package.json` configured
- [x] README.md with examples
- [ ] Build TypeScript (`npm run build`)
- [ ] Create npm account at https://www.npmjs.com/signup
- [ ] Login to npm (`npm login`)
- [ ] Test package locally

## 📦 Publishing Steps

### Python → PyPI

#### Step 1: Test Locally
```bash
cd sdk/python
.venv/bin/pip install dist/aethermind-1.0.0-py3-none-any.whl
python -c "from aethermind import AetherMindClient; print('✓ Works!')"
```

#### Step 2: Upload to TestPyPI
```bash
.venv/bin/twine upload --repository testpypi dist/*

# You'll be prompted:
# Username: __token__
# Password: pypi-xxxxx (your TestPyPI API token)
```

#### Step 3: Test Install from TestPyPI
```bash
pip install --index-url https://test.pypi.org/simple/ aethermind
python -c "from aethermind import AetherMindClient; print('Success!')"
```

#### Step 4: Upload to Production PyPI
```bash
.venv/bin/twine upload dist/*

# You'll be prompted:
# Username: __token__
# Password: pypi-xxxxx (your PyPI API token)
```

#### Step 5: Verify on PyPI
- Visit https://pypi.org/project/aethermind/
- Check README renders correctly
- Try: `pip install aethermind`

### JavaScript → npm

#### Step 1: Build Package
```bash
cd sdk/javascript
npm install
npm run build
```

#### Step 2: Test Locally
```bash
npm link
cd /tmp/test-project
npm link @aethermind/sdk
node -e "const {AetherMindClient} = require('@aethermind/sdk'); console.log('✓')"
```

#### Step 3: Login to npm
```bash
npm login
# Enter your npm username, password, and email
```

#### Step 4: Publish to npm
```bash
npm publish --access public
```

#### Step 5: Verify on npm
- Visit https://www.npmjs.com/package/@aethermind/sdk
- Check README renders correctly
- Try: `npm install @aethermind/sdk`

## 🧪 Post-Publishing Tests

### Python SDK Test
```bash
# Create fresh environment
python3 -m venv test-env
source test-env/bin/activate

# Install from PyPI
pip install aethermind

# Run hello example
python sdk/examples/hello_aethermind.py
```

### JavaScript SDK Test
```bash
# Create fresh project
mkdir test-project && cd test-project
npm init -y

# Install from npm
npm install @aethermind/sdk

# Run hello example
node ../sdk/examples/hello_aethermind.js
```

## 📝 API Credentials Setup

### Get PyPI API Token
1. Login to https://pypi.org/
2. Go to Account Settings → API Tokens
3. Click "Add API token"
4. Name: "AetherMind SDK"
5. Scope: "Entire account" or "Project: aethermind"
6. Copy token (starts with `pypi-`)

### Store Credentials (Option 1: .pypirc)
Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...your-token

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9y...your-token
```

Then publish without entering credentials:
```bash
twine upload dist/*
```

### Get npm Access Token
1. Login to https://www.npmjs.com/
2. Click your profile → Access Tokens
3. Generate New Token → Automation
4. Copy token

### Store npm Credentials
```bash
npm login
# Or manually create ~/.npmrc:
# //registry.npmjs.org/:_authToken=npm_xxxxxxxxxxxxx
```

## 🔄 Version Updates

When releasing new versions (e.g., 1.0.0 → 1.1.0):

### Python
1. Update `sdk/python/aethermind/__init__.py`:
   ```python
   __version__ = "1.1.0"
   ```

2. Update `sdk/python/pyproject.toml`:
   ```toml
   version = "1.1.0"
   ```

3. Rebuild and republish:
   ```bash
   rm -rf dist/
   python -m build
   twine upload dist/*
   ```

### JavaScript
1. Update `sdk/javascript/package.json`:
   ```json
   {
     "version": "1.1.0"
   }
   ```

2. Rebuild and republish:
   ```bash
   npm run build
   npm publish
   ```

## 📢 Post-Release Checklist

After successful publishing:

### Documentation
- [ ] Update main README.md with installation instructions
- [ ] Add badges (PyPI version, npm version, downloads)
- [ ] Update documentation site
- [ ] Update integration examples

### Git
- [ ] Commit SDK files
- [ ] Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
- [ ] Push with tags: `git push origin main --tags`
- [ ] Create GitHub Release with release notes

### Announcements
- [ ] Announce on Discord (#announcements)
- [ ] Tweet release announcement
- [ ] Post on Reddit (r/Python, r/javascript, r/MachineLearning)
- [ ] Submit to Hacker News ("Show HN: AetherMind SDK")
- [ ] Email Pro/Enterprise customers
- [ ] Update status page

### Monitoring
- [ ] Monitor PyPI download stats: https://pypistats.org/packages/aethermind
- [ ] Monitor npm download stats: https://www.npmjs.com/package/@aethermind/sdk
- [ ] Watch for GitHub issues
- [ ] Monitor error tracking (Sentry)
- [ ] Check Discord #sdk-support channel

## 🐛 Rollback Plan

If you need to remove a broken release:

### PyPI (Cannot delete, but can yank)
```bash
# Yank a release (hides it from pip install)
pip install pkginfo
python -m twine yank aethermind -v 1.0.0 -r "Broken release"
```

### npm
```bash
# Unpublish within 72 hours
npm unpublish @aethermind/sdk@1.0.0

# Or deprecate
npm deprecate @aethermind/sdk@1.0.0 "Broken release, use 1.0.1+"
```

## 📊 Success Metrics

Track these after publishing:

- **PyPI Downloads**: Target 1K+ in first week
- **npm Downloads**: Target 500+ in first week
- **GitHub Stars**: Monitor repository stars
- **Community**: Discord joins, GitHub issues
- **Integration**: Number of apps using SDK
- **Feedback**: User ratings and reviews

## ⚠️ Important Notes

1. **Cannot delete from PyPI** - Once published, versions cannot be deleted (only yanked)
2. **npm unpublish** - Only works within 72 hours
3. **Semantic Versioning** - Follow SemVer (MAJOR.MINOR.PATCH)
4. **Breaking Changes** - Always bump MAJOR version for breaking changes
5. **Changelogs** - Maintain CHANGELOG.md for all releases
6. **Security** - Never commit API tokens to git

## 🎯 Quick Commands

### Python One-Liner
```bash
cd sdk/python && python -m build && twine upload dist/*
```

### JavaScript One-Liner
```bash
cd sdk/javascript && npm run build && npm publish
```

---

**Status**: ✅ Packages built and ready to publish!

**Next Steps**:
1. Create PyPI and npm accounts
2. Generate API tokens
3. Run test publishes to TestPyPI/npm
4. Publish to production registries
5. Announce to community

**Questions?** Email dev@aethermind.ai or join Discord.
