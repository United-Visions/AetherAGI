# GitHub OAuth Configuration Guide

## Overview

AetherMind's GitHub OAuth integration automatically detects the environment (development vs production) and uses the appropriate redirect URI. This allows seamless authentication whether you're running locally or deployed on Render.

## Environment Variables

Configure these variables in your `.env` file:

```env
# GitHub OAuth Credentials
GITHUB_CLIENT_ID=Iv23liiw0LIvKeWKMEd0
GITHUB_CLIENT_SECRET=your_secret_here

# Production Redirect URI (Render/Production)
GITHUB_REDIRECT_URI=https://aethermind-frontend.onrender.com/callback

# Development Redirect URI (Localhost)
GITHUB_REDIRECT_URI_DEV=http://127.0.0.1:5000/callback
```

## How It Works

The Flask application (`frontend_flask/app.py`) automatically detects the environment:

1. **Development Detection**: The system checks if the request comes from:
   - `127.0.0.1` (localhost IPv4)
   - `localhost` hostname

2. **Automatic URI Selection**:
   - **Local Development**: Uses `GITHUB_REDIRECT_URI_DEV`
   - **Production/Render**: Uses `GITHUB_REDIRECT_URI`

3. **Request Flow**:
   ```
   User clicks "Login with GitHub"
   ↓
   System detects request.host
   ↓
   Selects appropriate redirect URI
   ↓
   Redirects to GitHub OAuth
   ↓
   GitHub redirects back to correct callback URL
   ```

## GitHub App Configuration

You need to configure BOTH redirect URIs in your GitHub OAuth App settings:

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Select your OAuth App (or create a new one)
3. Add **both** redirect URIs:
   - `https://aethermind-frontend.onrender.com/callback` (Production)
   - `http://127.0.0.1:5000/callback` (Development)

## Testing

### Local Development
```bash
# Start the Flask app
python frontend_flask/app.py

# Visit: http://127.0.0.1:5000
# Click "Login with GitHub"
# System will use: http://127.0.0.1:5000/callback
```

### Production (Render)
```bash
# Deploy to Render with environment variables set
# Visit: https://aethermind-frontend.onrender.com
# Click "Login with GitHub"
# System will use: https://aethermind-frontend.onrender.com/callback
```

## Logging

The system logs which redirect URI is being used:

```
INFO - GitHub OAuth: Using redirect URI: http://127.0.0.1:5000/callback (local=True)
```

or

```
INFO - GitHub OAuth: Using redirect URI: https://aethermind-frontend.onrender.com/callback (local=False)
```

## Troubleshooting

### Issue: "Redirect URI mismatch" error from GitHub

**Solution**: Ensure both URIs are registered in your GitHub OAuth App settings.

### Issue: Wrong URI being selected

**Solution**: Check the `request.host` value in logs. The system uses:
- `GITHUB_REDIRECT_URI_DEV` if host contains `127.0.0.1` or `localhost`
- `GITHUB_REDIRECT_URI` for all other hosts

### Issue: Environment variable not loading

**Solution**: 
1. Verify `.env` file is in project root
2. Restart the Flask application
3. Check Render dashboard for environment variables in production

## Security Notes

- **Never commit** your `.env` file to Git
- Use different GitHub OAuth credentials for development and production
- Keep `GITHUB_CLIENT_SECRET` secure and rotate it periodically
- The `FERNET_KEY` is used to encrypt GitHub tokens in session cookies

## Advanced Configuration

If you need to test with a different local port:

```env
GITHUB_REDIRECT_URI_DEV=http://127.0.0.1:8080/callback
```

Then update your GitHub OAuth App to include this URI as well.
