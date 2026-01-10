"""
PlayCanvas Local Automation Server
Automates PlayCanvas Editor via browser control (Playwright)
Bypasses REST API limitations by directly controlling the web UI

Install: pip install playwright fastapi uvicorn loguru
Then: playwright install chromium
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
from playwright.async_api import async_playwright, Page, Browser
import os
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("playcanvas_automation.log", rotation="10 MB")

app = FastAPI(title="PlayCanvas Automation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global browser instance
browser: Optional[Browser] = None
page: Optional[Page] = None
playwright_instance = None


class LoginRequest(BaseModel):
    username: str
    password: str


class UploadScriptRequest(BaseModel):
    project_id: int
    script_name: str
    script_content: str


class CreateEntityRequest(BaseModel):
    project_id: int
    entity_name: str
    components: Optional[list] = []


class AttachScriptRequest(BaseModel):
    project_id: int
    entity_name: str
    script_name: str
    attributes: Optional[Dict[str, Any]] = {}


@app.on_event("startup")
async def startup():
    """Initialize browser on startup"""
    global playwright_instance, browser
    logger.info("Initializing Playwright...")
    try:
        playwright_instance = await async_playwright().start()
        # Launch with headless=False to let user see what's happening
        # Increase teardown timeout to avoid premature closing
        browser = await playwright_instance.chromium.launch(headless=False, slow_mo=50)
        logger.success("🌐 Browser automation ready")
    except Exception as e:
        logger.error(f"Failed to start browser: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup browser on shutdown"""
    logger.info("Shutting down browser automation...")
    if browser:
        await browser.close()
    if playwright_instance:
        await playwright_instance.stop()
    logger.success("Shutdown complete")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


async def handle_cookie_consent(page: Page):
    """
    Detects and accepts cookie consent banners (OneTrust) 
    found in the provided editor dump.
    """
    try:
        # Check for the OneTrust banner container
        if await page.is_visible('#onetrust-banner-sdk', timeout=2000):
            logger.info("🍪 Cookie banner detected. Attempting to accept...")
            # Click the specific accept button ID found in the dump
            await page.click('#onetrust-accept-btn-handler', timeout=3000)
            # Wait for it to disappear so it doesn't intercept clicks
            await page.wait_for_selector('#onetrust-banner-sdk', state='hidden', timeout=5000)
            logger.success("🍪 Cookie banner accepted and hidden")
    except Exception as e:
        logger.debug(f"Cookie consent check passed (not found or error): {str(e)}")


async def wait_for_editor_load(page: Page):
    """
    Waits for the PlayCanvas editor to fully load by checking specific
    loading indicators found in the DOM dump.
    """
    logger.debug("Waiting for editor loading indicators...")
    
    # 1. Wait for specific initial loading screen to disappear
    # The dump shows .progress-widget.hidden when done
    try:
        # Wait for the main progress widget to be hidden
        await page.wait_for_selector('.progress-widget.hidden', state='attached', timeout=30000)
    except:
        logger.warning("Could not verify .progress-widget is hidden. Proceeding anyway.")

    # 2. Wait for the main layout to appear
    # The dump confirms #layout-assets exists deep in the structure
    try:
        await page.wait_for_selector('#layout-assets', state='visible', timeout=45000)
        logger.success("✅ Editor #layout-assets visible")
    except:
        logger.warning("Editor layout wait timed out - attempting to proceed")


@app.get("/")
def root():
    logger.debug("Health check accessed")
    return {
        "service": "PlayCanvas Automation API",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/login")
async def login(req: LoginRequest):
    """
    Login to PlayCanvas account
    """
    global page
    logger.info(f"Attempting login for user: {req.username}")
    
    try:
        if not browser:
            logger.error("Browser not initialized")
            raise HTTPException(500, "Browser not initialized")
        
        # Create new page if needed
        if not page:
            logger.debug("Creating new browser page")
            page = await browser.new_page()
        
        # Go to login page
        logger.debug("Navigating to https://login.playcanvas.com/")
        await page.goto("https://login.playcanvas.com/", timeout=60000)
        await page.wait_for_load_state("networkidle")

        # Ensure cookie banner doesn't block interaction
        await handle_cookie_consent(page)
        
        # Check if we are already logged in (redirected to projects)
        if "playcanvas.com/projects" in page.url or "playcanvas.com/user" in page.url:
             logger.info("Already logged in!")
             return {"status": "success", "message": "Already logged in"}

        # Fill login form
        logger.debug("Filling login credentials")
        
        # Adaptable selector strategy for login form variants
        username_filled = False
        possible_selectors = [
            'input[name="username"]',     # Standard
            'input[name="email"]',        # Email variation
            'input[type="email"]',        # Type-based
            'input[placeholder*="Email"]' # Text-based
        ]
        
        for selector in possible_selectors:
            try:
                # Short timeout to check if this selector exists on the page
                if await page.is_visible(selector, timeout=2000):
                    logger.info(f"Found username field: {selector}")
                    await page.fill(selector, req.username)
                    username_filled = True
                    break
            except:
                continue
                
        if not username_filled:
            logger.warning("Could not find standard username field. Taking screenshot...")
            debug_shot = f"/tmp/login_form_layout_{os.getpid()}.png"
            await page.screenshot(path=debug_shot)
            logger.info(f"Saved layout debug screenshot to {debug_shot}")
            # Fallback to the original selector so it errors out with a clear message if we really can't find it
            await page.fill('input[name="username"]', req.username)

        # Handle password field (usually standard, but good to be safe)
        password_selector = 'input[name="password"]'
        if not await page.is_visible(password_selector, timeout=2000):
             if await page.is_visible('input[type="password"]'):
                 password_selector = 'input[type="password"]'
        
        await page.fill(password_selector, req.password)
        
        logger.debug("Clicking submit")
        await page.click('button[type="submit"]')
        
        # Wait for redirect to dashboard
        logger.debug("Waiting for login to complete...")
        try:
             # Wait for either the projects page OR the user profile (which you hit in the logs)
             await page.wait_for_url(r"**/(projects|user/.*)", timeout=30000, wait_until="domcontentloaded")
        except Exception as e:
             # If we timed out but the URL looks logged in, proceed
             current_url = page.url
             if "playcanvas.com/user/" in current_url:
                  logger.warning(f"Navigated to user profile instead of projects: {current_url}. Accepting as success.")
             elif "playcanvas.com/projects" in current_url:
                  logger.warning("Navigated to projects but timed out waiting for load. Accepting as success.")
             else:
                  raise e
        
        logger.success("Login successful")
        return {
            "status": "success",
            "message": "Logged in to PlayCanvas"
        }
        
    except Exception as e:
        logger.exception("Login process failed")
        # Take a screenshot on failure for debugging
        if page:
             error_screen = f"/tmp/login_fail_{os.getpid()}.png"
             await page.screenshot(path=error_screen)
             logger.error(f"Screenshot saved to {error_screen}")
        raise HTTPException(500, f"Login failed: {str(e)}")


@app.post("/open-editor")
async def open_editor(project_id: int):
    """
    Open the PlayCanvas editor for a project
    """
    global page
    logger.info(f"Opening editor for project: {project_id}")
    
    try:
        if not page:
            logger.error("Session not active (not logged in)")
            raise HTTPException(400, "Not logged in. Call /login first")
        
        # Navigate to project
        url = f"https://playcanvas.com/project/{project_id}/overview"
        logger.debug(f"Navigating to {url}")
        await page.goto(url, timeout=60000)
        await page.wait_for_load_state("networkidle")
        
        # Click "EDITOR" button
        logger.debug("Clicking EDITOR button")
        
        # Handle the case where the editor opens in a new tab
        async with page.context.expect_page() as new_page_info:
            try:
                # Try multiple selectors for the Editor button
                await page.click('text=EDITOR', timeout=5000)
            except:
                logger.warning("Could not find 'EDITOR' text, trying other selectors...")
                await page.click('.icon-editor', timeout=5000)
        
        # Get the new page (the editor)
        new_page = await new_page_info.value
        await new_page.wait_for_load_state()
        
        # Update our global page reference to point to the editor
        page = new_page
        logger.info(f"Switched to editor tab: {page.url}")
        
        # Determine if we need to handle cookies or loading
        await handle_cookie_consent(page)
        await wait_for_editor_load(page)
        
        logger.success(f"Editor opened for project {project_id}")
        return {
            "status": "success",
            "message": f"Editor opened for project {project_id}"
        }
        
    except Exception as e:
        logger.exception(f"Failed to open editor for project {project_id}")
        raise HTTPException(500, f"Failed to open editor: {str(e)}")


@app.post("/upload-script")
async def upload_script(req: UploadScriptRequest):
    """
    Create and upload a script to PlayCanvas project
    """
    global page
    logger.info(f"Uploading script: {req.script_name}")
    
    try:
        if not page:
            raise HTTPException(400, "Not logged in")

        # Ensure no overlays block us
        await handle_cookie_consent(page)
        
        # Make sure we're in the editor
        logger.debug("Waiting for assets panel...")
        try:
             # Try multiple possible selectors, prioritizing the specific ID found in analysis
             await page.wait_for_selector('#layout-assets, .pcui-asset-panel, .ui-panel.assets-panel', timeout=30000)
        except Exception:
             logger.warning("Could not find standard assets panel. Taking debug screenshot and DOM dump...")
             pid = os.getpid()
             await page.screenshot(path=f"/tmp/upload_script_fail_{pid}.png")
             with open(f"/tmp/editor_dump_{pid}.html", "w") as f:
                 f.write(await page.content())
             logger.info(f"Saved DOM dump to /tmp/editor_dump_{pid}.html")
             # Try a highly generic fallback
             pass
        
        # Right-click in assets panel
        # Prioritize the content area of the panel
        assets_panel = await page.query_selector('#layout-assets .pcui-panel-content') or \
                       await page.query_selector('#layout-assets') or \
                       await page.query_selector('.pcui-asset-panel .pcui-panel-content') or \
                       await page.query_selector('.ui-panel.assets-panel')

        if not assets_panel:
            logger.error("Assets panel not found in DOM")
            raise HTTPException(500, "Assets panel not found - Editor layout might have changed")
        
        # Take screenshot for debugging before attempting context menu
        debug_screenshot = f"/tmp/playcanvas_before_context_{os.getpid()}.png"
        await page.screenshot(path=debug_screenshot)
        logger.debug(f"Pre-context menu screenshot: {debug_screenshot}")
        
        logger.debug("Opening context menu on assets panel")
        await assets_panel.click(button='right')
        
        # Wait for context menu to appear - look for the menu container
        logger.debug("Waiting for context menu to appear...")
        try:
            # PlayCanvas uses pcui-menu for context menus
            await page.wait_for_selector('.pcui-menu, .pcui-contextmenu, [class*="context-menu"]', timeout=5000, state='visible')
            logger.debug("Context menu detected")
        except:
            logger.warning("Context menu selector not found, continuing anyway...")
        
        await page.wait_for_timeout(300)
        
        # Click "New Asset" -> "Script" - use more specific selectors
        logger.debug("Selecting New Asset -> Script")
        
        # Try multiple approaches for "New Asset"
        new_asset_clicked = False
        new_asset_selectors = [
            '.pcui-menu >> text=New Asset',
            '.pcui-contextmenu >> text=New Asset',
            'text=New Asset >> visible=true',
            '.pcui-menu-item:has-text("New Asset")',
            '[class*="menu"] >> text=New Asset'
        ]
        
        for selector in new_asset_selectors:
            try:
                if await page.locator(selector).count() > 0:
                    await page.locator(selector).first.click(timeout=3000)
                    new_asset_clicked = True
                    logger.debug(f"Clicked New Asset using: {selector}")
                    break
            except Exception as sel_err:
                logger.debug(f"Selector {selector} failed: {sel_err}")
                continue
        
        if not new_asset_clicked:
            # Fallback: try clicking anywhere we find "New Asset" text
            await page.click('text=New Asset', timeout=5000)
        
        await page.wait_for_timeout(400)
        
        # Now click "Script" in the submenu
        script_selectors = [
            '.pcui-menu >> text=Script',
            '.pcui-contextmenu >> text=Script',
            'text=Script >> visible=true',
            '.pcui-menu-item:has-text("Script")'
        ]
        
        script_clicked = False
        for selector in script_selectors:
            try:
                if await page.locator(selector).count() > 0:
                    await page.locator(selector).first.click(timeout=3000)
                    script_clicked = True
                    logger.debug(f"Clicked Script using: {selector}")
                    break
            except Exception as sel_err:
                logger.debug(f"Script selector {selector} failed: {sel_err}")
                continue
        
        if not script_clicked:
            await page.click('text=Script', timeout=5000)
        
        # Wait for script editor/dialog to open - try multiple selectors
        logger.debug("Waiting for code editor or script dialog...")
        editor_found = False
        editor_selectors = [
            '.code-editor',
            '.pcui-code-editor',
            '[class*="code-editor"]',
            '.monaco-editor',
            'input[placeholder*="script" i]',
            'input[placeholder*="name" i]',
            '.pcui-text-input input'
        ]
        
        for sel in editor_selectors:
            try:
                await page.wait_for_selector(sel, timeout=5000)
                editor_found = True
                logger.debug(f"Found editor element with: {sel}")
                break
            except:
                continue
        
        if not editor_found:
            # Take debug screenshot
            debug_shot = f"/tmp/script_editor_missing_{os.getpid()}.png"
            await page.screenshot(path=debug_shot)
            logger.warning(f"Script editor not found. Screenshot saved to {debug_shot}")
        
        # Set script name - try multiple input field patterns
        logger.debug(f"Setting script name to {req.script_name}")
        name_input_selectors = [
            'input[placeholder="Script Name"]',
            'input[placeholder*="name" i]',
            '.pcui-text-input input',
            'input[type="text"]:visible'
        ]
        
        name_set = False
        for sel in name_input_selectors:
            try:
                name_input = await page.query_selector(sel)
                if name_input:
                    await name_input.fill(req.script_name)
                    await page.keyboard.press('Enter')
                    name_set = True
                    logger.debug(f"Set script name using: {sel}")
                    break
            except:
                continue
        
        if not name_set:
            logger.warning("Could not find script name input field")
        
        await page.wait_for_timeout(1000)
        
        # Find code editor and paste content - try multiple approaches
        logger.debug("Injecting script content")
        
        # Escape backticks and special chars in script content for JS template literal
        escaped_content = req.script_content.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
        
        inject_result = await page.evaluate(f"""
            (function() {{
                // Try CodeMirror
                const cmEditor = document.querySelector('.code-editor');
                if (cmEditor && cmEditor.CodeMirror) {{
                    cmEditor.CodeMirror.setValue(`{escaped_content}`);
                    return 'codemirror';
                }}
                
                // Try Monaco editor
                if (typeof monaco !== 'undefined') {{
                    const editors = monaco.editor.getEditors();
                    if (editors.length > 0) {{
                        editors[0].setValue(`{escaped_content}`);
                        return 'monaco';
                    }}
                }}
                
                // Try ACE editor
                if (typeof ace !== 'undefined') {{
                    const aceEditors = document.querySelectorAll('.ace_editor');
                    if (aceEditors.length > 0) {{
                        const editor = ace.edit(aceEditors[0]);
                        editor.setValue(`{escaped_content}`);
                        return 'ace';
                    }}
                }}
                
                // Fallback: find any textarea
                const textarea = document.querySelector('textarea.code, textarea[class*="editor"]');
                if (textarea) {{
                    textarea.value = `{escaped_content}`;
                    textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    return 'textarea';
                }}
                
                return 'none';
            }})()
        """)
        logger.debug(f"Script injection method: {inject_result}")
        
        # Save (Ctrl+S)
        logger.debug("Saving script...")
        await page.keyboard.press('Control+S')
        await page.wait_for_timeout(1000)
        
        logger.success(f"Script '{req.script_name}' uploaded successfully")
        return {
            "status": "success",
            "message": f"Script '{req.script_name}' uploaded",
            "script_name": req.script_name
        }
        
    except Exception as e:
        logger.exception(f"Script upload failed for {req.script_name}")
        raise HTTPException(500, f"Script upload failed: {str(e)}")


@app.post("/create-entity")
async def create_entity(req: CreateEntityRequest):
    """
    Create a new entity in the scene
    """
    global page
    logger.info(f"Creating entity: {req.entity_name}")
    
    try:
        if not page:
            raise HTTPException(400, "Not logged in")

        # Ensure no overlays block us
        await handle_cookie_consent(page)
        
        # Right-click in hierarchy panel
        logger.debug("Waiting for hierarchy panel...")
        hierarchy = await page.query_selector('#layout-hierarchy .pcui-panel-content') or \
                    await page.query_selector('.pcui-hierarchy-panel .pcui-panel-content') or \
                    await page.query_selector('.ui-panel.hierarchy-panel')
        
        if not hierarchy:
             # Fallback to container if content not selectable
             hierarchy = await page.query_selector('#layout-hierarchy')

        if not hierarchy:
            raise HTTPException(500, "Hierarchy panel not found")
        
        await hierarchy.click(button='right')
        await page.wait_for_timeout(300)
        
        # Click "New Entity"
        logger.debug("Selecting New Entity")
        await page.click('text=New Entity')
        await page.wait_for_timeout(500)
        
        # Rename entity
        logger.debug(f"Renaming to {req.entity_name}")
        await page.keyboard.type(req.entity_name)
        await page.keyboard.press('Enter')
        
        logger.success(f"Entity '{req.entity_name}' created")
        return {
            "status": "success",
            "message": f"Entity '{req.entity_name}' created"
        }
        
    except Exception as e:
        logger.exception(f"Entity creation failed for {req.entity_name}")
        raise HTTPException(500, f"Entity creation failed: {str(e)}")


@app.post("/attach-script")
async def attach_script(req: AttachScriptRequest):
    """
    Attach a script to an entity
    """
    global page
    logger.info(f"Attaching script '{req.script_name}' to '{req.entity_name}'")

    # Ensure no overlays block us
    await handle_cookie_consent(page)
    
    try:
        if not page:
            raise HTTPException(400, "Not logged in")
        
        # Find and select the entity
        logger.debug(f"Selecting entity {req.entity_name}")
        await page.click(f'text={req.entity_name}')
        await page.wait_for_timeout(500)
        
        # In inspector panel, click "Add Component"
        logger.debug("Adding Script component")
        await page.click('text=Add Component')
        await page.wait_for_timeout(300)
        
        # Click "Script"
        await page.click('text=Script')
        await page.wait_for_timeout(500)
        
        # Click "Add Script" and select the script
        logger.debug("Selecting script from list")
        await page.click('text=Add Script')
        await page.wait_for_timeout(300)
        await page.click(f'text={req.script_name}')
        
        # Configure attributes if provided
        if req.attributes:
            logger.debug(f"Configuring attributes: {req.attributes}")
            for key, value in req.attributes.items():
                logger.debug(f"Setting {key}={value}")
                # Find input field for attribute
                input_field = await page.query_selector(f'input[data-attribute="{key}"]')
                if input_field:
                    await input_field.fill(str(value))
        
        logger.success(f"Attached script {req.script_name} to {req.entity_name}")
        return {
            "status": "success",
            "message": f"Script '{req.script_name}' attached to '{req.entity_name}'"
        }
        
    except Exception as e:
        logger.exception(f"Script attachment failed")
        raise HTTPException(500, f"Script attachment failed: {str(e)}")


@app.post("/deploy-aether-bridge/{project_id}")
async def deploy_aether_bridge_auto(project_id: int):
    """
    Fully automated AetherBridge deployment
    """
    logger.info(f"Starting auto-deployment for project {project_id}")
    try:
        # Read AetherBridge script
        bridge_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "playcanvas_aether_bridge.js"
        )
        logger.debug(f"Reading bridge script from {bridge_path}")
        
        with open(bridge_path, 'r') as f:
            script_content = f.read()
        
        # Step 1: Open editor
        await open_editor(project_id)
        
        # Step 2: Upload script
        await upload_script(UploadScriptRequest(
            project_id=project_id,
            script_name="aetherBridge",
            script_content=script_content
        ))
        
        # Step 3: Create GameManager entity
        await create_entity(CreateEntityRequest(
            project_id=project_id,
            entity_name="GameManager"
        ))
        
        # Step 4: Attach script
        await attach_script(AttachScriptRequest(
            project_id=project_id,
            entity_name="GameManager",
            script_name="aetherBridge",
            attributes={
                "apiUrl": "http://localhost:8000/v1/game/unity/state",
                "syncInterval": 500,
                "debugMode": True
            }
        ))
        
        logger.success("Auto-deployment completed successfully!")
        return {
            "status": "success",
            "message": "🧠 AetherBridge deployed automatically!",
            "project_id": project_id,
            "next_steps": [
                "Click Launch button in editor",
                "Open browser console (F12)",
                "Start AetherMind backend: ./start_backend.sh"
            ]
        }
        
    except Exception as e:
        logger.exception("Auto-deployment failed")
        raise HTTPException(500, f"Auto-deployment failed: {str(e)}")


@app.get("/screenshot")
async def screenshot():
    """Take screenshot of current page"""
    global page
    logger.info("Taking screenshot request")
    
    if not page:
        raise HTTPException(400, "Not logged in")
    
    screenshot_path = "/tmp/playcanvas_screenshot.png"
    await page.screenshot(path=screenshot_path)
    logger.info(f"Screenshot saved to {screenshot_path}")
    
    return {
        "status": "success",
        "path": screenshot_path
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
