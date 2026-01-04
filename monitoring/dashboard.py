from fastapi import FastAPI
from .kill_switch import app as kill_app

app = FastAPI(title="Monitoring Dashboard")

# Mount the kill switch app.
# The kill_switch.py defines a route @app.post("/admin/kill")
# If we mount it at root "/", it will be available at "/admin/kill".
app.mount("/", kill_app)

@app.get("/")
async def root():
    return {"message": "Monitoring Dashboard"}
