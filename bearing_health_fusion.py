"""
Bearing Health Colour Updater — Fusion 360 Script
===================================================
Polls the Flask server for the current bearing health score
and updates the body's appearance colour in real-time.

HOW TO USE:
  1. Open your bearing model in Fusion 360
  2. Go to: Tools → Scripts and Add-Ins → Scripts → Add (+)
  3. Point it to this file
  4. Click Run

Make sure server.py is running before you run this script.
"""

import adsk.core
import adsk.fusion
import adsk.cam
import traceback
import threading
import urllib.request
import urllib.error
import json
import time
import math

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL   = "http://localhost:5000"   # change to RPi IP when on the Pi
POLL_INTERVAL = 2.0                      # seconds between health checks
# ─────────────────────────────────────────────────────────────────────────────

app = adsk.core.Application.get()
ui  = app.userInterface

# Global stop flag for the polling thread
_stop_event = threading.Event()
_handlers   = []


def health_to_rgb(score: float):
    """
    Map 0–100 health score to RGB (0.0–1.0 each).
      100  →  cyan   (0, 229, 255)  — healthy
       70  →  green  (0, 255, 100)
       50  →  yellow (255, 200, 0)  — degrading
       25  →  orange (255, 100, 0)  — warning
        0  →  red    (255, 40, 40)  — critical
    """
    score = max(0.0, min(100.0, score))

    if score >= 70:
        # cyan → green  (100–70 range mapped)
        t = (score - 70) / 30.0
        r = 0.0
        g = t * 0.9 + (1 - t) * 1.0
        b = t * 1.0 + (1 - t) * 0.39
    elif score >= 40:
        # green → yellow/orange
        t = (score - 40) / 30.0
        r = (1 - t) * 1.0
        g = t * 1.0 + (1 - t) * 0.78
        b = 0.0
    else:
        # orange → red
        t = score / 40.0
        r = 1.0
        g = t * 0.39
        b = 0.0

    return r, g, b


def fetch_health() -> dict | None:
    """Fetch latest health data from the /current endpoint."""
    try:
        url = f"{SERVER_URL}/current"
        req = urllib.request.urlopen(url, timeout=3)
        return json.loads(req.read().decode())
    except Exception:
        return None


def apply_colour(body: adsk.fusion.BRepBody, r: float, g: float, b: float):
    """Apply an override appearance with the given RGB to a BRepBody."""
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)

        # Build or reuse a custom appearance named 'BearingHealth'
        appearances = design.appearances
        app_name    = "BearingHealth"

        # Remove previous custom appearance so colour updates cleanly
        existing = appearances.itemByName(app_name)
        if existing:
            existing.deleteMe()

        # Clone from a base material in the Fusion library
        # 'Steel - Satin' is a reliable base available in all Fusion installs
        lib     = app.materialLibraries.itemByName("Fusion 360 Appearance Library")
        base    = lib.appearances.itemByName("Steel - Satin")
        if base is None:
            # Fallback: use first available appearance
            base = lib.appearances.item(0)

        new_app = appearances.addByCopy(base, app_name)

        # Set diffuse colour property
        props = new_app.appearanceProperties
        for i in range(props.count):
            prop = props.item(i)
            if prop.objectType == adsk.core.ColorProperty.classType():
                colour = adsk.core.Color.create(
                    int(r * 255), int(g * 255), int(b * 255), 255
                )
                adsk.core.ColorProperty.cast(prop).value = colour
                break

        body.appearance = new_app

    except Exception:
        pass   # silently ignore appearance errors during polling


def get_bearing_body() -> adsk.fusion.BRepBody | None:
    """Return the first solid body in the active component."""
    try:
        design    = adsk.fusion.Design.cast(app.activeProduct)
        component = design.activeComponent
        if component.bRepBodies.count > 0:
            return component.bRepBodies.item(0)
        # If in assembly context, try root component
        root = design.rootComponent
        if root.bRepBodies.count > 0:
            return root.bRepBodies.item(0)
        # Walk occurrences
        for occ in root.allOccurrences:
            if occ.component.bRepBodies.count > 0:
                return occ.component.bRepBodies.item(0)
    except Exception:
        pass
    return None


def update_ui_text(score: float, rpm, rul):
    """Show current stats in the Fusion 360 status bar / text box."""
    rul_str = f"{rul:.1f} days" if rul is not None else "estimating…"
    rpm_str = f"{rpm:.1f} RPM" if rpm else "—"
    status  = "HEALTHY" if score > 75 else "DEGRADING" if score > 45 else "WARNING" if score > 15 else "CRITICAL"
    ui.statusBarText = (
        f"[Bearing DT]  Health: {score:.1f}%  |  {status}  |  "
        f"RUL: {rul_str}  |  Shaft: {rpm_str}"
    )


# ── Polling thread ────────────────────────────────────────────────────────────
def poll_loop():
    while not _stop_event.is_set():
        data = fetch_health()
        if data and "health_score" in data:
            score = data["health_score"]
            rpm   = data.get("rpm")
            rul   = data.get("rul_days")
            r, g, b = health_to_rgb(score)

            # Fusion API calls must happen on the main thread via a custom event
            args = {"r": r, "g": g, "b": b, "score": score, "rpm": rpm, "rul": rul}
            custom_event.fire(json.dumps(args))

        _stop_event.wait(POLL_INTERVAL)


# ── Custom event (bridges thread → main Fusion thread) ───────────────────────
class ColourEventHandler(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            data    = json.loads(args.additionalInfo)
            body    = get_bearing_body()
            if body:
                apply_colour(body, data["r"], data["g"], data["b"])
            update_ui_text(data["score"], data.get("rpm"), data.get("rul"))
            adsk.core.Application.get().activeViewport.refresh()
        except Exception:
            pass

EVENT_ID = "BearingHealthEvent"
custom_event   = app.registerCustomEvent(EVENT_ID)
handler        = ColourEventHandler()
custom_event.add(handler)
_handlers.append(handler)


# ── Stop command (button in the toolbar) ─────────────────────────────────────
class StopHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        _stop_event.set()
        ui.messageBox("Bearing health monitor stopped.", "Bearing DT")

class StopTerminateHandler(adsk.core.ApplicationCommandEventHandler):
    def notify(self, args):
        _stop_event.set()


# ── Entry point ───────────────────────────────────────────────────────────────
def run(context):
    try:
        # Check server reachability
        try:
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3)
        except Exception:
            ui.messageBox(
                f"Cannot reach server at {SERVER_URL}\n\n"
                "Make sure server.py is running, then try again.",
                "Connection Error"
            )
            return

        body = get_bearing_body()
        if body is None:
            ui.messageBox(
                "No solid body found in the active component.\n"
                "Open your bearing model first.",
                "No Body Found"
            )
            return

        ui.messageBox(
            f"Bearing Health Monitor started.\n\n"
            f"Server: {SERVER_URL}\n"
            f"Body: {body.name}\n"
            f"Poll interval: {POLL_INTERVAL}s\n\n"
            f"The bearing will change colour based on health score:\n"
            f"  Cyan  = Healthy (100%)\n"
            f"  Green = Good\n"
            f"  Yellow = Degrading\n"
            f"  Orange = Warning\n"
            f"  Red   = Critical (0%)\n\n"
            f"Health score and RUL shown in status bar.\n"
            f"Close Fusion or run Stop script to halt.",
            "Bearing DT — Started"
        )

        # Start polling thread
        _stop_event.clear()
        t = threading.Thread(target=poll_loop, daemon=True)
        t.start()

    except Exception:
        ui.messageBox(traceback.format_exc())


def stop(context):
    """Called when Fusion unloads the script."""
    _stop_event.set()
    try:
        app.unregisterCustomEvent(EVENT_ID)
    except Exception:
        pass
    adsk.terminate()
