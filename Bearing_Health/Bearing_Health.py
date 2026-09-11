"""This file acts as the main module for this script."""

import traceback
import threading
import urllib.request
import json
import adsk.core
import adsk.fusion
# import adsk.cam

app = adsk.core.Application.get()
ui  = app.userInterface

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL    = "http://127.0.0.1:5000/"
POLL_INTERVAL = 2.0  # seconds

# ── Globals ───────────────────────────────────────────────────────────────────
_stop_event   = threading.Event()
_custom_event = None
_handler_refs = []
EVENT_ID      = "BearingHealthColorEvent"


# ── Colour mapping ─────────────────────────────────────────────────────────────
def health_to_color(score: float):
    score = max(0.0, min(100.0, float(score)))
    if score >= 70:
        t = (score - 70) / 30.0
        return 0, 220, int(t * 255 + (1 - t) * 80)
    elif score >= 40:
        t = (score - 40) / 30.0
        return int((1 - t) * 255), int(t * 220 + (1 - t) * 200), 0
    else:
        t = score / 40.0
        return 255, int(t * 200), 0


# ── Get bearing body ───────────────────────────────────────────────────────────
def get_body():
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        comp   = design.activeComponent
        if comp.bRepBodies.count > 0:
            return comp.bRepBodies.item(0)
        root = design.rootComponent
        if root.bRepBodies.count > 0:
            return root.bRepBodies.item(0)
        for occ in root.allOccurrences:
            if occ.component.bRepBodies.count > 0:
                return occ.component.bRepBodies.item(0)
    except Exception:
        pass
    return None


# ── Apply colour ───────────────────────────────────────────────────────────────
def apply_color(body: adsk.fusion.BRepBody, r: int, g: int, b: int):
    try:
        design   = adsk.fusion.Design.cast(app.activeProduct)
        app_name = "BearingHealthColor"

        existing = design.appearances.itemByName(app_name)
        if existing:
            existing.deleteMe()

        lib      = app.materialLibraries.itemByName("Fusion 360 Appearance Library")
        base_app = None
        for name in ["Steel - Satin", "Steel - Polished", "Aluminum - Satin"]:
            base_app = lib.appearances.itemByName(name)
            if base_app:
                break
        if base_app is None:
            base_app = lib.appearances.item(0)

        new_app = design.appearances.addByCopy(base_app, app_name)

        for i in range(new_app.appearanceProperties.count):
            prop = new_app.appearanceProperties.item(i)
            if prop.objectType == adsk.core.ColorProperty.classType():
                adsk.core.ColorProperty.cast(prop).value = \
                    adsk.core.Color.create(r, g, b, 255)
                break

        body.appearance = new_app
        app.log(f'[BearingDT] Colour applied: RGB({r},{g},{b}) — Health={round(100*(1-r/255),1) if r else 100}%')

    except Exception:
        app.log(f'apply_color failed:\n{traceback.format_exc()}')


# ── Fetch /current from server ─────────────────────────────────────────────────
def fetch_current():
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/current", timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        app.log(f'[BearingDT] fetch_current failed: {e}')
        return None


# ── Custom event handler (main thread) ────────────────────────────────────────
class HealthEventHandler(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            data  = json.loads(args.additionalInfo)
            score = float(data.get("health_score", 100))
            r, g, b = health_to_color(score)

            body = get_body()
            if body:
                apply_color(body, r, g, b)
            else:
                app.log('[BearingDT] No body found during update')

            # Status bar (bottom of Fusion window)
            rul    = data.get("rul_days")
            rpm    = data.get("rpm")
            day    = (data.get("day_index") or 0) + 1
            total  = data.get("day_total", 50)
            status = ("HEALTHY"   if score > 75 else
                      "DEGRADING" if score > 45 else
                      "WARNING"   if score > 15 else "CRITICAL")
            rul_str = f"{rul:.1f} days" if rul is not None else "estimating"
            rpm_str = f"{rpm:.1f} RPM"  if rpm else "—"

            ui.statusBarText = (
                f"[Bearing DT]  Day {day}/{total}  |  "
                f"Health: {score:.1f}%  |  {status}  |  "
                f"RUL: {rul_str}  |  Shaft: {rpm_str}"
            )

            app.activeViewport.refresh()
            app.log(f'[BearingDT] Day {day}/{total} | Health {score:.1f}% | {status} | RUL {rul_str}')

        except Exception:
            app.log(f'HealthEventHandler failed:\n{traceback.format_exc()}')


# ── Polling thread ─────────────────────────────────────────────────────────────
def poll_loop():
    app.log(f'[BearingDT] Polling {SERVER_URL}/current every {POLL_INTERVAL}s')
    while not _stop_event.is_set():
        data = fetch_current()
        if data and "health_score" in data:
            try:
                _custom_event.fire(json.dumps(data))
            except Exception as e:
                app.log(f'[BearingDT] fire failed: {e}')
        _stop_event.wait(POLL_INTERVAL)
    app.log('[BearingDT] Polling stopped')


# ── run() ─────────────────────────────────────────────────────────────────────
def run(_context: str):
    global _custom_event

    try:
        # 1. Ping server
        try:
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3)
        except Exception:
            ui.messageBox(
                f'Cannot reach server at:\n{SERVER_URL}\n\n'
                'Make sure server.py is running first.',
                'Connection Error',
                adsk.core.MessageBoxButtonTypes.OKButtonType,
                adsk.core.MessageBoxIconTypes.CriticalIconType
            )
            return

        # 2. Find body
        body = get_body()
        if body is None:
            ui.messageBox('No solid body found. Open your bearing model first.')
            return

        app.log(f'[BearingDT] Found body: {body.name}')

        # 3. Clean up any previously registered event from a prior run
        try:
            app.unregisterCustomEvent(EVENT_ID)
        except Exception:
            pass

        # 4. Register custom event
        _custom_event = app.registerCustomEvent(EVENT_ID)
        handler       = HealthEventHandler()
        _custom_event.add(handler)
        _handler_refs.clear()
        _handler_refs.append(handler)

        # 5. Do one immediate colour update before starting the thread
        data = fetch_current()
        if data and "health_score" in data:
            r, g, b = health_to_color(data["health_score"])
            apply_color(body, r, g, b)
            app.activeViewport.refresh()
            app.log(f'[BearingDT] Initial colour set for health {data["health_score"]}%')
        else:
            app.log('[BearingDT] No data yet from /current — start the simulation in the browser')

        # 6. Start polling thread
        _stop_event.clear()
        t = threading.Thread(target=poll_loop, daemon=True)
        t.start()

        # 7. Non-blocking confirmation — use app.log instead of messageBox
        #    so the main thread is NOT blocked and events can fire immediately
        app.log(
            f'[BearingDT] Monitor started!\n'
            f'  Body:   {body.name}\n'
            f'  Server: {SERVER_URL}\n'
            f'  Poll:   every {POLL_INTERVAL}s\n'
            f'  Check the TEXT COMMANDS panel for live updates.'
        )

        # Small non-blocking notification (disappears on its own)
        ui.statusBarText = f'[Bearing DT] Monitor running — polling {SERVER_URL}'

    except Exception:
        app.log(f'run() failed:\n{traceback.format_exc()}')