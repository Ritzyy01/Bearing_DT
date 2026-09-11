"""
debug_color_test.py  —  Run this as a Script in Fusion 360
-----------------------------------------------------------
This does ONE thing: finds your bearing component and turns it RED.
No server, no JSON, no threads. Pure isolated test.

If the bearing turns red → the color logic works, issue is in the watcher.
If nothing happens → the component name is wrong or appearance API is failing.
The messageBox will tell you exactly what's happening at each step.
"""

import adsk.core
import adsk.fusion
import traceback

BEARING_COMPONENT_NAME = "Bearing"  # <-- change this if your component has a different name

def find_component(root, name, depth=0):
    """Search entire tree and return (component, full_path) or (None, '')"""
    for occ in root.occurrences:
        comp = occ.component
        path = ("  " * depth) + comp.name
        if comp.name == name:
            return comp, path
        found, found_path = find_component(comp, name, depth + 1)
        if found:
            return found, found_path
    return None, ""


def list_all_components(root, depth=0):
    """Return a string listing every component in the design tree."""
    result = ""
    for occ in root.occurrences:
        comp = occ.component
        result += ("  " * depth) + f"• {comp.name}\n"
        result += list_all_components(comp, depth + 1)
    return result


def run(context):
    app = adsk.core.Application.get()
    ui  = app.userInterface

    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        root   = design.rootComponent

        # ── Step 1: List all components ──────────────────────────────────────
        all_comps = list_all_components(root)
        if not all_comps:
            all_comps = "(none found — is a design open?)"

        ui.messageBox(
            f'All components in design tree:\n\n{all_comps}\n\n'
            f'Looking for: "{BEARING_COMPONENT_NAME}"\n'
            f'If your bearing is named differently, update BEARING_COMPONENT_NAME in this script.',
            "Step 1: Component Tree"
        )

        # ── Step 2: Find the bearing ──────────────────────────────────────────
        comp, path = find_component(root, BEARING_COMPONENT_NAME)

        if comp is None:
            ui.messageBox(
                f'❌ Component "{BEARING_COMPONENT_NAME}" NOT FOUND.\n\n'
                f'Please rename it in Fusion 360\'s model tree to match exactly,\n'
                f'then update BEARING_COMPONENT_NAME in both scripts.',
                "Step 2: FAILED"
            )
            return

        body_count = comp.bRepBodies.count
        ui.messageBox(
            f'✅ Found component: "{comp.name}"\n'
            f'   Bodies inside: {body_count}\n\n'
            f'Attempting to apply RED color now...',
            "Step 2: Component Found"
        )

        if body_count == 0:
            ui.messageBox(
                f'❌ Component "{comp.name}" has 0 bodies.\n'
                f'Color is applied to bodies, not components.\n'
                f'Try selecting a sub-component that actually contains geometry.',
                "Step 2: No Bodies"
            )
            return

        # ── Step 3: Try applying a color ─────────────────────────────────────
        # Method: use renderStyle (most reliable across Fusion versions)
        try:
            for body in comp.bRepBodies:
                # Create a plain red appearance directly
                appearance_name = "DEBUG_RED_TEST"
                existing = design.appearances.itemByName(appearance_name)

                if existing is None:
                    # Find any base appearance to copy from
                    base = None
                    for lib in app.materialLibraries:
                        try:
                            base = lib.appearances.itemByName("Plastic - Matte (White)")
                            if base:
                                break
                        except Exception:
                            continue

                    if base is None:
                        base = app.materialLibraries.item(0).appearances.item(0)

                    existing = design.appearances.addByCopy(base, appearance_name)

                # Try setting every color-related property
                set_any = False
                prop_list = ""
                for prop in existing.appearanceProperties:
                    prop_list += f"  {prop.name} [{prop.objectType}]\n"
                    if prop.objectType == adsk.core.ColorProperty.classType():
                        prop.value = adsk.core.Color.create(255, 0, 0, 255)  # Pure RED
                        set_any = True

                body.appearance = existing

            app.activeViewport.refresh()

            ui.messageBox(
                f'✅ Color applied to {body_count} body/bodies.\n\n'
                f'Properties found on appearance:\n{prop_list}\n'
                f'Color property set: {set_any}\n\n'
                f'👉 Check your bearing in the viewport — is it RED now?\n'
                f'If yes: the color system works. The issue was in the watcher script.\n'
                f'If no: reply with the properties list above.',
                "Step 3: Result"
            )

        except Exception as e:
            ui.messageBox(
                f'❌ Error applying appearance:\n{traceback.format_exc()}',
                "Step 3: FAILED"
            )

    except Exception:
        ui.messageBox(traceback.format_exc(), "Unexpected Error")
