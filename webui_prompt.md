# Prompt for Generating Camera Control Web UI

**Role:** You are an expert Frontend Developer and IoT Systems Integrator.

**Task:** Create a React-based (or Vue/Svelte) Web UI Dashboard to remotely control a Raspberry Pi 5 camera system, with **Real-time Synchronization** across multiple connected clients. 

## Architecture context:
*   **Target Device (Running Camera):** A Raspberry Pi 5 running a Python script (`sender.py`).
*   **Host Device (Running UI):** Multiple devices (PCs, tablets, other Pis) opening the Web UI.
*   **Communication Protocol:** MQTT over WebSockets.
*   **The Synchronization Requirement:** If User A adjusts a parameter (e.g., changes the LensPosition slider), User B must instantly see that slider move to the new value on their screen.

## MQTT Specifications:
*   **Broker IP:** `10.10.10.199` (Configurable)
*   **WebSocket Port:** `9001` (Crucial for Web UI access)
*   **Camera Selection:** The system has multiple cameras (e.g. `cam0`, `cam1`). The UI should have a selector (Dropdown or Tabs) to allow users to switch between cameras. The MQTT topics used must be dynamically changed based on the selected camera.
    *   **Command Topic (Publisher):** `camera/<camera_name>/command` (e.g., `camera/cam0/command`)
    *   **State Topic (Publisher & Subscriber):** `camera/<camera_name>/state` (e.g., `camera/cam0/state`)
    *   **Status Topic (Subscriber):** `camera/<camera_name>/status` (e.g., `camera/cam0/status`)
*   *(Note: The `sender.py` scripts must also be re-configured to use these dynamic topic paths instead of generic paths so that the UI can individually target them).*

## Real-time Sync Logic (CRITICAL INSTRUCTION):
To achieve synchronization across all clients without a dedicated backend database, use the MQTT broker as the source of truth for the UI state.
1. When a client changes an input for a specific camera (e.g., changes `ExposureTime` to 15000 for `cam0`):
   *   Publish the *action* payload to `camera/cam0/command`: `{"ExposureTime": 15000}`
   *   Publish the *updated full state* to `camera/cam0/state` with `retain: true`: `{"ExposureTime": 15000, "AnalogueGain": 2.5, "ColourGains": [1.5, 1.2], "LensPosition": 3.0, "resolution": [2304, 1296]}`
2. All connected UI clients must subscribe to the `camera/+/state` topic (using the `+` wildcard for all cameras, or subscribe to individual cameras based on selection).
3. Upon receiving a message on a state topic (e.g. `camera/cam0/state`), the UI must update its local components (sliders, inputs) for that specific camera to match the incoming JSON payload.
4. Because the message is published with the `retain` flag, any *newly* connected UI client will immediately receive the last known state for each camera upon subscribing to `camera/+/state` and render correctly.

## UI Requirements (Control Panel):
The dashboard needs controls for these parameters:
1.  **Exposure Time:** Slider + Number Input. Key: `ExposureTime` (Range: 1000 - 50000)
2.  **Analogue Gain:** Slider + Number Input. Key: `AnalogueGain` (Range: 1.0 - 16.0)
3.  **Colour Gains:** Two Sliders (Red/Blue). Key: `ColourGains` (Format: `[red_gain, blue_gain]`, Range: 0.0 - 4.0)
4.  **Lens Position (Manual Focus):** Slider + Number Input. Key: `LensPosition` (Range: 0.0 - 15.0)
5.  **Resolution:** Dropdown `[width, height]`. Key: `resolution` (Examples: [1920, 1080], [2304, 1296])
6.  **Action Buttons:**
    *   **"Capture":** Send `{"action": "capture"}` to `camera/<camera_name>/command`. (Do not update `camera/<camera_name>/state` for actions).
    *   **"Restart/Shutdown":** Send `{"system": "restart"}` or `{"system": "shutdown"}` (require confirmation).

## UI Requirements (Telemetry Status):
Listen to `camera/+/status` (or dynamically subscribe to selected camera like `camera/cam0/status`) to display data arriving every 5 seconds.
Payload format: `{"cpu_temp": 55.2, "ram_usage_percent": 45.1, "cpu_usage_percent": 12.5, "resolution": [2304, 1296]}`
*Create a visual system health section (color-code CPU temp: Green < 60C, Yellow 60-80C, Red > 80C).*

## Tech Stack Requirements:
*   Use React (or a similar modern framework) because its state management (useState, useEffect) is perfectly suited for handling the incoming MQTT `camera/state` updates and triggering re-renders of the sliders.
*   Use `mqtt.js` for handling the WebSocket connection.
*   Make the UI look clean, industrial, and responsive (Tailwind CSS recommended). Ensure sliders don't jitter continuously when the user is actively dragging them (implement a debounce or separate local state while dragging vs. external state update).
