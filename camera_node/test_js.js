const logs = `2026-03-19T10:17:06+07:00 wf51 systemd[1]: Started camera-sender@0.service - Camera Sender Service for Camera 0.
2026-03-19T10:17:06+07:00 wf51 python3[8628]: [2:14:30.330243042] [8628]  INFO Camera camera_manager.cpp:330 libcamera v0.5.2+99-bfd68f78
2026-03-19T10:17:06+07:00 wf51 python3[8628]: [2:14:30.353367409] [8637] ERROR V4L2 v4l2_device.cpp:390 'imx708_wide': Unable to set controls: Device or resource busy
2026-03-19T10:17:06+07:00 wf51 python3[8628]: camera __init__ sequence did not complete.`;

function formatLogLine(line) {
    if (!line.trim()) return '';
    const match = line.match(/^([0-9T:+-]+)\s+(\S+)\s+(.+?):\s+(.*)$/);
    if (match) {
        const [_, time, host, process, message] = match;
        let msgClass = "";
        let lowerMsg = message.toLowerCase();
        if (lowerMsg.includes("error") || lowerMsg.includes("fail") || lowerMsg.includes("critical")) {
            msgClass = "log-level-error";
        } else if (lowerMsg.includes("warn")) {
            msgClass = "log-level-warn";
        } else if (lowerMsg.includes("info")) {
            msgClass = "log-level-info";
        }
        return `<div class="log-line"><span class="log-time">[${time.split('T')[1].split('+')[0]}]</span> <span class="${msgClass}">${message}</span></div>`;
    }
    let msgClass = "";
    let lowerLine = line.toLowerCase();
    if (lowerLine.includes("error") || lowerLine.includes("fail") || lowerLine.includes("traceback")) {
        msgClass = "log-level-error";
    }
    return `<div class="log-line ${msgClass}">${line.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>`;
}

const lines = logs.split('\n');
let html = '';
for (const line of lines) {
    html += formatLogLine(line) + "\n";
}
console.log(html);
