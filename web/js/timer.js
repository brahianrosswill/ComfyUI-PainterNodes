import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "comfyui.painternodes.timer",

    setup() {
        function getLocale() {
            try {
                if (app.ui && app.ui.settings && app.ui.settings.getSettingValue) {
                    return app.ui.settings.getSettingValue("Comfy.Locale", "en") || "en";
                }
            } catch (e) {}
            try {
                const raw = localStorage.getItem("Comfy.Settings.Comfy.Locale");
                if (raw) return JSON.parse(raw);
            } catch (e) {}
            return "en";
        }

        function isZh() {
            const loc = getLocale().toLowerCase();
            return loc.startsWith("zh");
        }

        function t(zhText, enText) {
            return isZh() ? zhText : enText;
        }

        const timerContainer = document.createElement("div");
        timerContainer.id = "painter-workflow-timer";
        Object.assign(timerContainer.style, {
            position: "fixed",
            top: "12px",
            right: "12px",
            zIndex: "99999",
            display: "none",
            alignItems: "center",
            gap: "8px",
            background: "rgba(20, 20, 30, 0.92)",
            backdropFilter: "blur(8px)",
            border: "1px solid rgba(100, 180, 255, 0.3)",
            borderRadius: "10px",
            padding: "8px 16px",
            fontFamily: "'Segoe UI', 'PingFang SC', sans-serif",
            fontSize: "14px",
            color: "#e0e8f0",
            boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
            userSelect: "none",
            transition: "opacity 0.3s ease",
        });

        const dot = document.createElement("span");
        Object.assign(dot.style, {
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            background: "#66bb6a",
            display: "inline-block",
            animation: "painterTimerPulse 1s ease-in-out infinite",
        });

        const label = document.createElement("span");
        label.textContent = t("运行中", "Running");
        label.style.color = "#66bb6a";
        label.style.fontWeight = "bold";

        const timeText = document.createElement("span");
        timeText.id = "painter-workflow-timer-text";
        Object.assign(timeText.style, {
            fontWeight: "bold",
            fontSize: "16px",
            color: "#ffffff",
            fontVariantNumeric: "tabular-nums",
            minWidth: "70px",
            textAlign: "right",
        });

        timerContainer.appendChild(dot);
        timerContainer.appendChild(label);
        timerContainer.appendChild(timeText);
        document.body.appendChild(timerContainer);

        const style = document.createElement("style");
        style.textContent = `
            @keyframes painterTimerPulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.4; transform: scale(0.7); }
            }
        `;
        document.head.appendChild(style);

        let startTime = null;
        let intervalId = null;
        let running = false;

        function formatTime(ms) {
            const totalSeconds = Math.floor(ms / 1000);
            const minutes = Math.floor(totalSeconds / 60);
            const seconds = totalSeconds % 60;
            if (minutes > 0) {
                return isZh() ? `${minutes}分 ${seconds}秒` : `${minutes}m ${seconds}s`;
            }
            return isZh() ? `${seconds}秒` : `${seconds}s`;
        }

        function startTimer() {
            if (running) return;
            running = true;
            startTime = Date.now();

            label.textContent = t("运行中", "Running");
            label.style.color = "#66bb6a";
            dot.style.background = "#66bb6a";
            dot.style.animation = "painterTimerPulse 1s ease-in-out infinite";
            timeText.textContent = isZh() ? "0秒" : "0s";

            timerContainer.style.display = "flex";
            timerContainer.style.opacity = "1";

            if (intervalId) clearInterval(intervalId);
            intervalId = setInterval(() => {
                const elapsed = Date.now() - startTime;
                timeText.textContent = formatTime(elapsed);
            }, 200);
        }

        function stopTimer() {
            if (!running) return;
            running = false;

            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }

            const elapsed = Date.now() - startTime;
            timeText.textContent = formatTime(elapsed);

            label.textContent = t("总耗时", "Total");
            label.style.color = "#ffa726";
            dot.style.background = "#ffa726";
            dot.style.animation = "none";

            startTime = null;

            setTimeout(() => {
                if (!running) {
                    timerContainer.style.opacity = "0";
                    setTimeout(() => {
                        if (!running) {
                            timerContainer.style.display = "none";
                        }
                    }, 300);
                }
            }, 8000);
        }

        api.addEventListener("execution_start", () => {
            startTimer();
        });

        api.addEventListener("executing", ({ detail }) => {
            if (detail === null) {
                stopTimer();
            }
        });

        api.addEventListener("execution_success", () => {
            stopTimer();
        });

        api.addEventListener("execution_error", () => {
            stopTimer();
        });

        api.addEventListener("execution_interrupted", () => {
            stopTimer();
        });

        console.log("[PainterNodes] Workflow timer loaded.");
    },
});