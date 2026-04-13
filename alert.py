from datetime import datetime
import os


def trigger_alert(prediction, log_file="alerts.log"):
    """
    Trigger alert for a single prediction (live IDS style).
    prediction: 1 -> attack, 0 -> normal
    """

    if prediction == 1:
        message = (
            f"[{datetime.now()}] 🚨 ALERT: Intrusion detected (malicious connection)\n"
        )

        print("🚨🚨🚨 ALERT: INTRUSION DETECTED 🚨🚨🚨")

        # Ensure log file exists
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message)
