import requests

def trigger_n8n_webhook(webhook_url, payload):
    """
    Triggers an n8n webhook with the given payload.
    """
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True, "Webhook triggered successfully."
    except requests.exceptions.RequestException as e:
        return False, f"Failed to trigger webhook: {e}"
