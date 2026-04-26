import os
import requests

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")


def get_flow_segment(lat, lng):
    if not TOMTOM_API_KEY:
        raise RuntimeError("TOMTOM_API_KEY not set in .env")

    url = (
        "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        f"?key={TOMTOM_API_KEY}&point={lat},{lng}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def get_incidents(bbox):
    if not TOMTOM_API_KEY:
        raise RuntimeError("TOMTOM_API_KEY not set in .env")

    url = (
        "https://api.tomtom.com/traffic/services/5/incidentDetails"
        f"?key={TOMTOM_API_KEY}"
        f"&bbox={bbox}"
        "&fields={incidents{type,geometry,properties{iconCategory,magnitudeOfDelay}}}"
        "&language=en-GB"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()