"""Location and Facility Lookup Tools.

Provides geocoding, nearby healthcare facility discovery (hospitals and pharmacies),
and directions generation using Google Maps Platform (when GOOGLE_MAPS_API_KEY is present)
with zero-key fallback to OpenStreetMap (Nominatim & Overpass API).
"""

import logging
import math
import os
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("location-tools")

# Custom User-Agent header per OpenStreetMap Nominatim/Overpass usage policy
OSM_USER_AGENT = "ClinicTriageCopilot/1.0 (healthcare-triage-copilot@local.ai)"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two geographic coordinates in kilometers."""
    r = 6371.0  # Earth's mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(r * c, 2)


def get_current_location() -> Optional[Dict[str, Any]]:
    """
    Auto-detect current geographic location (latitude, longitude, address, city).
    Uses high-accuracy IP geolocation with multi-provider fallback:
    1. ipapi.co
    2. ip-api.com
    3. freeipapi.com
    Returns a dict with {'lat', 'lng', 'address', 'city', 'region', 'country'} or None.
    """
    # 1. ipapi.co
    try:
        resp = requests.get(
            "https://ipapi.co/json/",
            headers={"User-Agent": OSM_USER_AGENT},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            lat = data.get("latitude")
            lng = data.get("longitude")
            if lat is not None and lng is not None:
                city = data.get("city") or "Local Area"
                region = data.get("region") or ""
                country = data.get("country_name") or ""
                parts = [p for p in [city, region, country] if p]
                address = ", ".join(parts)
                logger.info("Auto-detected location via ipapi.co: %s (%f, %f)", address, lat, lng)
                return {
                    "lat": float(lat),
                    "lng": float(lng),
                    "address": address,
                    "city": city,
                    "region": region,
                    "country": country,
                }
    except Exception as exc:
        logger.debug("ipapi.co auto-location failed: %s", exc)

    # 2. ip-api.com
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                lat = data.get("lat")
                lng = data.get("lon")
                if lat is not None and lng is not None:
                    city = data.get("city") or "Local Area"
                    region = data.get("regionName") or ""
                    country = data.get("country") or ""
                    parts = [p for p in [city, region, country] if p]
                    address = ", ".join(parts)
                    logger.info("Auto-detected location via ip-api.com: %s (%f, %f)", address, lat, lng)
                    return {
                        "lat": float(lat),
                        "lng": float(lng),
                        "address": address,
                        "city": city,
                        "region": region,
                        "country": country,
                    }
    except Exception as exc:
        logger.debug("ip-api.com auto-location failed: %s", exc)

    # 3. freeipapi.com
    try:
        resp = requests.get("https://freeipapi.com/api/json", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            lat = data.get("latitude")
            lng = data.get("longitude")
            if lat is not None and lng is not None:
                city = data.get("cityName") or "Local Area"
                region = data.get("regionName") or ""
                country = data.get("countryName") or ""
                parts = [p for p in [city, region, country] if p]
                address = ", ".join(parts)
                logger.info("Auto-detected location via freeipapi.com: %s (%f, %f)", address, lat, lng)
                return {
                    "lat": float(lat),
                    "lng": float(lng),
                    "address": address,
                    "city": city,
                    "region": region,
                    "country": country,
                }
    except Exception as exc:
        logger.debug("freeipapi.com auto-location failed: %s", exc)

    return None


def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """
    Geocode an address string into (latitude, longitude) coordinates.
    Tries Google Geocoding API if GOOGLE_MAPS_API_KEY is set;
    otherwise falls back to OpenStreetMap Nominatim.
    If address indicates current/auto location, retrieves exact live coordinates.
    """
    if not address or not address.strip():
        # Auto-detect current location if no address given
        loc = get_current_location()
        return (loc["lat"], loc["lng"]) if loc else None

    address_clean = address.strip()

    # Check for auto-location keywords
    if address_clean.lower() in {"current", "current location", "my location", "auto", "here", "live location"}:
        loc = get_current_location()
        if loc:
            return (loc["lat"], loc["lng"])

    google_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

    # 1. Try Google Geocoding API
    if google_key:
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={urllib.parse.quote_plus(address_clean)}&key={google_key}"
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK" and data.get("results"):
                    loc = data["results"][0]["geometry"]["location"]
                    lat = float(loc["lat"])
                    lng = float(loc["lng"])
                    logger.info("Geocoded '%s' via Google Geocoding to (%f, %f)", address_clean, lat, lng)
                    return (lat, lng)
        except Exception as exc:
            logger.warning("Google Geocoding call failed, falling back to Nominatim: %s", exc)

    # 2. OpenStreetMap Nominatim fallback
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote_plus(address_clean)}&format=json&limit=1"
        headers = {"User-Agent": OSM_USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                lat = float(data[0]["lat"])
                lng = float(data[0]["lon"])
                logger.info("Geocoded '%s' via OSM Nominatim to (%f, %f)", address_clean, lat, lng)
                return (lat, lng)
    except Exception as exc:
        logger.warning("OSM Nominatim geocoding failed: %s", exc)

    return None


NON_EMERGENCY_EXCLUSIONS = [
    # Dental clinics / hospitals
    "dental",
    "dentist",
    "tooth",
    "teeth",
    "orthodontic",
    "oral care",
    "dento",
    # Orthopedic / Bone standalone specialty clinics
    "bone and joint",
    "bone & joint",
    "bone setting",
    "bone setter",
    "bonesetter",
    "bone hospital",
    "orthopedic foundation",
    "ortho clinic",
    "spine clinic",
    "joint replacement",
    # Eye / Vision
    "eye hospital",
    "eye care",
    "eye centre",
    "eye center",
    "ophthalmology",
    "lasik",
    "vision care",
    "retina",
    "optometry",
    "optical",
    # Cosmetic / Skin / Dermatology / Hair / Plastic surgery
    "skin clinic",
    "skin hospital",
    "derma",
    "dermatology",
    "cosmetic",
    "plastic surgery",
    "hair transplant",
    "aesthetic",
    "laser clinic",
    # Fertility / IVF / Specialty centers
    "fertility",
    "ivf",
    "test tube",
    "reproduction",
    # Alternative medicine / Wellness / Spa / Homeopathy / Ayurveda
    "ayurvedic",
    "ayurveda",
    "homeopathy",
    "homeo",
    "naturopathy",
    "unani",
    "siddha",
    "massage",
    "spa",
    "wellness",
    # Veterinary
    "veterinary",
    "animal",
    "pet",
    # Labs / Diagnostic / Scan centers
    "diagnostic",
    "scan center",
    "imaging",
    "blood bank",
    "laboratory",
    "pathology",
    "x-ray",
    "mri",
    "collection centre",
    "sample collection",
]


def is_relevant_healthcare_facility(name: str, place_type: str = "hospital") -> bool:
    """
    Clinically validates whether a facility is appropriate for acute/general medical triage.
    Filters out non-emergency facilities (e.g. dental, eye care, cosmetic, bone/joint standalone clinics).
    """
    if not name or not name.strip():
        return False

    name_lower = name.lower()

    if place_type == "hospital":
        if any(exclusion in name_lower for exclusion in NON_EMERGENCY_EXCLUSIONS):
            return False
        return True

    elif place_type == "pharmacy":
        pharmacy_exclusions = [
            "dental",
            "dentist",
            "optical",
            "opticals",
            "optician",
            "optometry",
            "eyewear",
            "eye wear",
            "eye care",
            "eyecare",
            "cosmetics",
            "cosmetic",
            "pet",
            "animal",
            "veterinary",
        ]
        if any(exclusion in name_lower for exclusion in pharmacy_exclusions):
            return False
        return True

    return True


def find_nearby_places(
    lat: float,
    lng: float,
    place_type: str = "hospital",
    radius_m: int = 8000,
) -> List[Dict[str, Any]]:
    """
    Find nearby places of place_type ('hospital' or 'pharmacy') within radius_m.
    Filters for acute, relevant medical centers (excluding dental, cosmetic, etc.).
    Tries Google Places API if GOOGLE_MAPS_API_KEY is configured;
    otherwise queries OpenStreetMap Overpass & Nominatim APIs.
    Returns up to 5 facilities sorted by distance in km.
    """
    if lat is None or lng is None:
        return []

    place_type_normalized = place_type.lower().strip()
    if place_type_normalized not in {"hospital", "pharmacy"}:
        place_type_normalized = "hospital"

    google_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    results: List[Dict[str, Any]] = []

    # 1. Try Google Places Nearby Search
    if google_key:
        try:
            url = (
                f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                f"?location={lat},{lng}&radius={radius_m}&type={place_type_normalized}&key={google_key}"
            )
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") in {"OK", "ZERO_RESULTS"}:
                    for item in data.get("results", []):
                        name = item.get("name", f"Local {place_type_normalized.capitalize()}")
                        if not is_relevant_healthcare_facility(name, place_type_normalized):
                            continue
                        loc = item.get("geometry", {}).get("location", {})
                        plat = float(loc.get("lat", lat))
                        plng = float(loc.get("lng", lng))
                        addr = item.get("vicinity") or item.get("formatted_address") or ""
                        dist = haversine_distance(lat, lng, plat, plng)
                        results.append(
                            {
                                "name": name,
                                "address": addr,
                                "lat": plat,
                                "lng": plng,
                                "distance_km": dist,
                                "place_type": place_type_normalized,
                            }
                        )
                    if results:
                        results.sort(key=lambda x: x["distance_km"])
                        return results[:5]
        except Exception as exc:
            logger.warning("Google Places search failed, falling back to Overpass: %s", exc)

    # 2. OpenStreetMap Overpass API fallback
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    query = f"""[out:json][timeout:10];
(
  node["amenity"="{place_type_normalized}"](around:{radius_m},{lat},{lng});
  way["amenity"="{place_type_normalized}"](around:{radius_m},{lat},{lng});
);
out center 35;"""

    for endpoint in overpass_endpoints:
        try:
            headers = {"User-Agent": OSM_USER_AGENT}
            resp = requests.post(endpoint, data={"data": query}, headers=headers, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                seen_names = set()
                for el in elements:
                    tags = el.get("tags", {})
                    name = (
                        tags.get("name")
                        or tags.get("operator")
                        or tags.get("brand")
                        or f"Local {place_type_normalized.capitalize()}"
                    )
                    norm_name = name.lower().strip()
                    if norm_name in seen_names or not is_relevant_healthcare_facility(name, place_type_normalized):
                        continue
                    seen_names.add(norm_name)

                    # Extract coordinates (node lat/lon or way center)
                    plat = el.get("lat") or el.get("center", {}).get("lat")
                    plng = el.get("lon") or el.get("center", {}).get("lon")
                    if plat is None or plng is None:
                        continue

                    plat = float(plat)
                    plng = float(plng)

                    # Extract address
                    addr_parts = [
                        tags.get("addr:housenumber", ""),
                        tags.get("addr:street", ""),
                        tags.get("addr:suburb", "") or tags.get("addr:district", ""),
                        tags.get("addr:city", "") or tags.get("addr:town", ""),
                    ]
                    clean_addr = ", ".join(p for p in addr_parts if p.strip())

                    dist = haversine_distance(lat, lng, plat, plng)
                    results.append(
                        {
                            "name": name,
                            "address": clean_addr,
                            "lat": plat,
                            "lng": plng,
                            "distance_km": dist,
                            "place_type": place_type_normalized,
                        }
                    )

                if results:
                    results.sort(key=lambda x: x["distance_km"])
                    return results[:5]
        except Exception as exc:
            logger.debug("Overpass endpoint %s failed: %s", endpoint, exc)

    # 3. OpenStreetMap Nominatim Bounded POI Search (Fast secondary fallback)
    try:
        delta_lat = radius_m / 111000.0
        cos_lat = math.cos(math.radians(lat))
        delta_lng = radius_m / (111000.0 * (cos_lat if abs(cos_lat) > 0.001 else 1.0))
        viewbox = f"{lng - delta_lng},{lat + delta_lat},{lng + delta_lng},{lat - delta_lat}"

        nom_url = (
            f"https://nominatim.openstreetmap.org/search?"
            f"q={place_type_normalized}&format=json&viewbox={viewbox}&bounded=1&limit=25"
        )
        headers = {"User-Agent": OSM_USER_AGENT}
        resp = requests.get(nom_url, headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            seen_names = set()
            for item in data:
                plat = float(item.get("lat", lat))
                plng = float(item.get("lon", lng))
                full_name = item.get("display_name", "")
                parts = [p.strip() for p in full_name.split(",") if p.strip()]
                name = parts[0] if parts else f"Local {place_type_normalized.capitalize()}"
                norm_name = name.lower().strip()
                if norm_name in seen_names or not is_relevant_healthcare_facility(name, place_type_normalized):
                    continue
                seen_names.add(norm_name)

                addr = ", ".join(parts[1:4]) if len(parts) > 1 else ""
                dist = haversine_distance(lat, lng, plat, plng)
                results.append(
                    {
                        "name": name,
                        "address": addr,
                        "lat": plat,
                        "lng": plng,
                        "distance_km": dist,
                        "place_type": place_type_normalized,
                    }
                )
            if results:
                results.sort(key=lambda x: x["distance_km"])
                return results[:5]
    except Exception as exc:
        logger.warning("Nominatim POI search fallback failed: %s", exc)

    return results


def build_directions_url(
    origin_lat: Optional[float],
    origin_lng: Optional[float],
    dest_lat: float,
    dest_lng: float,
    dest_name: Optional[str] = None,
) -> str:
    """
    Build a direct Google Maps turn-by-turn driving navigation URL.
    By specifying destination coordinates, Google Maps automatically
    routes from the user's actual live device GPS location ('Your location'),
    avoiding static IP tower / warehouse misidentification.
    """
    dest_str = f"{dest_lat:.6f},{dest_lng:.6f}"
    return f"https://www.google.com/maps/dir/?api=1&destination={dest_str}&travelmode=driving"
