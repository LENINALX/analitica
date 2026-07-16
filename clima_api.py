"""
Integracion con la API publica Open-Meteo (https://open-meteo.com) para el Proyecto
Integrador de Analitica de Negocios (ULEAM). No requiere API key.

Se usan dos endpoints:
- Geocoding API: convierte un nombre de ciudad de Ecuador en coordenadas (lat, lon).
- Historical Weather API (archive-api): trae temperatura, precipitacion y humedad
  diarias reales de los ultimos anios, para calcular una climatologia mensual
  (que tan calido/lluvioso/humedo es en promedio cada mes en esa ciudad).

Todas las funciones son pequenas y usan cache de Streamlit (TTL de 1 dia) para no
golpear la API en cada interaccion del usuario.
"""

import requests
import pandas as pd
from datetime import date, timedelta

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}


def geocode_city(city_name: str, timeout: int = 10):
    """Busca una ciudad y devuelve una lista de coincidencias (prioriza Ecuador)."""
    params = {"name": city_name, "count": 8, "language": "es", "format": "json"}
    r = requests.get(GEOCODING_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", []) or []
    ec_results = [x for x in results if x.get("country_code") == "EC"]
    return ec_results if ec_results else results


def get_monthly_climatology(lat: float, lon: float, years: int = 3, timeout: int = 20):
    """
    Trae datos historicos diarios reales (temperatura, precipitacion, humedad) de los
    ultimos `years` anios para una coordenada, y devuelve un DataFrame con el promedio
    por mes (climatologia mensual: 12 filas, una por mes calendario).
    """
    end_date = date.today() - timedelta(days=6)  # la API exige un pequeno margen
    start_date = date(end_date.year - years, end_date.month, 1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        "timezone": "auto",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    if "daily" not in payload:
        raise ValueError(f"Respuesta inesperada de la API: {payload}")

    daily = pd.DataFrame(payload["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily["year"] = daily["time"].dt.year
    daily["month"] = daily["time"].dt.month

    # precipitacion: sumar por year-month (total mensual), luego promediar entre los anios
    monthly_precip_totals = (
        daily.groupby(["year", "month"])["precipitation_sum"].sum().reset_index()
    )
    avg_precip = monthly_precip_totals.groupby("month")["precipitation_sum"].mean()

    avg_temp = daily.groupby("month")["temperature_2m_mean"].mean()
    avg_hum = daily.groupby("month")["relative_humidity_2m_mean"].mean()

    monthly = pd.DataFrame({
        "mes": range(1, 13),
        "temperatura": [avg_temp.get(m, float("nan")) for m in range(1, 13)],
        "precipitacion_mm": [avg_precip.get(m, float("nan")) for m in range(1, 13)],
        "humedad": [avg_hum.get(m, float("nan")) for m in range(1, 13)],
    })
    monthly["mes_nombre"] = monthly["mes"].map(MESES_ES)
    monthly["temporada"] = monthly["precipitacion_mm"].apply(
        lambda v: "Lluviosa" if v >= monthly["precipitacion_mm"].median() else "Seca"
    )
    return monthly