"""
Datos de contexto Ecuador: tipos de suelo representativos y metadatos de cultivos.

IMPORTANTE (transparencia metodologica): los perfiles de N,P,K,pH,humedad por tipo de
suelo son ESTIMACIONES REPRESENTATIVAS con fines pedagogicos, construidas a partir de
literatura edafologica general sobre los ordenes de suelo presentes en Ecuador
(Espinosa, Moreno y Bernal, 2022, "Suelos del Ecuador", IGM; caracterizacion de suelos
de la Costa y Sierra). NO son mediciones de laboratorio de una parcela real. Para una
decision agronomica real siempre se recomienda un analisis de suelo de laboratorio.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

REGIONES = {
    "Costa": {"temperature": 26.0, "humidity": 80.0, "rainfall": 140.0},
    "Sierra": {"temperature": 15.5, "humidity": 68.0, "rainfall": 90.0},
    "Oriente (Amazonia)": {"temperature": 25.0, "humidity": 88.0, "rainfall": 260.0},
}

SUELOS = {
    "andisol": {
        "nombre": "Andisol (ceniza volcánica)",
        "region_tipica": "Sierra norte y llanura costera cerca de Quinindé",
        "descripcion": "Suelo oscuro derivado de cenizas volcánicas, alta materia orgánica, buena estructura, pero el fósforo queda \"fijado\" y poco disponible para la planta.",
        "perfil": {"N": 92, "P": 32, "K": 48, "ph": 5.8},
        "imagen": str(ASSETS_DIR / "andisol.png"),
    },
    "vertisol": {
        "nombre": "Vertisol (arcilloso)",
        "region_tipica": "Costa sur — Manabí, Guayas",
        "descripcion": "Suelo muy arcilloso, se agrieta en época seca y es pegajoso en época húmeda. Retiene bien el potasio pero es difícil de trabajar mecánicamente.",
        "perfil": {"N": 48, "P": 52, "K": 92, "ph": 7.2},
        "imagen": str(ASSETS_DIR / "vertisol.png"),
    },
    "aluvial": {
        "nombre": "Entisol / Aluvial (vega de río)",
        "region_tipica": "Llanuras de la Costa y la Amazonía",
        "descripcion": "Suelo joven depositado por ríos, naturalmente fértil, aunque puede tener problemas de drenaje en época de crecidas.",
        "perfil": {"N": 70, "P": 55, "K": 55, "ph": 6.7},
        "imagen": str(ASSETS_DIR / "aluvial.png"),
    },
    "inceptisol": {
        "nombre": "Inceptisol (suelo joven moderado)",
        "region_tipica": "Zonas de transición en Costa y Sierra",
        "descripcion": "Suelo con desarrollo incipiente, fertilidad media, sin limitaciones extremas — es el \"término medio\" agronómico.",
        "perfil": {"N": 55, "P": 50, "K": 40, "ph": 6.3},
        "imagen": str(ASSETS_DIR / "inceptisol.png"),
    },
    "ferralitico": {
        "nombre": "Suelo ferralítico / ácido (Ultisol)",
        "region_tipica": "Amazonía y zona húmeda norte de la Costa",
        "descripcion": "Suelo muy lixiviado por la lluvia constante: ácido, bajo en nutrientes disponibles y con riesgo de toxicidad por aluminio.",
        "perfil": {"N": 25, "P": 15, "K": 20, "ph": 4.8},
        "imagen": str(ASSETS_DIR / "ultisol.png"),
    },
    "franco_arenoso": {
        "nombre": "Franco-arenoso costero seco",
        "region_tipica": "Zonas secas de Manabí y Santa Elena",
        "descripcion": "Suelo suelto y arenoso, baja retención de agua y nutrientes; requiere riego frecuente y aportes regulares de materia orgánica.",
        "perfil": {"N": 35, "P": 20, "K": 30, "ph": 6.8},
        "imagen": str(ASSETS_DIR / "franco_arenoso.png"),
    },
}

# Traduccion y etiqueta de "tipicidad" en Ecuador para los 22 cultivos del dataset.
CROP_ES = {
    "rice": ("Arroz", True), "maize": ("Maíz", True), "banana": ("Banano", True),
    "coffee": ("Café", True), "cotton": ("Algodón", True), "coconut": ("Coco", True),
    "papaya": ("Papaya", True), "mango": ("Mango", True), "watermelon": ("Sandía", True),
    "muskmelon": ("Melón", True), "orange": ("Naranja", True), "apple": ("Manzana", True),
    "grapes": ("Uva", True), "pomegranate": ("Granada", True),
    "kidneybeans": ("Fréjol rojo", True), "pigeonpeas": ("Guandú", False),
    "chickpea": ("Garbanzo", False), "lentil": ("Lenteja", False),
    "mothbeans": ("Frijol mungo negro", False), "mungbean": ("Frijol mungo", False),
    "blackgram": ("Urad (frijol negro)", False), "jute": ("Yute", False),
}