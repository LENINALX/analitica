"""Asesor conversacional para AgroSmart EC.

El modelo de lenguaje no calcula la compatibilidad: interpreta el resultado del
Random Forest y el contexto de suelo que recibe la aplicación. Así se conserva
la trazabilidad de la recomendación cuantitativa.
"""

import os
from typing import Any


SYSTEM_INSTRUCTIONS = """
Eres el asesor agrícola de AgroSmart EC para un prototipo académico en Ecuador.
Responde siempre en español, de forma clara y práctica. Usa únicamente los datos
entregados como evidencia numérica. Explica por qué las alternativas sugeridas por
el modelo pueden ajustarse al suelo y menciona limitaciones (riego, drenaje, pH o
nutrientes) cuando correspondan. No inventes resultados de laboratorio, precios,
plagas ni dosis exactas de fertilizante. No presentes una predicción como garantía.
Incluye un aviso breve: la decisión final requiere validación de un ingeniero
agrónomo y, para una parcela real, análisis de laboratorio.
""".strip()


def get_api_key(streamlit_secrets: Any | None = None) -> str | None:
    """Lee la clave sin guardarla en el código fuente."""
    # Prioriza la variable de entorno. No se debe evaluar `st.secrets` como
    # booleano: Streamlit lanza StreamlitSecretNotFoundError si no existe aún
    # un archivo secrets.toml.
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    if streamlit_secrets is not None:
        try:
            key = streamlit_secrets.get("OPENAI_API_KEY")
            if key:
                return str(key)
        except Exception:
            pass
    return os.getenv("OPENAI_API_KEY")


def ask_agro_agent(
    question: str,
    context: dict[str, Any],
    api_key: str,
    model: str | None = None,
) -> str:
    """Solicita una explicación contextual al modelo mediante Responses API."""
    from openai import OpenAI

    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-5")
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=selected_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=(
            "Contexto técnico calculado por AgroSmart EC:\n"
            f"{context}\n\n"
            f"Pregunta del agricultor: {question}"
        ),
    )
    return response.output_text
