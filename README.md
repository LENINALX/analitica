# AgroSmart EC — Dashboard interactivo (Streamlit)

Prototipo de dashboard en Python para el Proyecto Integrador de Analítica de Negocios (ULEAM),
usando el Crop Recommendation Dataset (2,200 registros, 22 cultivos).

## Cómo ejecutarlo

1. Instala Python 3.9+ si no lo tienes.
2. Abre una terminal en esta carpeta y ejecuta:

```bash
pip install -r requirements.txt
streamlit run app.py
```

3. Se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## Qué incluye

- **Panorama General**: KPIs, comparación de los 3 modelos (Árbol de Decisión, Random Forest,
  Regresión Logística) y las categorías de cultivos.
- **Perfil por Cultivo**: selector múltiple + gráfico de radar comparando el perfil normalizado
  de cada cultivo contra el promedio global.
- **Correlaciones**: mapa de calor real de Pearson + gráfico de dispersión configurable
  (elige tú qué dos variables comparar).
- **Recomendador (What-If)**: sliders para simular las 7 variables de una parcela; el sistema
  usa el **Random Forest real entrenado en vivo** (no una aproximación) para predecir el
  cultivo más probable, con su nivel de confianza y el Top 5 de alternativas.

- **Asesor IA**: interpreta el diagnóstico del suelo y el Top 5 de alternativas calculado por
  Random Forest para responder preguntas en español. No sustituye el modelo: recibe sus
  resultados como contexto para que las recomendaciones sean trazables.

## Activar el asesor IA (opcional)

1. Instala las dependencias de nuevo: `pip install -r requirements.txt`.
2. Crea una clave de API de OpenAI y configúrala fuera del código fuente. En PowerShell para la
   sesión actual:

```powershell
$env:OPENAI_API_KEY="tu_clave"
streamlit run app.py
```

También puedes crear `.streamlit/secrets.toml` (no lo subas a Git):

```toml
OPENAI_API_KEY = "tu_clave"
# Opcional: OPENAI_MODEL = "gpt-5"
```

Sin clave, las pestañas de analítica, diagnóstico, alternativas y clima continúan disponibles;
solamente se desactiva el botón del asesor.

Todo el modelo se entrena una sola vez al iniciar la app (con caché de Streamlit), usando el
mismo dataset y la misma metodología (80/20 estratificado, random_state=42) documentados en el
Informe Técnico.
