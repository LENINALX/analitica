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

Todo el modelo se entrena una sola vez al iniciar la app (con caché de Streamlit), usando el
mismo dataset y la misma metodología (80/20 estratificado, random_state=42) documentados en el
Informe Técnico.
