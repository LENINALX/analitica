"""
AgroSmart EC — Dashboard interactivo de recomendación de cultivos
Proyecto Integrador · Analítica de Negocios · ULEAM

Para ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from ai_agent import ask_agro_agent, get_api_key

# ------------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AgroSmart EC — Recomendador de Cultivos",
    page_icon="🌱",
    layout="wide",
)

NAVY = "#1F2A44"
GREEN = "#1D9E75"
GREY = "#5B6472"

VAR_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
VAR_LABELS = {
    "N": "Nitrógeno (N)", "P": "Fósforo (P)", "K": "Potasio (K)",
    "temperature": "Temperatura (°C)", "humidity": "Humedad (%)",
    "ph": "pH", "rainfall": "Lluvia (mm)",
}
CROP_CATEGORY = {
    'rice': 'Cereal', 'maize': 'Cereal', 'chickpea': 'Leguminosa', 'kidneybeans': 'Leguminosa',
    'pigeonpeas': 'Leguminosa', 'mothbeans': 'Leguminosa', 'mungbean': 'Leguminosa', 'blackgram': 'Leguminosa',
    'lentil': 'Leguminosa', 'pomegranate': 'Fruta', 'banana': 'Fruta', 'mango': 'Fruta', 'grapes': 'Fruta',
    'watermelon': 'Fruta', 'muskmelon': 'Fruta', 'apple': 'Fruta', 'orange': 'Fruta', 'papaya': 'Fruta',
    'coconut': 'Fruta', 'cotton': 'Fibra', 'jute': 'Fibra', 'coffee': 'Estimulante'
}

# ------------------------------------------------------------------
# Carga y modelo (cacheados: se calculan una sola vez)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = Path(__file__).resolve().parent / "Crop_recommendation.csv"
    df = pd.read_csv(csv_path)
    df["categoria"] = df["label"].map(CROP_CATEGORY)
    return df


@st.cache_resource
def train_models(df):
    X = df[VAR_COLS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(random_state=42, max_depth=8).fit(X_train, y_train)
    rf = RandomForestClassifier(random_state=42, n_estimators=200).fit(X_train, y_train)

    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(X_train), y_train)

    results = {
        "Árbol de Decisión": {
            "accuracy": accuracy_score(y_test, dt.predict(X_test)),
            "f1": f1_score(y_test, dt.predict(X_test), average="macro"),
        },
        "Random Forest": {
            "accuracy": accuracy_score(y_test, rf.predict(X_test)),
            "f1": f1_score(y_test, rf.predict(X_test), average="macro"),
        },
        "Regresión Logística": {
            "accuracy": accuracy_score(y_test, lr.predict(scaler.transform(X_test))),
            "f1": f1_score(y_test, lr.predict(scaler.transform(X_test)), average="macro"),
        },
    }

    importances = pd.Series(rf.feature_importances_, index=VAR_COLS).sort_values(ascending=False)
    return rf, results, importances


df = load_data()
rf_model, model_results, feature_importance = train_models(df)

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background:{NAVY};padding:1.2rem 1.5rem;border-radius:10px;margin-bottom:1rem">
        <h1 style="color:white;margin:0;font-size:1.6rem">🌱 AgroSmart EC — Recomendador de Cultivos</h1>
        <p style="color:#C7CEDA;margin:4px 0 0;font-size:0.95rem">
            Analítica de Negocios · ULEAM · Dataset real: {len(df):,} parcelas, {df['label'].nunique()} cultivos
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Panorama General", "🌾 Perfil por Cultivo", "🔗 Correlaciones",
     "🧭 Diagnóstico de Suelo (Ecuador)", "🌦️ Clima y Temporada (API)",
     "🤖 Asesor IA"]
)

# ------------------------------------------------------------------
# TAB 1 — Panorama General
# ------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Cultivos", df["label"].nunique())
    c3.metric("Accuracy Random Forest", f"{model_results['Random Forest']['accuracy']*100:.1f}%")
    c4.metric("Variable más importante", VAR_LABELS[feature_importance.index[0]])

    col1, col2 = st.columns(2)

    with col1:
        res_df = pd.DataFrame(model_results).T.reset_index().rename(columns={"index": "modelo"})
        fig = px.bar(
            res_df, x="modelo", y="accuracy", color="modelo",
            color_discrete_map={"Árbol de Decisión": "#85B7EB", "Random Forest": GREEN, "Regresión Logística": "#7F77DD"},
            text_auto=".2%", title="Comparación de exactitud entre modelos",
        )
        fig.update_yaxes(range=[0.8, 1.0], tickformat=".0%", title="Accuracy")
        fig.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig, width='stretch')

    with col2:
        cat_counts = df.drop_duplicates("label")["categoria"].value_counts().reset_index()
        cat_counts.columns = ["categoria", "cultivos"]
        fig2 = px.pie(cat_counts, names="categoria", values="cultivos", hole=0.5,
                      title="Cultivos del catálogo por categoría")
        st.plotly_chart(fig2, width='stretch')

    imp_df = feature_importance.reset_index()
    imp_df.columns = ["variable", "importancia"]
    imp_df["variable"] = imp_df["variable"].map(VAR_LABELS)
    fig3 = px.bar(
        imp_df.sort_values("importancia"), x="importancia", y="variable", orientation="h",
        title="Importancia de variables (Random Forest — XAI)", color_discrete_sequence=[GREEN],
        text_auto=".1%",
    )
    fig3.update_xaxes(tickformat=".0%", title="Importancia relativa")
    fig3.update_layout(yaxis_title="")
    st.plotly_chart(fig3, width='stretch')

# ------------------------------------------------------------------
# TAB 2 — Perfil por Cultivo
# ------------------------------------------------------------------
with tab2:
    crops_sorted = sorted(df["label"].unique())
    selected_crops = st.multiselect(
        "Selecciona uno o más cultivos para comparar",
        crops_sorted, default=["rice", "chickpea"],
    )

    if selected_crops:
        global_mean = df[VAR_COLS].mean()
        global_std = df[VAR_COLS].std()

        fig = go.Figure()
        for crop in selected_crops:
            crop_mean = df[df["label"] == crop][VAR_COLS].mean()
            z = (crop_mean - global_mean) / global_std
            fig.add_trace(go.Scatterpolar(
                r=z.values, theta=[VAR_LABELS[c] for c in VAR_COLS],
                fill="toself", name=crop,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
            title="Perfil normalizado (z-score) vs. promedio global",
            height=500,
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("**Valores reales (media ± desviación estándar) por cultivo seleccionado**")
        summary = df[df["label"].isin(selected_crops)].groupby("label")[VAR_COLS].agg(["mean", "std"]).round(2)
        st.dataframe(summary, width='stretch')
    else:
        st.info("Selecciona al menos un cultivo para ver su perfil.")

# ------------------------------------------------------------------
# TAB 3 — Correlaciones
# ------------------------------------------------------------------
with tab3:
    corr = df[VAR_COLS].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        labels=dict(color="Correlación de Pearson"),
        x=[VAR_LABELS[c] for c in VAR_COLS], y=[VAR_LABELS[c] for c in VAR_COLS],
        title="Matriz de correlación de Pearson (n = 2,200)",
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "La correlación más fuerte es P–K (0.74). Humedad y lluvia, pese a estar climatológicamente "
        "relacionadas en la realidad, muestran una correlación casi nula (0.09) en esta muestra — "
        "recuerda que correlación no implica causalidad."
    )

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        var_x = st.selectbox("Variable eje X", VAR_COLS, index=1, format_func=lambda v: VAR_LABELS[v])
    with col2:
        var_y = st.selectbox("Variable eje Y", VAR_COLS, index=2, format_func=lambda v: VAR_LABELS[v])

    fig2 = px.scatter(
        df, x=var_x, y=var_y, color="categoria", hover_data=["label"],
        title=f"{VAR_LABELS[var_x]} vs. {VAR_LABELS[var_y]} por categoría de cultivo",
    )
    st.plotly_chart(fig2, width='stretch')

import ecuador_data as ec

# ------------------------------------------------------------------
# TAB 4 — Diagnóstico y Recomendación (contexto Ecuador)
# ------------------------------------------------------------------
with tab4:
    st.markdown(
        """
        **Flujo pensado para el agricultor:** 1) elige tu región, 2) elige tu tipo de suelo
        (por imagen), 3) elige el cultivo que quieres sembrar → recibe un diagnóstico y
        recomendaciones concretas de qué hacer con tu suelo.
        """
    )
    st.caption(
        "⚠️ Los perfiles de suelo son estimaciones representativas basadas en literatura "
        "edafológica de Ecuador (Espinosa, Moreno y Bernal, 2022 — IGM), no mediciones de "
        "laboratorio de una parcela real. Para una decisión agronómica real siempre se "
        "recomienda un análisis de suelo en laboratorio."
    )

    st.markdown("### Paso 1 — ¿En qué región estás?")
    region = st.radio("Región", list(ec.REGIONES.keys()), horizontal=True, label_visibility="collapsed")
    clima = ec.REGIONES[region]

    st.markdown("### Paso 2 — Elige tu tipo de suelo")
    soil_keys = list(ec.SUELOS.keys())
    cols = st.columns(3)
    if "soil_choice" not in st.session_state:
        st.session_state.soil_choice = "aluvial"

    for i, key in enumerate(soil_keys):
        soil = ec.SUELOS[key]
        with cols[i % 3]:
            if Path(soil["imagen"]).exists():
                import base64

                with open(soil["imagen"], "rb") as f:
                    img = base64.b64encode(f.read()).decode()

                st.markdown(f"""
                    <div style="display:flex;justify-content:center;">
                    <img src="data:image/png;base64,{img}"
                         style="
                             width:450px;
                             height:350px;
                             object-fit:cover;
                             border-radius:10px;
                             border:1px solid #ddd;">
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Imagen no encontrada: {soil['imagen']}")
            selected = st.session_state.soil_choice == key
            label = f"✅ {soil['nombre']}" if selected else soil["nombre"]
            if st.button(label, key=f"btn_{key}", width='stretch'):
                st.session_state.soil_choice = key
            st.caption(soil["region_tipica"])

    soil_choice = st.session_state.soil_choice
    soil = ec.SUELOS[soil_choice]
    st.info(f"**{soil['nombre']}** — {soil['descripcion']}")

    st.markdown("### Paso 3 — ¿Qué cultivo quieres sembrar?")
    crop_options = sorted(df["label"].unique())
    def fmt_crop(c):
        nombre, tipico = ec.CROP_ES.get(c, (c, False))
        return f"{nombre}" + ("" if tipico else " (referencial, poco común en Ecuador)")
    desired_crop = st.selectbox("Cultivo deseado", crop_options, format_func=fmt_crop, index=crop_options.index("rice"))

    st.markdown("---")

    # Construir el vector de entrada: perfil del suelo elegido + clima de la region
    soil_vector = {
        "N": soil["perfil"]["N"], "P": soil["perfil"]["P"], "K": soil["perfil"]["K"],
        "ph": soil["perfil"]["ph"],
        "temperature": clima["temperature"], "humidity": clima["humidity"], "rainfall": clima["rainfall"],
    }
    X_soil = pd.DataFrame([soil_vector])[VAR_COLS]
    probs = rf_model.predict_proba(X_soil)[0]
    classes = rf_model.classes_
    prob_desired = probs[list(classes).index(desired_crop)] if desired_crop in classes else 0.0

    col_diag, col_alt = st.columns([1.3, 1])

    with col_diag:
        st.subheader(f"Diagnóstico para sembrar {fmt_crop(desired_crop)}")
        nivel = "alta" if prob_desired > 0.5 else ("media" if prob_desired > 0.15 else "baja")
        color = GREEN if nivel == "alta" else ("#C9971F" if nivel == "media" else "#C0392B")
        st.markdown(
            f"""
            <div style="background:{color}22;border-left:5px solid {color};border-radius:6px;padding:0.9rem 1.1rem">
                <p style="margin:0;color:{color};font-weight:600">Probabilidad de éxito estimada: {prob_desired*100:.1f}% (compatibilidad {nivel})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ¿Qué hacer con tu suelo? (recomendaciones)")
        crop_profile = df[df["label"] == desired_crop][VAR_COLS].mean()
        recs = []

        gap_n = crop_profile["N"] - soil_vector["N"]
        gap_p = crop_profile["P"] - soil_vector["P"]
        gap_k = crop_profile["K"] - soil_vector["K"]
        gap_ph = crop_profile["ph"] - soil_vector["ph"]
        gap_rain = crop_profile["rainfall"] - soil_vector["rainfall"]

        if gap_n > 15:
            recs.append("🟤 **Nitrógeno bajo** respecto al óptimo del cultivo: incorporar materia orgánica (compost, estiércol) o fertilizante nitrogenado (ej. urea) de forma fraccionada.")
        elif gap_n < -15:
            recs.append("🟤 **Nitrógeno por encima** de lo requerido: reducir/evitar fertilización nitrogenada adicional para no favorecer plagas y enfermedades foliares.")

        if gap_p > 15:
            recs.append("🟠 **Fósforo bajo**: aplicar fertilizante fosfatado (ej. roca fosfórica o DAP) incorporado cerca de la raíz, ya que el fósforo se mueve poco en el suelo.")
        if gap_k > 15:
            recs.append("🟡 **Potasio bajo**: aplicar fertilizante potásico (ej. cloruro o sulfato de potasio), importante para llenado de fruto/grano.")
        elif gap_k < -15:
            recs.append("🟡 **Potasio muy por encima** de lo requerido: no es necesario aplicar más potasio en los próximos ciclos.")

        if gap_ph > 0.5:
            recs.append("⚪ **pH del suelo más ácido** de lo ideal para este cultivo: aplicar cal agrícola (encalado) para subir el pH gradualmente.")
        elif gap_ph < -0.5:
            recs.append("⚪ **pH del suelo más alcalino** de lo ideal: aplicar azufre elemental o materia orgánica ácida para bajarlo.")

        if gap_rain > 40:
            recs.append("💧 **Lluvia/humedad insuficiente** para este cultivo en tu zona: planificar riego complementario (goteo o aspersión).")
        elif gap_rain < -40:
            recs.append("💧 **Exceso de lluvia** esperado para este cultivo: mejorar el drenaje de la parcela (camellones, canales) para evitar pudrición de raíz.")

        if not recs:
            recs.append("✅ Tu suelo está razonablemente alineado con los requerimientos del cultivo elegido — enfócate en buenas prácticas de manejo (rotación, monitoreo de plagas) más que en corregir nutrientes.")

        for r in recs:
            st.markdown(r)

        st.caption(
            "Recomendaciones generadas por reglas simples de brecha (gap) entre el perfil del suelo "
            "y el promedio histórico del cultivo en el dataset — a nivel de prototipo académico. "
            "En un producto real, estas reglas deberían ser validadas por un ingeniero agrónomo."
        )

    with col_alt:
        st.subheader("Si buscas alternativas")
        st.markdown("Cultivos más compatibles con **este mismo suelo**, según el modelo:")
        top_idx = np.argsort(probs)[::-1][:5]
        alt_df = pd.DataFrame({
            "cultivo": [fmt_crop(classes[i]) for i in top_idx],
            "probabilidad": [probs[i] for i in top_idx],
        })
        fig = px.bar(
            alt_df.sort_values("probabilidad"), x="probabilidad", y="cultivo", orientation="h",
            text_auto=".1%", color_discrete_sequence=[GREEN],
        )
        fig.update_xaxes(tickformat=".0%", title="Probabilidad estimada")
        fig.update_layout(yaxis_title="", height=320)
        st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------------
# TAB 5 — Clima real (API pública Open-Meteo) + mejor temporada de siembra
# ------------------------------------------------------------------
import clima_api as clima

with tab5:
    st.markdown(
        """
        **Fuente de datos:** [Open-Meteo](https://open-meteo.com) — API pública, gratuita,
        sin necesidad de clave de acceso. Trae temperatura, precipitación y humedad
        **reales** (no simuladas) de los últimos 3 años para calcular el clima típico de
        cada mes en tu ciudad, y así recomendar la mejor temporada de siembra.
        """
    )

    col_city, col_crop = st.columns([1, 1])
    with col_city:
        city_input = st.text_input("Ciudad de Ecuador", value="Portoviejo")
    with col_crop:
        crop_for_season = st.selectbox(
            "Cultivo a evaluar", sorted(df["label"].unique()),
            format_func=lambda c: ec.CROP_ES.get(c, (c, False))[0],
            key="crop_season",
        )

    buscar = st.button("🔎 Buscar clima real y recomendar temporada")

    if buscar:
        with st.spinner("Consultando Open-Meteo…"):
            try:
                matches = clima.geocode_city(f"{city_input}, Ecuador")
                if not matches:
                    matches = clima.geocode_city(city_input)
                if not matches:
                    st.error(
                        f"No se encontró la ciudad '{city_input}'. Intenta con otro nombre "
                        "(ej. 'Portoviejo', 'Quito', 'Guayaquil', 'Babahoyo')."
                    )
                    st.stop()

                place = matches[0]
                lat, lon = place["latitude"], place["longitude"]
                nombre_lugar = f"{place.get('name','')}, {place.get('admin1','')}, Ecuador"

                monthly = clima.get_monthly_climatology(lat, lon, years=3)
                st.session_state["monthly_climate"] = monthly
                st.session_state["place_name"] = nombre_lugar

            except requests.exceptions.RequestException as e:
                st.error(
                    "No se pudo conectar con la API de clima (revisa tu conexión a "
                    f"internet). Detalle técnico: {e}"
                )
                st.stop()
            except Exception as e:
                st.error(f"Ocurrió un problema al procesar los datos de clima: {e}")
                st.stop()

    if "monthly_climate" in st.session_state:
        monthly = st.session_state["monthly_climate"]
        st.success(f"Clima real obtenido para: **{st.session_state['place_name']}**")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly["mes_nombre"], y=monthly["precipitacion_mm"],
                                  name="Precipitación (mm)", marker_color="#2a78d6"))
            fig.update_layout(title="Precipitación mensual promedio (últimos 3 años)",
                               yaxis_title="mm/mes", height=350)
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=monthly["mes_nombre"], y=monthly["temperatura"],
                                       mode="lines+markers", name="Temperatura (°C)",
                                       line=dict(color="#e34948")))
            fig2.update_layout(title="Temperatura mensual promedio (últimos 3 años)",
                                yaxis_title="°C", height=350)
            st.plotly_chart(fig2, width='stretch')

        st.dataframe(
            monthly[["mes_nombre", "temperatura", "precipitacion_mm", "humedad", "temporada"]]
            .rename(columns={"mes_nombre": "Mes", "temperatura": "Temp. (°C)",
                              "precipitacion_mm": "Lluvia (mm)", "humedad": "Humedad (%)",
                              "temporada": "Temporada"})
            .round(1),
            width='stretch', hide_index=True,
        )

        st.markdown("---")
        st.subheader(f"¿Cuándo sembrar {ec.CROP_ES.get(crop_for_season, (crop_for_season, False))[0]}?")

        # Comparacion por PATRON RELATIVO (z-scores), no por unidades absolutas:
        # el "rainfall" del dataset no es directamente comparable en mm a la
        # precipitacion mensual real de la API, asi que comparamos si el mes es
        # relativamente mas humedo/calido/lluvioso de lo normal en esa ciudad, contra
        # si el cultivo generalmente prefiere condiciones por encima/debajo del
        # promedio de los demas cultivos del dataset.
        crop_row = df[df["label"] == crop_for_season][VAR_COLS].mean()
        gmean, gstd = df[VAR_COLS].mean(), df[VAR_COLS].std()
        crop_z = {
            "temperatura": (crop_row["temperature"] - gmean["temperature"]) / gstd["temperature"],
            "precipitacion_mm": (crop_row["rainfall"] - gmean["rainfall"]) / gstd["rainfall"],
            "humedad": (crop_row["humidity"] - gmean["humidity"]) / gstd["humidity"],
        }
        w = {
            "temperatura": feature_importance["temperature"],
            "precipitacion_mm": feature_importance["rainfall"],
            "humedad": feature_importance["humidity"],
        }
        w_total = sum(w.values())
        w = {k: v / w_total for k, v in w.items()}

        m_mean = monthly[["temperatura", "precipitacion_mm", "humedad"]].mean()
        m_std = monthly[["temperatura", "precipitacion_mm", "humedad"]].std().replace(0, 1)

        scores = []
        for _, row in monthly.iterrows():
            dist = 0.0
            for var in ["temperatura", "precipitacion_mm", "humedad"]:
                month_z = (row[var] - m_mean[var]) / m_std[var]
                dist += w[var] * (crop_z[var] - month_z) ** 2
            scores.append(np.sqrt(dist))
        monthly["compatibilidad"] = 1 / (1 + np.array(scores))

        fig3 = px.bar(
            monthly.sort_values("mes"), x="mes_nombre", y="compatibilidad",
            color="temporada", color_discrete_map={"Lluviosa": "#2a78d6", "Seca": "#eda100"},
            title="Compatibilidad estimada por mes para sembrar este cultivo",
        )
        fig3.update_layout(yaxis_title="Índice de compatibilidad (mayor = mejor)", xaxis_title="")
        st.plotly_chart(fig3, width='stretch')

        best_months = monthly.sort_values("compatibilidad", ascending=False).head(3)
        best_names = ", ".join(best_months["mes_nombre"])
        st.markdown(
            f"""
            <div style="background:#E4F1E9;border-left:5px solid {GREEN};border-radius:6px;padding:1rem 1.2rem">
                <p style="margin:0;color:{GREEN};font-weight:600">
                    📅 Los meses más recomendados para sembrar en {st.session_state['place_name'].split(',')[0]}
                    son: {best_names}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Metodología: se compara el patrón relativo de cada mes (¿es más cálido/lluvioso/húmedo "
            "que el promedio anual de esa ciudad?) contra si el cultivo, en general, prefiere "
            "condiciones por encima o por debajo del promedio de los demás cultivos del dataset — "
            "no se comparan unidades absolutas de lluvia del dataset contra mm/mes de la API, "
            "ya que representan cosas distintas (lluvia del ciclo del cultivo vs. lluvia mensual real)."
        )
    else:
        st.info("Ingresa una ciudad y pulsa el botón para traer el clima real desde la API.")

# ------------------------------------------------------------------
# TAB 6 — Asesor IA con contexto del modelo
# ------------------------------------------------------------------
with tab6:
    st.subheader("Asesor IA para alternativas de cultivo")
    st.write(
        "Consulta dudas sobre el suelo o pide opciones de cultivo. El asesor recibe el "
        "diagnóstico calculado por Random Forest y lo explica en lenguaje sencillo."
    )

    # Selección independiente para que el asesor pueda explorar escenarios.
    agent_region = st.selectbox("Región de referencia", list(ec.REGIONES.keys()), key="agent_region")
    agent_soil_key = st.selectbox(
        "Tipo de suelo", list(ec.SUELOS.keys()),
        format_func=lambda key: ec.SUELOS[key]["nombre"], key="agent_soil",
    )
    agent_crop = st.selectbox(
        "Cultivo que deseas evaluar", sorted(df["label"].unique()),
        format_func=lambda c: ec.CROP_ES.get(c, (c, False))[0], key="agent_crop",
    )

    agent_soil = ec.SUELOS[agent_soil_key]
    agent_climate = ec.REGIONES[agent_region]
    agent_vector = {
        **agent_soil["perfil"],
        "temperature": agent_climate["temperature"],
        "humidity": agent_climate["humidity"],
        "rainfall": agent_climate["rainfall"],
    }
    agent_X = pd.DataFrame([agent_vector])[VAR_COLS]
    agent_probs = rf_model.predict_proba(agent_X)[0]
    agent_top = np.argsort(agent_probs)[::-1][:5]
    agent_alternatives = [
        {
            "cultivo": ec.CROP_ES.get(classes[i], (classes[i], False))[0],
            "probabilidad_modelo": round(float(agent_probs[i]) * 100, 1),
        }
        for i in agent_top
    ]
    agent_desired_probability = float(agent_probs[list(classes).index(agent_crop)]) * 100

    st.caption(
        "Base del asesor: " + ", ".join(
            f"{item['cultivo']} ({item['probabilidad_modelo']}%)" for item in agent_alternatives
        )
    )
    question = st.text_area(
        "Pregunta al asesor",
        value="¿Qué cultivo alternativo me recomiendas y qué debo vigilar en este suelo?",
        height=100,
    )

    api_key = get_api_key(st.secrets)
    if not api_key:
        st.warning(
            "Para activar el asesor, configura `OPENAI_API_KEY` como variable de entorno "
            "o en `.streamlit/secrets.toml`. El diagnóstico y el Top 5 siguen funcionando "
            "sin IA."
        )

    if st.button("🤖 Pedir recomendación al asesor", type="primary", disabled=not api_key):
        agent_context = {
            "region": agent_region,
            "tipo_de_suelo": agent_soil["nombre"],
            "descripcion_suelo": agent_soil["descripcion"],
            "perfil_estimado": agent_vector,
            "cultivo_consultado": ec.CROP_ES.get(agent_crop, (agent_crop, False))[0],
            "compatibilidad_cultivo_consultado_pct": round(agent_desired_probability, 1),
            "alternativas_top_5_random_forest": agent_alternatives,
        }
        with st.spinner("El asesor está analizando el diagnóstico…"):
            try:
                answer = ask_agro_agent(question, agent_context, api_key)
                st.session_state["agent_answer"] = answer
            except Exception as error:
                st.error(f"No se pudo consultar el asesor: {error}")

    if st.session_state.get("agent_answer"):
        st.markdown(st.session_state["agent_answer"])
        st.caption(
            "El asesor interpreta la salida del modelo; no reemplaza un análisis de suelo "
            "ni la evaluación de un profesional agrónomo."
        )

st.markdown("---")
st.caption("Prototipo académico — AgroSmart EC · Proyecto Integrador de Analítica de Negocios · ULEAM")
