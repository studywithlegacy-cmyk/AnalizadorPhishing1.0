"""
================================================================================
PHISHGUARD FORENSICS - PLATAFORMA DE DETECCIÓN AVANZADA DE PHISHING
================================================================================
Interfaz Gráfica Interactiva basada en Streamlit
================================================================================
"""

import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from analizador_phishing_avanzado import (
    analizar_base_correos_avanzada,
    DetectorTecnicoLexico,
    DetectorSemanticoVectorizado,
    normalizar_texto
)

# ------------------------------------------------------------------------------
# Configuración general de la página Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard SOC | Detector Forense de Phishing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para apariencia SOC / Cybersecurity
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: bold;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Barra lateral (Sidebar) y selección de modo
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.title("PhishGuard SOC")
    st.caption("Sistema Forense Multicapa de Ciberseguridad")
    st.markdown("---")

    modo = st.radio(
        "Navegación",
        ["📊 Análisis de Dataset / CSV", "🔍 Escáner Individual", "📖 Documentación"],
        index=0
    )
    st.markdown("---")

    if modo == "📊 Análisis de Dataset / CSV":
        st.subheader("📁 Fuente de Datos")
        origen_datos = st.radio(
            "Seleccione el origen:",
            ["Subir archivo CSV propio", "Usar base predeterminada (1,500 correos)"],
            index=1
        )

        archivo_subido = None
        if origen_datos == "Subir archivo CSV propio":
            archivo_subido = st.file_uploader(
                "Arrastre o cargue su archivo CSV",
                type=["csv"],
                help="El archivo puede contener columnas como: asunto, cuerpo, remitente, texto, es_phishing, etc."
            )
        else:
            default_path = "base_datos_correos_phishing_1500.csv"
            if os.path.exists(default_path):
                st.success("✔ Base predeterminada lista (1,500 registros).")
            else:
                st.warning("No se encontró el archivo predeterminado en el directorio.")

        st.markdown("---")
        btn_analizar = st.button("🚀 Ejecutar Análisis Forense", type="primary", use_container_width=True)


# ------------------------------------------------------------------------------
# MODO 1: ANÁLISIS DE DATASET / CSV COMPLETO
# ------------------------------------------------------------------------------
if modo == "📊 Análisis de Dataset / CSV":
    st.markdown('<div class="main-title">🛡️ Panel de Control Forense de Phishing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Auditoría, Triage y Detección en Profundidad con Inteligencia Artificial y Reglas Técnicas</div>', unsafe_allow_html=True)

    # Estado de sesión para almacenar resultados
    if 'reporte_df' not in st.session_state:
        st.session_state['reporte_df'] = None
        st.session_state['metricas'] = {}

    # Ejecutar análisis si se pulsa el botón o si no hay datos cargados
    if btn_analizar or (st.session_state['reporte_df'] is None and os.path.exists("reporte_analisis_correos_mejorado.csv")):
        fuente = None
        if btn_analizar:
            if origen_datos == "Subir archivo CSV propio":
                if archivo_subido is None:
                    st.error("Por favor suba un archivo CSV antes de ejecutar el análisis.")
                else:
                    fuente = archivo_subido
            else:
                fuente = "base_datos_correos_phishing_1500.csv"
        elif os.path.exists("reporte_analisis_correos_mejorado.csv"):
            # Carga rápida del reporte pre-calculado para visualización instantánea
            fuente = "base_datos_correos_phishing_1500.csv"

        if fuente is not None:
            with st.spinner("Procesando corpus con arquitectura multicapa (Léxica, Semántica y Machine Learning)..."):
                start_time = time.time()
                df_rep, metrics = analizar_base_correos_avanzada(
                    ruta_o_df=fuente,
                    columna_etiqueta='es_phishing',
                    ruta_salida_reporte='reporte_analisis_correos_mejorado.csv'
                )
                elapsed = time.time() - start_time
                st.session_state['reporte_df'] = df_rep
                st.session_state['metricas'] = metrics
                st.session_state['elapsed_time'] = elapsed
                if btn_analizar:
                    st.success(f"¡Análisis completado exitosamente en {elapsed:.2f} segundos!")

    df = st.session_state['reporte_df']

    if df is not None:
        total_correos = len(df)
        total_phishing = int(df['es_phishing_detectado'].sum())
        total_legit = total_correos - total_phishing
        tasa_phishing = (total_phishing / total_correos) * 100 if total_correos > 0 else 0

        # Tarjetas Métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Analizados", f"{total_correos:,}")
        with col2:
            st.metric("Phishing Confirmado", f"{total_phishing:,}", delta=f"{tasa_phishing:.1f}%", delta_color="inverse")
        with col3:
            st.metric("Correos Legítimos", f"{total_legit:,}")
        with col4:
            criticos = (df['clasificacion_triage'].str.contains('CRÍTICO')).sum()
            st.metric("Amenazas Críticas", f"{criticos:,}")
        with col5:
            acc = st.session_state['metricas'].get('accuracy', 1.0) * 100
            st.metric("Precisión / Accuracy", f"{acc:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Pestañas de Navegación del Reporte
        tab_graficos, tab_tabla, tab_detalle, tab_descargas = st.tabs([
            "📈 Visualizaciones y Triage",
            "📋 Explorador de Correos",
            "🔬 Inspección Forense Individual",
            "📥 Descarga de Informes"
        ])

        # ----------------------------------------------------------------------
        # Pestaña 1: Gráficos y Triage
        # ----------------------------------------------------------------------
        with tab_graficos:
            st.subheader("Distribución Operativa del Triage SOC")
            c_graf1, c_graf2 = st.columns([1, 1])

            with c_graf1:
                st.markdown("**Nivel de Criticidad Asignado**")
                triage_counts = df['clasificacion_triage'].value_counts().reset_index()
                triage_counts.columns = ['Triage', 'Cantidad']
                
                chart_triage = alt.Chart(triage_counts).mark_bar(cornerRadius=6).encode(
                    x=alt.X('Cantidad:Q', title="Número de Correos"),
                    y=alt.Y('Triage:N', sort='-x', title=None),
                    color=alt.Color('Triage:N', legend=None, scale=alt.Scale(
                        domain=[
                            'CRÍTICO - Phishing Confirmado (Cuarentena)',
                            'ALTO - Probable Phishing (Bloqueo Preventivo)',
                            'MEDIO - Sospechoso (Revisión Analista SOC)',
                            'BAJO - Correo Legítimo'
                        ],
                        range=['#DC2626', '#F97316', '#FBBF24', '#10B981']
                    )),
                    tooltip=['Triage', 'Cantidad']
                ).properties(height=280)
                st.altair_chart(chart_triage, use_container_width=True)

            with c_graf2:
                st.markdown("**Vectores de Ataque Identificados**")
                vector_counts = df['vector_ataque_estimado'].value_counts().reset_index()
                vector_counts.columns = ['Vector', 'Cantidad']

                chart_vector = alt.Chart(vector_counts).mark_bar(cornerRadius=6, color="#4F46E5").encode(
                    x=alt.X('Cantidad:Q', title="Correos"),
                    y=alt.Y('Vector:N', sort='-x', title=None),
                    tooltip=['Vector', 'Cantidad']
                ).properties(height=280)
                st.altair_chart(chart_vector, use_container_width=True)

            st.markdown("---")
            st.markdown("**Distribución de Puntuaciones de Riesgo Compuesto (0.0 a 1.0)**")
            chart_hist = alt.Chart(df).mark_bar(bin=alt.Bin(maxbins=30), color="#2563EB").encode(
                x=alt.X('score_riesgo_compuesto:Q', title="Score de Riesgo Compuesto"),
                y=alt.Y('count():Q', title="Frecuencia"),
                tooltip=['count()']
            ).properties(height=240)
            st.altair_chart(chart_hist, use_container_width=True)

        # ----------------------------------------------------------------------
        # Pestaña 2: Tabla Filtrable
        # ----------------------------------------------------------------------
        with tab_tabla:
            st.subheader("Base de Correos Analizados")
            
            f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
            with f_col1:
                filtro_triage = st.selectbox(
                    "Filtrar por Triage:",
                    ["Todos"] + df['clasificacion_triage'].unique().tolist()
                )
            with f_col2:
                filtro_vector = st.selectbox(
                    "Filtrar por Vector:",
                    ["Todos"] + df['vector_ataque_estimado'].unique().tolist()
                )
            with f_col3:
                busqueda = st.text_input("🔍 Buscar en Asunto o Remitente:", "")

            df_filtrado = df.copy()
            if filtro_triage != "Todos":
                df_filtrado = df_filtrado[df_filtrado['clasificacion_triage'] == filtro_triage]
            if filtro_vector != "Todos":
                df_filtrado = df_filtrado[df_filtrado['vector_ataque_estimado'] == filtro_vector]
            if busqueda.strip():
                mask_asunto = df_filtrado['asunto'].astype(str).str.contains(busqueda, case=False, na=False) if 'asunto' in df_filtrado.columns else False
                mask_rem = df_filtrado['remitente'].astype(str).str.contains(busqueda, case=False, na=False) if 'remitente' in df_filtrado.columns else False
                df_filtrado = df_filtrado[mask_asunto | mask_rem]

            st.caption(f"Mostrando {len(df_filtrado):,} de {len(df):,} correos")
            st.dataframe(
                df_filtrado,
                use_container_width=True,
                height=450
            )

        # ----------------------------------------------------------------------
        # Pestaña 3: Inspección Individual Detallada
        # ----------------------------------------------------------------------
        with tab_detalle:
            st.subheader("Auditoría Forense por Correo")
            st.caption("Seleccione un correo para revisar sus indicadores técnicos, scores por capa y evidencia forense.")
            
            id_col = 'id' if 'id' in df.columns else df.columns[0]
            ids_disponibles = df[id_col].tolist()
            
            sel_id = st.selectbox("Seleccionar ID de correo a inspeccionar:", ids_disponibles[:300])
            registro = df[df[id_col] == sel_id].iloc[0]

            det_col1, det_col2 = st.columns([1, 1])
            with det_col1:
                st.markdown(f"**Remitente:** `{registro.get('remitente', 'N/A')}`")
                st.markdown(f"**Asunto:** {registro.get('asunto', 'N/A')}")
                st.markdown(f"**Triage Asignado:** `{registro.get('clasificacion_triage', 'N/A')}`")
                st.markdown(f"**Vector Estimado:** `{registro.get('vector_ataque_estimado', 'N/A')}`")

            with det_col2:
                st.metric("Score Compuesto Final", f"{registro.get('score_riesgo_compuesto', 0.0):.3f}")
                st.write(f"- Score Léxico: `{registro.get('score_lexico', 'N/A')}`")
                st.write(f"- Score Semántico: `{registro.get('score_semantico', 'N/A')}`")
                st.write(f"- Score Probabilidad ML: `{registro.get('score_ml_prob', 'N/A')}`")

            st.markdown("**Hallazgos Forenses y Artefactos Detectados:**")
            hallazgos_txt = registro.get('hallazgos_tecnicos', 'Ninguno')
            if hallazgos_txt and hallazgos_txt != "Ninguno":
                for h in hallazgos_txt.split("; "):
                    st.warning(f"⚠️ {h}")
            else:
                st.success("✔ No se detectaron artefactos técnicos sospechosos en este correo.")

        # ----------------------------------------------------------------------
        # Pestaña 4: Descargas
        # ----------------------------------------------------------------------
        with tab_descargas:
            st.subheader("Exportación de Resultados Forenses")
            
            # Exportar a CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_data = csv_buffer.getvalue().encode('utf-8-sig')

            st.download_button(
                label="📥 Descargar Base Completa Enriquecida (CSV)",
                data=csv_data,
                file_name="reporte_analisis_correos_mejorado.csv",
                mime="text/csv",
                type="primary"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Resumen Ejecutivo de la Auditoría")
            
            resumen_md = f"""# Resumen Ejecutivo de Auditoría de Phishing
- **Fecha de Análisis**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total de Correos Evaluados**: {total_correos:,}
- **Ataques de Phishing Confirmados**: {total_phishing:,} ({tasa_phishing:.1f}%)
- **Correos Legítimos Identificados**: {total_legit:,}

## Triage de Criticidad SOC
{df['clasificacion_triage'].value_counts().to_markdown()}

## Desglose de Vectores de Ataque
{df['vector_ataque_estimado'].value_counts().to_markdown()}
"""
            st.code(resumen_md, language="markdown")
            st.download_button(
                label="📄 Descargar Resumen Ejecutivo (Markdown)",
                data=resumen_md.encode('utf-8'),
                file_name="resumen_ejecutivo_phishing.md",
                mime="text/markdown"
            )
    else:
        st.info("👆 Seleccione el origen de datos en la barra lateral y presione 'Ejecutar Análisis Forense'.")


# ------------------------------------------------------------------------------
# MODO 2: ESCÁNER FORENSE INDIVIDUAL
# ------------------------------------------------------------------------------
elif modo == "🔍 Escáner Individual":
    st.markdown('<div class="main-title">🔍 Escáner Forense de Correo Individual</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Pegue los detalles de un correo electrónico sospechoso para obtener un dictamen instantáneo.</div>', unsafe_allow_html=True)

    c_in1, c_in2 = st.columns([1, 1])
    with c_in1:
        rem_input = st.text_input("Remitente (Email):", placeholder="ejemplo: soporte@banco-seguridad-alerta.xyz")
    with c_in2:
        asunto_input = st.text_input("Asunto:", placeholder="ejemplo: URGENTE: Notificación de suspensión de cuenta")

    cuerpo_input = st.text_area(
        "Cuerpo del Mensaje:",
        placeholder="Pegue aquí el texto o contenido del correo sospechoso...",
        height=200
    )

    if st.button("⚡ Analizar Este Correo", type="primary"):
        if not cuerpo_input.strip() and not asunto_input.strip():
            st.warning("Por favor ingrese al menos el asunto o cuerpo del correo.")
        else:
            texto_unido = f"{asunto_input} {cuerpo_input}"
            
            # 1. Capa Léxica / Técnica
            det_lex = DetectorTecnicoLexico()
            res_lex = det_lex.analizar(asunto_input, cuerpo_input, rem_input)
            
            # 2. Capa Semántica
            det_sem = DetectorSemanticoVectorizado()
            scores_sem, alertas_sem = det_sem.analizar_lote(pd.Series([texto_unido]))
            score_sem = scores_sem[0]

            # Score combinado heurístico
            score_compuesto = round((0.45 * res_lex['score_lexico']) + (0.55 * score_sem), 3)

            # Hallazgos
            hallazgos = res_lex['hallazgos_lexicos']
            if any("ejecutable" in h for h in hallazgos):
                score_compuesto = max(score_compuesto, 0.75)

            st.markdown("---")
            st.subheader("Resultado del Análisis")

            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                if score_compuesto >= 0.65:
                    st.error("🚨 CRÍTICO: PHISHING CONFIRMADO")
                    dictamen = "Bloqueo inmediato, aislar remitente y aplicar cuarentena."
                elif score_compuesto >= 0.40:
                    st.warning("⚠️ ALTO: PROBABLE PHISHING")
                    dictamen = "Bloqueo preventivo y análisis de sandbox."
                elif score_compuesto >= 0.22:
                    st.info("🟡 MEDIO: SOSPECHOSO")
                    dictamen = "Requiere revisión manual por analista de seguridad."
                else:
                    st.success("✅ BAJO: CORREO LEGÍTIMO")
                    dictamen = "No se detectaron indicadores evidentes de fraude."

            with r_col2:
                st.metric("Puntuación de Riesgo", f"{score_compuesto:.3f}")

            with r_col3:
                st.markdown(f"**Recomendación SOC:**\n{dictamen}")

            st.markdown("#### Desglose de Indicadores Técnicos Detectados")
            if hallazgos:
                for h in hallazgos:
                    st.markdown(f"- ⚠️ **{h}**")
            else:
                st.markdown("- ✔ Ningún indicador técnico de compromiso detectado.")


# ------------------------------------------------------------------------------
# MODO 3: DOCUMENTACIÓN
# ------------------------------------------------------------------------------
elif modo == "📖 Documentación":
    st.markdown('<div class="main-title">📖 Arquitectura y Metodología PhishGuard</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Defensa en Profundidad de 4 Capas

    1. **Capa Técnica y Léxica (`DetectorTecnicoLexico`)**:
       - Inspecciona indicadores técnicos de compromiso (IOCs): enlaces HTTP inseguros, TLDs de alto riesgo (`.xyz`, `.top`, `.tk`), enlaces directos a direcciones IP y archivos adjuntos peligrosos (`.exe`, `.zip`, `.js`).
       - Evalúa patrones de ingeniería social en categorías: *Urgencia extrema*, *Captura de credenciales*, *Fraude de facturas / cobro judicial*, *CEO Spear Phishing* y *Paquetería*.
    
    2. **Capa Semántica Vectorizada (`DetectorSemanticoVectorizado`)**:
       - Modela representaciones vectoriales TF-IDF con n-gramas (1, 2) y sublinear TF.
       - Realiza una comparación por lotes (Batch processing) contra una base representativa de ataques prototipo utilizando la similitud del coseno acelerada en C.
    
    3. **Capa Supervisada de Machine Learning (`ModeloSupervisadoPhishing`)**:
       - Clasificador `LinearSVC` con pesos balanceados, calibrado mediante `CalibratedClassifierCV` (regresión sigmoide) para emitir probabilidades reales $P(\\text{Phishing} \\mid \\text{Correo})$.
       - Validación cruzada estratificada de 5 pliegues para evitar sobreajuste y fuga de datos.
    
    4. **Motor de Fusión y Triage Operativo SOC**:
       - Ensamble ponderado de puntuaciones:
         $$\\text{Score Compuesto} = 0.25 \\times S_{\\text{léxico}} + 0.35 \\times S_{\\text{semántico}} + 0.40 \\times P_{\\text{ML}}$$
       - Asignación de niveles operativos de respuesta SOC: `CRÍTICO`, `ALTO`, `MEDIO` y `BAJO`.
    """)
