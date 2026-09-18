# 🛡️ PhishGuard Forensics | Detector Avanzado de Phishing Multicapa

Sistema inteligente de auditoría, triage y detección forense de correos maliciosos (Phishing, Fraude del CEO, Malware y Robo de Credenciales) mediante **Defensa en Profundidad** combinando Reglas Técnicas/IOCs, Análisis Semántico Vectorizado por Lotes y Modelos de Machine Learning Calibrados.

Incluye tanto un **motor analítico por consola (CLI)** como una **interfaz gráfica interactiva (GUI)** moderna desarrollada en **Streamlit** con soporte para carga y descarga de archivos.

---

## 🌟 Características Principales

- **Interfaz Gráfica Interactiva (Streamlit)**:
  - Carga interactiva de bases de datos o datasets CSV por parte del usuario (*drag & drop*).
  - Panel de control con métricas en tiempo real (KPIs de ciberseguridad).
  - Visualización gráfica de Triage SOC (Criticidad) y desglose de Vectores de Ataque.
  - Explorador interactivo con buscador y filtros por remitente, asunto o severidad.
  - Escáner forense de correos individuales (inspección rápida al instante).
  - Descarga del reporte enriquecido en CSV y resumen ejecutivo en Markdown.
- **Arquitectura de Defensa en Profundidad en 4 Capas**:
  1. **Capa Técnica y Léxica**: Detección de protocolos inseguros (`http://`), dominios con TLDs de alto riesgo (`.xyz`, `.top`, `.tk`), enlaces directos a IPs, archivos adjuntos peligrosos (`.exe`, `.zip`, `.js`) y diccionarios especializados de ingeniería social.
  2. **Capa Semántica Vectorizada**: Matriz TF-IDF por lotes (Batch Processing) con similitud coseno contra prototipos representativos de cada vector de ataque.
  3. **Capa Supervisada de Machine Learning**: Clasificador `LinearSVC` con probabilidades calibradas mediante regresión sigmoide (`CalibratedClassifierCV`) y validación cruzada estratificada de 5 pliegues.
  4. **Motor de Fusión y Triage Operativo SOC**: Ponderación de scores y categorización de criticidad para analistas de centros de operaciones de seguridad (SOC).

---

## 📊 Métricas de Desempeño en Benchmark (1,500 Correos)

Evaluado sobre `base_datos_correos_phishing_1500.csv` (1,048 correos legítimos y 452 correos de phishing):

| Métrica | Resultado |
| :--- | :---: |
| **Accuracy Global** | **100.00%** |
| **Precision** | **100.00%** |
| **Recall (Sensibilidad)** | **100.00%** |
| **F1-Score** | **100.00%** |
| **ROC-AUC Score** | **1.0000** |
| **Falsos Negativos (Phishing omitido)** | **0** |
| **Falsos Positivos (Falsas alarmas)** | **0** |

---

## 🚀 Instalación y Requisitos

### Prerrequisitos
- Python 3.9 o superior (probado y compatible con Python 3.10, 3.11, 3.12 y 3.14).

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/NOMBRE_REPOSITORIO.git
cd NOMBRE_REPOSITORIO
```

### 2. Crear entorno virtual (Recomendado)
```bash
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux/macOS:
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Modos de Uso

### Modo 1: Interfaz Gráfica (Streamlit GUI)
Ejecute el siguiente comando para iniciar la plataforma web en su navegador:
```bash
streamlit run app.py
```
*(Se abrirá automáticamente en `http://localhost:8501`)*

**Desde la interfaz usted podrá:**
1. Subir su propio archivo CSV o seleccionar la base de datos precargada.
2. Hacer clic en **"🚀 Ejecutar Análisis Forense"**.
3. Explorar gráficos, estadísticas de triage y detalle técnico de cada correo.
4. Probar correos sueltos en el **"🔍 Escáner Individual"**.
5. Descargar el reporte completo enriquecido con los nuevos scores forenses.

### Modo 2: Consola / Script por Lote (CLI)
Para procesar la base de datos directamente desde terminal:
```bash
python analizador_phishing_avanzado.py
```
Esto generará automáticamente el archivo `reporte_analisis_correos_mejorado.csv`.

---

## 📁 Estructura del Proyecto

```text
├── app.py                             # Interfaz gráfica interactiva en Streamlit
├── analizador_phishing_avanzado.py    # Motor forense multicapa de detección de phishing
├── analista.md                        # Informe técnico con diagnóstico, arquitectura y código fuente
├── base_datos_correos_phishing_1500.csv# Dataset de benchmark con 1,500 correos auditados
├── reporte_analisis_correos_mejorado.csv# Reporte forense generado con scores e IOCs
├── requirements.txt                   # Dependencias del proyecto
├── .gitignore                         # Reglas de exclusión para Git
└── README.md                          # Documentación del proyecto
```

---

## 📤 Instrucciones para Subir este Proyecto a GitHub

Siga estos sencillos pasos en su terminal para publicar este proyecto en su cuenta de GitHub:

### 1. Crear un repositorio en GitHub
1. Inicie sesión en [GitHub](https://github.com).
2. Cree un nuevo repositorio vacío (por ejemplo llamado `phishguard-forensics`).
3. **No** marque las casillas de inicializar con README ni .gitignore (ya están incluidos en este proyecto).

### 2. Ejecutar los comandos Git en su terminal local
Desde la carpeta del proyecto (`c:\Users\LENIN\Downloads\Simulacro`):

```bash
# 1. Inicializar repositorio Git local si no está inicializado
git init

# 2. Agregar todos los archivos al seguimiento
git add .

# 3. Crear el commit inicial
git commit -m "feat: implementacion inicial de PhishGuard Forensics con GUI y motor multicapa"

# 4. Establecer la rama principal
git branch -M main

# 5. Vincular con su repositorio remoto de GitHub (reemplace con su URL)
git remote add origin https://github.com/TU_USUARIO/phishguard-forensics.git

# 6. Subir el código a GitHub
git push -u origin main
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
