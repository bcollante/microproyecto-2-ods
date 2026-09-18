# Microproyecto 2: clasificación de textos por ODS

Proyecto de Machine Learning No Supervisado para preparar textos con TF-IDF, construir tópicos mediante LSA y clasificar textos según los ODS disponibles en el conjunto de datos.

## Inicio rápido en Visual Studio Code

### 1. Abrir la carpeta

```powershell
cd microproyecto-2-ods
code .
```

### 2. Crear el entorno virtual

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name microproyecto2-ods --display-name "Python - Microproyecto 2 ODS"
```

También puede ejecutarse `powershell -ExecutionPolicy Bypass -File setup.ps1`.

### 3. Configurar VS Code

Instalar las extensiones recomendadas cuando VS Code las sugiera. Después:

1. Abrir `notebooks/microproyecto2_ods.ipynb`.
2. Seleccionar el kernel `Python - Microproyecto 2 ODS`.
3. Abrir Codex desde su icono o ejecutar `Codex: Open Codex Sidebar` en la paleta de comandos.
4. Comprobar las skills disponibles escribiendo `/skills` o `$` en el chat de Codex.

Las skills específicas del proyecto están en `.agents/skills` y Codex las detecta al abrir la carpeta. Si no aparecen, reiniciar VS Code/Codex.

## Primer prompt recomendado

```text
Lee AGENTS.md, CHECKLIST.md y el enunciado de references/Microproyecto2.pdf.
Usa $microproyecto-mlns. Revisa la estructura del repositorio y dime qué se
encuentra listo para iniciar la Fase 1. No modifiques archivos todavía.
```

## Desarrollo por etapas

1. Contexto, objetivo y alcance.
2. Carga y auditoría del dataset.
3. Análisis exploratorio.
4. División estratificada train/test.
5. Preparación TF-IDF y pipeline.
6. LSA e interpretación de tópicos.
7. Clasificación y búsqueda de hiperparámetros.
8. Evaluación final sobre test.
9. Cuatro o más predicciones explicadas.
10. Conclusión, revisión y exportación.
11. Streamlit opcional.

## Estructura

- `data/`: dataset del proyecto.
- `references/`: enunciado y tutoriales del curso.
- `notebooks/`: notebook de trabajo.
- `src/`: funciones reutilizables.
- `outputs/`: figuras y tablas generadas.
- `models/`: modelos serializados; no se versionan por defecto.
- `entregables/`: versiones finales `.ipynb` y `.html`.
- `app/`: aplicación Streamlit opcional.
- `.agents/skills/`: instrucciones especializadas para Codex.

## Revisión final

Antes de entregar, invocar:

```text
Usa $revisar-notebook-rubrica para comparar el notebook ejecutado con
references/Microproyecto2.pdf. No modifiques archivos; entrega primero la matriz
de cumplimiento y los faltantes.
```

