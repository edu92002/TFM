# TFM

Guía de Uso


📂 Estructura Final del Proyecto

![image](https://github.com/user-attachments/assets/abb2703c-36b3-4f22-afae-d13896ce1f0b)


Guía Paso a Paso
# 0. Descarga del Dataset
Descargar FakeInteractionDataset desde https://uma365-my.sharepoint.com/:f:/g/personal/fjbaldan_uma_es/EpOHI1BfAPZOv4mJ96M6CxIBwdzL4j2nohXDJXuiqEPORQ?e=bEIgTr.

Colocar la carpeta en la raíz del proyecto.

# 1. Ejecutar tfmglobal.py (Desde Raíz)
bash
Situarse en la carpeta raíz del proyecto
cd /ruta/a/TFM-Deteccion-Bots

Ejecutar (genera resultados/global/interacciones.csv)
python src/tfmglobal.py
Nota: Este script debe ejecutarse desde la raíz.

# 2. Ejecutar Scripts Restantes (Desde resultados/global/)
bash
Navegar a la carpeta de resultados
cd resultados/global/

2.1. Filtrar usuarios (genera filtrado.csv)
python ../../src/filtrado30.py

2.2. Extraer resultados con datos crudos
python ../../src/temporaleraw.py

2.2. Extraer resultados con datos temporales
python ../../src/temporales.py

2.3. Extraer resultados con datos temporales y texto
python ../../src/temporalestexto.py


tfmglobal.py → filtrado30.py → temporalesraw.py/temporales.py/temporalestexto.py.

Cada script depende del output del anterior.

Sin parámetros CLI:
Todos los scripts usan rutas predefinidas. Si necesitas modificarlas, edita los scripts directamente.

Resultados predecibles:

Se usan semillas aleatorias fijas (random_state=42) para reproducibilidad.

