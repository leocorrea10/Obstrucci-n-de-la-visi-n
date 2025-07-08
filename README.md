# Obstrucci-n-de-la-visi-n

Sistema Inteligente de Detección de Obstrucción Visual en Tiempo Real

Este proyecto presenta un sistema avanzado de visión por computadora, diseñado para detectar en tiempo real la obstrucción de los ojos humanos, ya sea por ojos cerrados, manos cubriéndolos o el uso de gafas oscuras.Su función principal es alertar mediante una alarma sonora cuando se detecta una pérdida de visibilidad sostenida, permitiendo prevenir accidentes o mejorar el monitoreo de atención visual.

⚙️Tecnologías Utilizadas

.Python para la lógica de detección.

.OpenCV para procesamiento de video en tiempo real.

.MediaPipe (Face Mesh & Hands) para detectar rostros y manos con alta precisión.

.NumPy para cálculos geométricos.

.Pygame para la emisión de alertas sonoras.

🌀  Cómo Funciona

.El sistema captura video en tiempo real desde la cámara.

.Detecta el rostro y calcula el EAR (Eye Aspect Ratio) para determinar si los ojos están cerrados.

.Verifica si hay manos cubriendo los ojos mediante la distancia relativa.

.Analiza el brillo del área ocular para detectar gafas oscuras.

.Si alguna condición persiste por más de 0.5 segundos, activa una alarma sonora y muestra mensajes en pantalla.

🔎 Casos de Uso Reales

-Asistencia en la conducción vehicular: Detecta si el conductor pierde la visión o se queda dormido.

-Monitoreo de seguridad laboral: Prevención de accidentes en entornos de alto riesgo.

-Vigilancia educativa o exámenes online: Asegura la atención y visibilidad del usuario.

-Apoyo en salud y rehabilitación visual: Seguimiento en terapias para problemas oculares o neurológicos.

🧠Este proyecto representa un paso importante hacia soluciones de monitoreo visual inteligentes, accesibles y efectivas para múltiples industrias.
