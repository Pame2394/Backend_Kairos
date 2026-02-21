from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import tempfile
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from reportlab.pdfgen import canvas
import openpyxl
import logging
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
import time
import concurrent.futures

# Try modern GenAI SDK first, fall back to legacy
GENAI_SDK = None
genai_modern = None
genai_legacy = None
try:
    import google.genai as genai_modern
    GENAI_SDK = "modern"
except Exception:
    genai_modern = None
    try:
        import google.generativeai as genai_legacy
        GENAI_SDK = "legacy"
    except Exception:
        genai_legacy = None
        GENAI_SDK = None

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE = os.getenv("USE_GOOGLE", "false").lower() in ("1", "true", "yes")
if GOOGLE_API_KEY and USE_GOOGLE:
    try:
        if GENAI_SDK == "modern" and genai_modern is not None:
            # modern SDK often uses a Client object; we'll prefer that at call time
            logger.info("Usando SDK moderno google.genai para Gemini")
        elif GENAI_SDK == "legacy" and genai_legacy is not None:
            try:
                genai_legacy.configure(api_key=GOOGLE_API_KEY)
                logger.info("google.generativeai configurado con la clave proporcionada")
            except Exception:
                logger.exception("No se pudo configurar google.generativeai con la clave proporcionada")
        else:
            logger.warning("No se detectó ninguna SDK de GenAI instalada (ni modern ni legacy)")
    except Exception:
        logger.exception("Error al intentar configurar GenAI SDK")

app = FastAPI(title="backendKiros - Cotizaciones con IA")
# CORS: permitir peticiones desde el frontend en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuoteRequest(BaseModel):
    servicio: str = Field(..., min_length=1)
    horas: int = Field(1, ge=1, description="Número de horas, entero >= 1")
    complejidad: Literal["baja", "media", "alta"] = "media"
    cliente: Optional[str] = Field("Cliente")
    nombre: Optional[str] = Field(None)
    telefono: Optional[str] = Field(None)
    correo: Optional[str] = Field(None)
    fecha: Optional[str] = Field(None)
    condiciones: Optional[str] = Field(None)
    vigencia: Optional[str] = Field("15 días")


def calcular_precio(horas: int, complejidad: str) -> int:
    base = 1000
    if complejidad == "alta":
        return base + horas * 200
    if complejidad == "baja":
        return base + horas * 50
    return base + horas * 100


async def generar_proforma_ai(
    nombre: str,
    telefono: str,
    correo: str,
    servicio: str,
    horas: int,
    complejidad: str,
    precio: int,
    fecha: str = None,
    condiciones: str = None,
    vigencia: str = "15 días"
) -> str:
    """
    Prompt optimizado para que el LLM genere una proforma
    que coincida con los datos del formulario del cliente.
    """
    prompt = (
        f"Genera una proforma profesional en formato de cotización oficial.\n\n"
        f"Encabezado:\n"
        f"- Empresa emisora: [Nombre de la empresa]\n"
        f"- Fecha: {fecha if fecha else '[Fecha actual]'}\n"
        f"- Número de Proforma: PRF-2026-XXXX\n\n"
        f"Datos del cliente:\n"
        f"- Nombre completo: {nombre}\n"
        f"- Teléfono: {telefono}\n"
        f"- Correo electrónico: {correo}\n\n"
        f"Detalles del servicio solicitado:\n"
        f"- Servicio: {servicio}\n"
        f"- Horas estimadas: {horas}\n"
        f"- Complejidad: {complejidad}\n"
        f"- Precio total: ${precio}\n\n"
        f"Instrucciones para el formato:\n"
        f"1. Organiza la información en secciones: Encabezado, Datos del Cliente, Detalles del Servicio, Resumen de Costos, Condiciones, Nota Final.\n"
        f"2. Usa un tono formal y claro.\n"
        f"3. Incluye una breve descripción del alcance del servicio.\n"
        f"4. Añade condiciones: {condiciones if condiciones else '[Condiciones de pago y entrega]'}.\n"
        f"5. Indica la vigencia de la oferta: {vigencia}.\n"
        f"6. Finaliza con la nota: 'Esta es una cotización referencial, sujeta a ajustes.'\n"
        f"7. Formatea la salida como documento listo para enviar por correo y exportar a PDF/Excel."
    )

    if GOOGLE_API_KEY and USE_GOOGLE:
        max_retries = int(os.getenv("GENAI_MAX_RETRIES", "3"))
        timeout_sec = float(os.getenv("GENAI_TIMEOUT", "10.0"))
        backoff_base = float(os.getenv("GENAI_BACKOFF_BASE", "1.0"))

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Intentando generar proforma con GenAI (sdk=%s) intento %d/%d", GENAI_SDK, attempt, max_retries)

                def _call_genai() -> Any:
                    # Modern SDK: google.genai
                    if GENAI_SDK == "modern" and genai_modern is not None:
                        client_gen = genai_modern.Client(api_key=GOOGLE_API_KEY)
                        model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
                        return client_gen.models.generate_content(model=model_name, contents=prompt)

                    # Legacy SDK: google.generativeai
                    if GENAI_SDK == "legacy" and genai_legacy is not None:
                        model_obj = genai_legacy.GenerativeModel(os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"))
                        return model_obj.generate_content(prompt)

                    raise RuntimeError("No hay SDK GenAI disponible para realizar la llamada")

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_call_genai)
                    resp = fut.result(timeout=timeout_sec)

                # extraer texto de la respuesta de forma robusta
                text = None
                if isinstance(resp, dict):
                    candidates = resp.get("candidates") or resp.get("outputs") or []
                    if candidates and isinstance(candidates, list) and candidates[0]:
                        cand = candidates[0]
                        if isinstance(cand, dict):
                            text = cand.get("content") or cand.get("text") or cand.get("output")
                    text = text or resp.get("output_text") or resp.get("text") or resp.get("content")
                else:
                    # modern/legacy SDK: .text is the simplest accessor
                    text = getattr(resp, "text", None) or getattr(resp, "output_text", None) or getattr(resp, "outputs", None) or getattr(resp, "content", None)

                if text:
                    logger.info("Proforma generada por GenAI (long=%d)", len(str(text)))
                    return str(text).strip()
                else:
                    logger.warning("GenAI devolvió respuesta sin texto útil (intento %d): %r", attempt, resp)
                    last_exc = RuntimeError("Respuesta vacía de GenAI")
            except concurrent.futures.TimeoutError:
                last_exc = TimeoutError(f"Timeout al llamar a GenAI después de {timeout_sec}s")
                logger.warning("Timeout en llamada a GenAI (intento %d/%d)", attempt, max_retries)
            except Exception as e:
                last_exc = e
                logger.exception("Error en intento de GenAI (intento %d/%d): %s", attempt, max_retries, e)

            # backoff exponencial antes del siguiente intento
            if attempt < max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                logger.info("Esperando %.1fs antes del siguiente intento", sleep_time)
                time.sleep(sleep_time)

        # si llegamos aquí, todos los intentos fallaron
        logger.error("Todos los intentos a GenAI fallaron: última excepción: %s", last_exc)

    # Fallback local si Gemini no está configurado o falló
    return (
        f"Proforma para {nombre}: Servicio {servicio}, {horas} horas, "
        f"complejidad {complejidad}. Precio estimado: {precio} dólares.\n\n"
        "Gracias por su interés. Esta proforma fue generada localmente."
    )


def _cleanup_file(path: str) -> None:
    try:
        os.unlink(path)
        logger.info("Archivo temporal eliminado: %s", path)
    except Exception:
        logger.exception("No se pudo eliminar el archivo temporal: %s", path)


@app.post("/cotizar")
async def cotizar(payload: QuoteRequest):
    precio = calcular_precio(payload.horas, payload.complejidad)
    proforma = await generar_proforma_ai(
        nombre=payload.nombre or payload.cliente or "Cliente",
        telefono=payload.telefono or "No especificado",
        correo=payload.correo or "No especificado",
        servicio=payload.servicio,
        horas=payload.horas,
        complejidad=payload.complejidad,
        precio=precio,
        fecha=payload.fecha,
        condiciones=payload.condiciones,
        vigencia=payload.vigencia or "15 días",
    )
    return JSONResponse({"precio": precio, "proforma": proforma})


@app.post("/cotizar_pdf")
async def cotizar_pdf(payload: QuoteRequest, background_tasks: BackgroundTasks):
    precio = calcular_precio(payload.horas, payload.complejidad)
    proforma = await generar_proforma_ai(
        nombre=payload.nombre or payload.cliente or "Cliente",
        telefono=payload.telefono or "No especificado",
        correo=payload.correo or "No especificado",
        servicio=payload.servicio,
        horas=payload.horas,
        complejidad=payload.complejidad,
        precio=precio,
        fecha=payload.fecha,
        condiciones=payload.condiciones,
        vigencia=payload.vigencia or "15 días",
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    filename = tmp.name
    tmp.close()

    c = canvas.Canvas(filename)
    c.setFont("Helvetica", 12)
    c.drawString(100, 780, f"Cotización para: {payload.nombre or payload.cliente}")
    c.drawString(100, 760, f"Servicio: {payload.servicio}")
    c.drawString(100, 740, f"Horas: {payload.horas}")
    c.drawString(100, 720, f"Complejidad: {payload.complejidad}")
    c.drawString(100, 700, f"Precio estimado: {precio} dólares")
    c.drawString(100, 660, "Proforma:")

    textobject = c.beginText(100, 640)
    for line in proforma.splitlines():
        textobject.textLine(line)
    c.drawText(textobject)
    c.showPage()
    c.save()

    background_tasks.add_task(_cleanup_file, filename)
    return FileResponse(filename, media_type="application/pdf", filename="cotizacion.pdf")


@app.post("/cotizar_excel")
async def cotizar_excel(payload: QuoteRequest, background_tasks: BackgroundTasks):
    precio = calcular_precio(payload.horas, payload.complejidad)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Cotización"
    ws.append(["Cliente", "Teléfono", "Correo", "Servicio", "Horas", "Complejidad", "Precio"])
    ws.append([payload.nombre or payload.cliente, payload.telefono, payload.correo, payload.servicio, payload.horas, payload.complejidad, precio])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    filename = tmp.name
    tmp.close()

    wb.save(filename)

    background_tasks.add_task(_cleanup_file, filename)
    return FileResponse(filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="cotizacion.xlsx")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)), reload=True)
