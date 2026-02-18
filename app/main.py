from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from reportlab.pdfgen import canvas
import openpyxl
import logging
from pydantic import BaseModel, Field
from typing import Literal, Optional
import google.generativeai as genai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() in ("1", "true", "yes")
client = None
if OPENAI_API_KEY and USE_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY)
# Google Gemini configuration (optional)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE = os.getenv("USE_GOOGLE", "false").lower() in ("1", "true", "yes")
if GOOGLE_API_KEY and USE_GOOGLE:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception("No se pudo configurar google.generativeai con la clave proporcionada")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def calcular_precio(horas: int, complejidad: str) -> int:
    base = 1000
    if complejidad == "alta":
        return base + horas * 200
    if complejidad == "baja":
        return base + horas * 50
    return base + horas * 100


async def generar_proforma_ai(cliente: str, servicio: str, horas: int, complejidad: str, precio: int) -> str:
    # Si el usuario desea usar Google Gemini y hay clave, intentarlo primero
    prompt = (
        f"Genera una proforma profesional para el servicio: {servicio}. "
        f"Cliente: {cliente}. Horas estimadas: {horas}. Complejidad: {complejidad}. "
        f"Precio total: ${precio}. Nota: Esta es una cotización referencial, sujeta a ajustes."
    )

    if GOOGLE_API_KEY and USE_GOOGLE:
        try:
            logger.info("Intentando generar proforma con Google Gemini (GenerativeModel)")
            model_obj = genai.GenerativeModel("gemini-2.5-flash")
            resp = model_obj.generate_content(prompt)
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
                # respuesta podría ser un objeto con atributos
                text = getattr(resp, "content", None) or getattr(resp, "text", None) or getattr(resp, "output_text", None)

            if text:
                logger.info("Proforma generada por Gemini (long=%d)", len(text))
                return text.strip()
            else:
                logger.warning("Gemini devolvió respuesta sin texto útil: %r", resp)
        except Exception as e:
            logger.exception("Error generando proforma con Google Gemini: %s", e)

    # Si el modo IA de OpenAI está desactivado o no hay clave, generar localmente
    if not OPENAI_API_KEY or not USE_OPENAI or not client:
        return (
            f"Proforma para {cliente}: Servicio {servicio}, {horas} horas, "
            f"complejidad {complejidad}. Precio estimado: {precio} dólares.\n\n"
            "Gracias por su interés. Esta proforma fue generada localmente sin usar servicios externos."
        )

    # Intentar con OpenAI si está configurado
    system_msg = (
        "Eres un asistente que genera proformas y cotizaciones profesionales y concisas. "
        "Devuelve un texto cordial, claro y estructurado para enviar a un cliente."
    )
    user_msg = (
        f"Genera una proforma breve y profesional para {cliente}. "
        f"El servicio solicitado es '{servicio}', con {horas} horas de trabajo "
        f"y complejidad {complejidad}. El costo estimado es {precio} dólares. "
        "Redacta un texto claro y cordial."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=350,
        )
        if resp and getattr(resp, "choices", None):
            choice0 = resp.choices[0]
            content = None
            if hasattr(choice0, "message") and isinstance(choice0.message, dict):
                content = choice0.message.get("content")
            elif hasattr(choice0, "message") and hasattr(choice0.message, "content"):
                content = getattr(choice0.message, "content")
            elif isinstance(choice0, dict):
                content = choice0.get("message", {}).get("content") or choice0.get("text")
            return (content or "(Respuesta vacía de la IA)").strip()
    except Exception as e:
        logger.exception("Error generando proforma IA con la API moderna de OpenAI: %s", e)
        return f"(Error generando proforma IA: {e})"


def _cleanup_file(path: str) -> None:
    try:
        os.unlink(path)
        logger.info("Archivo temporal eliminado: %s", path)
    except Exception:
        logger.exception("No se pudo eliminar el archivo temporal: %s", path)


@app.post("/cotizar")
async def cotizar(payload: QuoteRequest):
    precio = calcular_precio(payload.horas, payload.complejidad)
    proforma = await generar_proforma_ai(payload.cliente or "Cliente", payload.servicio, payload.horas, payload.complejidad, precio)
    return JSONResponse({"precio": precio, "proforma": proforma})


@app.post("/cotizar_pdf")
async def cotizar_pdf(payload: QuoteRequest, background_tasks: BackgroundTasks):
    precio = calcular_precio(payload.horas, payload.complejidad)
    proforma = await generar_proforma_ai(payload.cliente or "Cliente", payload.servicio, payload.horas, payload.complejidad, precio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    filename = tmp.name
    tmp.close()

    c = canvas.Canvas(filename)
    c.setFont("Helvetica", 12)
    c.drawString(100, 780, f"Cotización para: {payload.cliente}")
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
    ws.append(["Cliente", "Servicio", "Horas", "Complejidad", "Precio"])
    ws.append([payload.cliente, payload.servicio, payload.horas, payload.complejidad, precio])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    filename = tmp.name
    tmp.close()

    wb.save(filename)

    background_tasks.add_task(_cleanup_file, filename)
    return FileResponse(filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="cotizacion.xlsx")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)), reload=True)
