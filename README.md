# backendKiros

Backend minimal para generar cotizaciones con IA y documentos (PDF/Excel).

Rutas importantes:
- `POST /cotizar` -> devuelve JSON con `precio` y `proforma`.
- `POST /cotizar_pdf` -> devuelve PDF descargable.
- `POST /cotizar_excel` -> devuelve Excel descargable.

Instalación rápida (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# editar .env y poner OPENAI_API_KEY
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ejemplo de petición JSON (POST):

```json
{
  "servicio": "Desarrollo Web",
  "horas": 10,
  "complejidad": "alta",
  "cliente": "ACME S.A."
}
```

Notas:
- Coloca tu clave de OpenAI en `.env` como `OPENAI_API_KEY`.
- Los archivos PDF/XLSX generados se crean en directorio temporal y se devuelven al cliente.

Chat completions
----------------
- Puedes usar la API de Chat (recomendada) configurando `OPENAI_API_KEY` en `.env`.
- Opcionalmente, define `OPENAI_MODEL` en `.env` (p. ej. `gpt-3.5-turbo` o `gpt-4o-mini`).
 - Si no se configura `OPENAI_API_KEY`, el backend devuelve un texto de proforma por defecto.

Modo sin IA (gratis)
--------------------
- Si no quieres usar la API de OpenAI (evitar costes), añade en `.env`: `USE_OPENAI=false`.
- En ese modo el backend generará una proforma local y no hará llamadas externas.
