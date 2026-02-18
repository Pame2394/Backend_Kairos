import os
import time
import httpx

BASE = "http://127.0.0.1:8000"
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

payload = {
    "servicio": "Prueba Automatizada",
    "horas": 3,
    "complejidad": "media",
    "cliente": "Cliente Test"
}


def test_cotizar():
    print("-> Probando /cotizar ...")
    with httpx.Client(timeout=10.0) as client:
        r = client.post(f"{BASE}/cotizar", json=payload)
    print("status:", r.status_code)
    print("json:", r.json())
    assert r.status_code == 200


def test_cotizar_pdf():
    print("-> Probando /cotizar_pdf ...")
    fname = os.path.join(OUT_DIR, "cotizacion_test.pdf")
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{BASE}/cotizar_pdf", json=payload)
    print("status:", r.status_code)
    assert r.status_code == 200
    with open(fname, "wb") as f:
        f.write(r.content)
    print("guardado:", fname, "size=", os.path.getsize(fname))
    # quick magic bytes check for PDF
    with open(fname, "rb") as f:
        head = f.read(4)
    assert head == b"%PDF"
    return fname


def test_cotizar_excel():
    print("-> Probando /cotizar_excel ...")
    fname = os.path.join(OUT_DIR, "cotizacion_test.xlsx")
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{BASE}/cotizar_excel", json=payload)
    print("status:", r.status_code)
    assert r.status_code == 200
    with open(fname, "wb") as f:
        f.write(r.content)
    print("guardado:", fname, "size=", os.path.getsize(fname))
    # xlsx files are zip archives starting with PK
    with open(fname, "rb") as f:
        head = f.read(2)
    assert head == b"PK"
    return fname


if __name__ == "__main__":
    try:
        test_cotizar()
    except Exception as e:
        print("/cotizar fallo:", e)
    try:
        pdf_file = test_cotizar_pdf()
    except Exception as e:
        print("/cotizar_pdf fallo:", e)
        pdf_file = None
    try:
        xlsx_file = test_cotizar_excel()
    except Exception as e:
        print("/cotizar_excel fallo:", e)
        xlsx_file = None

    # wait a moment for background cleanup to run, then check files
    print("Esperando 2s para que el servidor pueda limpiar archivos temporales...")
    time.sleep(2)
    for f in (pdf_file, xlsx_file):
        if f is None:
            continue
        exists = os.path.exists(f)
        print(f"Archivo {os.path.basename(f)} existe después de 2s? {exists}")

    print("Pruebas finalizadas.")
