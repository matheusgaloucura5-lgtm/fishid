"""
FishID Backend — FastAPI
=========================
Rota principal: GET /buscar-peixe?nome=...

Fluxo (versão sem foto de identificação — busca só por nome):
1. Usuário digita o nome do peixe (comum ou científico) no app.
2. O backend procura no seu banco (Postgres) por nome parecido
   (ILIKE + fuzzy match via pg_trgm — tolera acento/erro de digitação).
3. Se achar, busca a ficha técnica completa (água, alimentação,
   temperamento, compatibilidade) na view v_species_full.
4. Busca uma FOTO da espécie na API pública do iNaturalist
   (gratuita, sem chave, usa o nome científico).
5. Loga a busca em `identification_log` (pra você acompanhar quais
   nomes as pessoas mais procuram e quais não foram encontrados).
6. Retorna um JSON único com tudo — dados do seu banco + foto.

Instalação:
    pip install fastapi uvicorn asyncpg httpx python-multipart

Variáveis de ambiente esperadas:
    DATABASE_URL   (ex: postgresql://postgres:password@localhost:5432/fishid)

Rodar:
    uvicorn 03_backend_main:app --reload --port 8000
"""

import os
import uuid
from typing import Optional

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://user:password@localhost:5432/fishid"
).replace("postgresql+asyncpg://", "postgresql://").replace("postgres://", "postgresql://", 1)

INATURALIST_TAXA_URL = "https://api.inaturalist.org/v1/taxa"

db_pool: Optional[asyncpg.Pool] = None
http_client = httpx.AsyncClient(timeout=10.0)

app = FastAPI(title="FishID API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)


@app.on_event("shutdown")
async def shutdown():
    if db_pool is not None:
        await db_pool.close()
    await http_client.aclose()


# ------------------------------------------------------------------
# Schemas de resposta
# ------------------------------------------------------------------
class FeedingItem(BaseModel):
    food_type: str
    is_live_food: bool
    frequency: Optional[str] = None
    notes: Optional[str] = None


class CompatibilityItem(BaseModel):
    species_id: str
    common_name: str
    level: str
    notes: Optional[str] = None


class FichaTecnica(BaseModel):
    id: str
    common_name: str
    scientific_name: str
    family: Optional[str]
    temperament_level: Optional[int]
    temperament_notes: Optional[str]
    capture_technique: Optional[str]
    handling_care_notes: Optional[str]
    ph_min: Optional[float]
    ph_max: Optional[float]
    temp_min_celsius: Optional[float]
    temp_max_celsius: Optional[float]
    dh_min: Optional[float]
    dh_max: Optional[float]
    min_tank_liters: Optional[int]
    feeding: list[FeedingItem]
    compatibility: list[CompatibilityItem]
    photo_url: Optional[str] = None
    photo_attribution: Optional[str] = None


# ------------------------------------------------------------------
# Busca no banco por nome digitado (comum OU científico)
# Tenta "contém" (ILIKE) primeiro, cai pra similaridade (trigram) se
# não achar — cobre erro de digitação e acento faltando.
# ------------------------------------------------------------------
async def buscar_especie_por_nome(conn: asyncpg.Connection, termo: str):
    termo = termo.strip()

    row = await conn.fetchrow(
        """
        SELECT id
        FROM species
        WHERE common_name ILIKE '%' || $1 || '%'
           OR scientific_name ILIKE '%' || $1 || '%'
        ORDER BY
            CASE WHEN common_name ILIKE $1 || '%' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        termo,
    )
    if row is not None:
        return row

    row = await conn.fetchrow(
        """
        SELECT id,
               GREATEST(similarity(common_name, $1), similarity(scientific_name, $1)) AS score
        FROM species
        WHERE similarity(common_name, $1) > 0.3 OR similarity(scientific_name, $1) > 0.3
        ORDER BY score DESC
        LIMIT 1
        """,
        termo,
    )
    return row


async def buscar_ficha_completa(conn: asyncpg.Connection, species_id):
    row = await conn.fetchrow("SELECT * FROM v_species_full WHERE id = $1", species_id)
    return row


# ------------------------------------------------------------------
# Foto da espécie via iNaturalist (API pública, sem chave)
# Se a chamada falhar por qualquer motivo (rede fora do ar, espécie
# não catalogada lá, etc.), devolve (None, None) — o app mostra um
# placeholder nesse caso, não quebra a resposta.
# ------------------------------------------------------------------
async def buscar_foto_especie(scientific_name: str) -> tuple[Optional[str], Optional[str]]:
    try:
        resp = await http_client.get(
            INATURALIST_TAXA_URL,
            params={"q": scientific_name, "rank": "species", "per_page": 1},
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results") or []
        if not results:
            return None, None

        photo = results[0].get("default_photo")
        if not photo:
            return None, None

        # medium_url costuma ser ~500px, boa pra exibir no card do app
        photo_url = photo.get("medium_url") or photo.get("square_url")
        attribution = photo.get("attribution")
        return photo_url, attribution

    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None, None


# ------------------------------------------------------------------
# Log de buscas (auditoria simples — quais nomes as pessoas procuram
# e quais não foram encontrados no seu banco)
# ------------------------------------------------------------------
async def registrar_log(conn: asyncpg.Connection, termo_busca: str, species_id: Optional[str]):
    await conn.execute(
        """
        INSERT INTO identification_log
            (id, image_url, resolved_species_id, confidence, ai_engine)
        VALUES ($1, $2, $3, $4, $5)
        """,
        str(uuid.uuid4()),
        f"busca-por-nome:{termo_busca}",
        species_id,
        1.0 if species_id else 0.0,
        "manual-search",
    )


def montar_ficha_tecnica(ficha, photo_url: Optional[str], photo_attribution: Optional[str]) -> FichaTecnica:
    import json as _json
    feeding_raw = _json.loads(ficha["feeding"]) if isinstance(ficha["feeding"], str) else ficha["feeding"]
    compatibility_raw = _json.loads(ficha["compatibility"]) if isinstance(ficha["compatibility"], str) else ficha["compatibility"]

    return FichaTecnica(
        id=str(ficha["id"]),
        common_name=ficha["common_name"],
        scientific_name=ficha["scientific_name"],
        family=ficha["family"],
        temperament_level=ficha["temperament_level"],
        temperament_notes=ficha["temperament_notes"],
        capture_technique=ficha["capture_technique"],
        handling_care_notes=ficha["handling_care_notes"],
        ph_min=float(ficha["ph_min"]) if ficha["ph_min"] is not None else None,
        ph_max=float(ficha["ph_max"]) if ficha["ph_max"] is not None else None,
        temp_min_celsius=float(ficha["temp_min_celsius"]) if ficha["temp_min_celsius"] is not None else None,
        temp_max_celsius=float(ficha["temp_max_celsius"]) if ficha["temp_max_celsius"] is not None else None,
        dh_min=float(ficha["dh_min"]) if ficha["dh_min"] is not None else None,
        dh_max=float(ficha["dh_max"]) if ficha["dh_max"] is not None else None,
        min_tank_liters=ficha["min_tank_liters"],
        feeding=[FeedingItem(**f) for f in feeding_raw],
        compatibility=[CompatibilityItem(**c) for c in compatibility_raw],
        photo_url=photo_url,
        photo_attribution=photo_attribution,
    )


# ------------------------------------------------------------------
# Rota principal: busca por nome
# ------------------------------------------------------------------
@app.get("/buscar-peixe", response_model=FichaTecnica)
async def buscar_peixe_por_nome(nome: str):
    nome = nome.strip()
    if not nome:
        raise HTTPException(400, "Informe um nome para buscar.")

    async with db_pool.acquire() as conn:
        match = await buscar_especie_por_nome(conn, nome)

        if match is None:
            await registrar_log(conn, nome, None)
            raise HTTPException(
                404,
                detail={"mensagem": f'Nenhuma espécie encontrada para "{nome}" no seu banco de dados.'},
            )

        species_id = match["id"]
        ficha = await buscar_ficha_completa(conn, species_id)
        await registrar_log(conn, nome, str(species_id))

    # Busca a foto fora da transação do banco (chamada de rede externa,
    # não precisa seguar a conexão do Postgres esperando)
    photo_url, photo_attribution = await buscar_foto_especie(ficha["scientific_name"])

    return montar_ficha_tecnica(ficha, photo_url, photo_attribution)


@app.get("/health")
async def health():
    return {"status": "ok"}
