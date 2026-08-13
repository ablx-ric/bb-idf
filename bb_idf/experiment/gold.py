"""Gold standard: author-declared keywords for the 33-document corpus.

Each entry corresponds to the deterministic load order of
``sorted(glob("data/corpus/*.pdf"))`` (index 0..32). Keywords were manually
transcribed from the "Palabras clave(s)" / "Keywords" sections of each PDF.

Language policy (documented):
  * The gold standard uses the language of the document's MAIN text so that
    matching against the (Spanish-lemmatized) document tokens is meaningful.
  * index 16 ("Public Management of Tourism...") is an English article with
    English keywords -> keywords kept in English.
  * index 20 ("VALORACION ECONOMICA ... KUELAP") has a Spanish body but the
    authors declared keywords ONLY in English -> translated to Spanish
    (faithful translation of the declared concepts, flagged in ``notes``).
  * index 10 ("La (re)construccion de Chachapoyas") and index 14
    ("Planeamiento Estrategico...") do NOT declare keywords -> excluded from
    gold-standard evaluation (``excluded``).

The raw phrases are normalized with the same spaCy pipeline as the documents
(lemmatization + stopwords + like_num + len>=3) and then flattened to the set
of content unigrams, so they live in the SAME term space as the algorithms.
"""

from __future__ import annotations

# Ordered by sorted(glob("data/corpus/*.pdf"))
GOLD = [
    # 0  2011.pdf
    {
        "file": "2011.pdf",
        "keywords": [
            "Chachapuya", "Purum", "incario", "etnia", "identidad",
            "categorías", "no civilizados", "transformación", "movilidad",
            "diluvio", "simbolismo", "mítico", "tradición oral",
        ],
        "language": "es",
    },
    # 1  Chachapoyas Resort. una experiencia espléndida.pdf
    {
        "file": "Chachapoyas Resort. una experiencia espléndida.pdf",
        "keywords": ["turismo", "experiencias", "cultura", "naturaleza", "estrategia"],
        "language": "es",
    },
    # 2  Cultura turisticay conservación de recurso turistico.pdf
    {
        "file": "Cultura turisticay conservación de recurso turistico.pdf",
        "keywords": ["Cultura turística", "conservación", "recurso turístico"],
        "language": "es",
    },
    # 3  DESARROLLO DE LA ACTIVIDAD TURISTICA.pdf
    {
        "file": "DESARROLLO DE LA ACTIVIDAD TURISTICA.pdf",
        "keywords": ["actividad turística", "amazonas", "economía regional", "turismo cultural"],
        "language": "es",
    },
    # 4  DISEÑO DE UN ECOLODGE VIVENCIAL-CHACHAPOYAS.pdf (keywords on p.13)
    {
        "file": "DISEÑO DE UN ECOLODGE VIVENCIAL-CHACHAPOYAS.pdf",
        "keywords": ["Ecolodge", "Turismo", "materiales tradicionales", "vernácula"],
        "language": "es",
    },
    # 5  Dialnet-GestionPublicaDelTurismo...pdf (ES)
    {
        "file": "Dialnet-GestionPublicaDelTurismoEnLaSatisfaccionDeLosTuris-9864976.pdf",
        "keywords": [
            "gestión pública del turismo", "satisfacción de los turistas",
            "Complejo Arqueológico de Kuelap", "calidad de los servicios", "mercadeo",
        ],
        "language": "es",
    },
    # 6  EVALUACIÓN DE LA CULTURA TURÍSTICA Y SU INFLUENC.pdf
    {
        "file": "EVALUACIÓN DE LA CULTURA TURÍSTICA Y SU INFLUENC.pdf",
        "keywords": [
            "cultura turística", "comportamiento turístico", "actitud turística",
            "conocimiento turístico", "turismo sostenible",
        ],
        "language": "es",
    },
    # 7  Eevaluaci_on de los recursos turisticos.magdalena.pdf
    {
        "file": "Eevaluaci_on de los recursos turisticos.magdalena.pdf",
        "keywords": [
            "recursos turísticos", "recursos arqueológicos", "cultura viva",
            "cultura Chachapoyas",
        ],
        "language": "es",
    },
    # 8  Gestion_Municipal_y_Desarrollo_Turistico_de_la_ciu.pdf
    {
        "file": "Gestion_Municipal_y_Desarrollo_Turistico_de_la_ciu.pdf",
        "keywords": ["Diseño Organizacional", "Planificación Municipal", "Políticas Públicas"],
        "language": "es",
    },
    # 9  Innovación MYPES en turismo.pdf
    {
        "file": "Innovación MYPES en turismo.pdf",
        "keywords": ["innovación", "competitividad", "mypes", "sector turismo"],
        "language": "es",
    },
    # 10 La (re)construcción de Chachapoyas.pdf  -> NO keywords declared
    {
        "file": "La (re)construcción de Chachapoyas.pdf",
        "keywords": [],
        "language": "es",
    },
    # 11 Orosco Tuesta Lesly Fiorela.pdf
    {
        "file": "Orosco Tuesta Lesly Fiorela.pdf",
        "keywords": ["Turismo", "Oferta", "Satisfacción del Turista"],
        "language": "es",
    },
    # 12 PERCEPCION DEL DESTINO CHACHAPOYAS.pdf -> NO keywords declared
    #    (the "palabras clave" mentions are methodological, not author keywords)
    {
        "file": "PERCEPCION DEL DESTINO CHACHAPOYAS.pdf",
        "keywords": [],
        "language": "es",
    },
    # 13 POTENCIAL TURISTÍCO DEL DISTRITO DE CHUQUIBAMBA...
    {
        "file": "POTENCIAL TURISTÍCO DEL DISTRITO DE CHUQUIBAMBA, PROVINCIA CHACHAPOYAS, DEPARTAMENTO DE LA LIBERT.pdf",
        "keywords": [
            "Potencial turístico", "recursos turísticos", "inventario turístico",
            "categorización", "jerarquización",
        ],
        "language": "es",
    },
    # 14 Planeamiento Estratégico... -> NO keywords declared
    {
        "file": "Planeamiento Estratégico de la Provincia de Chachapoyas - Amazonas.pdf",
        "keywords": [],
        "language": "es",
    },
    # 15 Potencial turistico colcamar.pdf
    {
        "file": "Potencial turistico colcamar.pdf",
        "keywords": ["sitios arqueológicos", "paisajes naturales", "comunidades", "turismo", "Amazonas"],
        "language": "es",
    },
    # 16 Public Management of Tourism in Tourist Satisfaction.pdf (EN article)
    {
        "file": "Public Management of Tourism in Tourist Satisfaction.pdf",
        "keywords": [
            "public management of tourism", "tourist satisfaction",
            "Kuelap Archaeological Complex", "quality of services", "marketing",
        ],
        "language": "en",
    },
    # 17 TURISMO Y LAS CONDICIONES SOCIOECONÓMICAS.pdf
    {
        "file": "TURISMO Y LAS CONDICIONES SOCIOECONÓMICAS.pdf",
        "keywords": ["acceso a servicios básicos", "ingreso", "pobreza", "relación", "turismo"],
        "language": "es",
    },
    # 18 Turismo rural comunitario en kuelap.pdf
    {
        "file": "Turismo rural comunitario en kuelap.pdf",
        "keywords": ["Turismo Rural Comunitario", "Desarrollo Sostenible"],
        "language": "es",
    },
    # 19 Turismo sostenioble.pdf
    {
        "file": "Turismo sostenioble.pdf",
        "keywords": [
            "Turismo sostenible", "infraestructura", "desarrollo sostenible",
            "comunidades nativas",
        ],
        "language": "es",
    },
    # 20 VALORACIÓN ECONÓMICA... KUELAP.pdf (ES body, EN keywords -> translated)
    {
        "file": "VALORACIÓN ECONÓMICA DEL COMPLEJO ARQUEOLÓGICO DE KUELAP.pdf",
        "keywords": [
            "valoración económica", "complejo arqueológico",
            "costo de viaje individual", "excedente del consumidor",
        ],
        "language": "es",
        "notes": "Keywords declared in English; translated to Spanish (document body is Spanish).",
    },
    # 21 análisis del turismo sostenible... Utcubamba.pdf
    {
        "file": "análisis del turismo sostenible en la provincia de Utcubamba, Amazonas-Perú.pdf",
        "keywords": ["turismo alternativo", "sostenible", "aventura", "rural", "vivencial"],
        "language": "es",
    },
    # 22 capacidad de carga turística.pdf
    {
        "file": "capacidad de carga turística.pdf",
        "keywords": ["Capacidad de carga", "turismo", "conservación"],
        "language": "es",
    },
    # 23 estategias de marketin. desarrollo turistico.pdf
    {
        "file": "estategias de marketin. desarrollo turistico.pdf",
        "keywords": [
            "Estrategias de Marketing", "Desarrollo Turístico", "Estudio Correlacional",
            "Agencias de viaje", "Distrito de Levanto",
        ],
        "language": "es",
    },
    # 24 festividades folkloricas y actividades turisticas.pdf
    {
        "file": "festividades folkloricas y actividades turisticas.pdf",
        "keywords": ["Turismo", "festividades folclóricas", "actividad turística", "folclore"],
        "language": "es",
    },
    # 25 guias de turismo en chachapoyas.pdf
    {
        "file": "guias de turismo en chachapoyas.pdf",
        "keywords": ["satisfacción del turista", "calidad del servicio", "guías de turismo"],
        "language": "es",
    },
    # 26 identidad cultural y artesania.pdf
    {
        "file": "identidad cultural y artesania.pdf",
        "keywords": ["identidad cultural", "artesanía", "textil"],
        "language": "es",
    },
    # 27 paisajes culturales.pdf
    {
        "file": "paisajes culturales.pdf",
        "keywords": [
            "Paisajes culturales", "desarrollo turístico", "desarrollo local",
            "turistas", "actores turísticos",
        ],
        "language": "es",
    },
    # 28 percepción del impacto.pdf
    {
        "file": "percepción del impacto.pdf",
        "keywords": [
            "Conservación", "Desarrollo sostenible", "Impacto ambiental",
            "Percepción", "Turismo sostenible",
        ],
        "language": "es",
    },
    # 29 pilasres del turismo.pdf
    {
        "file": "pilasres del turismo.pdf",
        "keywords": ["Administración pública", "pilares del turismo", "AHP"],
        "language": "es",
    },
    # 30 redes sociales y turismo.pdf
    {
        "file": "redes sociales y turismo.pdf",
        "keywords": ["Redes sociales", "ventas", "turismo"],
        "language": "es",
    },
    # 31 turismo alternativo.folklore.pdf
    {
        "file": "turismo alternativo.folklore.pdf",
        "keywords": [
            "Turismo alternativo", "folclore funerario", "estrategia",
            "diversificación", "oferta turística",
        ],
        "language": "es",
    },
    # 32 turismo comunitario.san bartolo.pdf
    {
        "file": "turismo comunitario.san bartolo.pdf",
        "keywords": ["Turismo comunitario", "Desarrollo sostenible"],
        "language": "es",
    },
]

# Documents excluded from gold-standard evaluation (no author keywords),
# keyed by filename (robust to load-order changes).
EXCLUDED_FILES = {
    "La (re)construcción de Chachapoyas.pdf",
    "Planeamiento Estratégico de la Provincia de Chachapoyas - Amazonas.pdf",
}

def gold_by_file() -> dict[str, dict]:
    """Map filename -> gold entry (keywords, language, notes)."""
    return {entry["file"]: entry for entry in GOLD}
