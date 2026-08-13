# Análisis de casos (F1@10)

n evaluados: 30 documentos (excluidos: [10, 12, 14])

## Mayor ventaja de BB-IDF sobre TF-IDF

### Doc 20: VALORACIÓN ECONÓMICA DEL COMPLEJO ARQUEOLÓGICO DE KUELAP.pdf
- Gold (autor): arqueológico, complejo, consumidor, costo, económico, excedente, individual, valoración, viaje
- F1@10: TF-IDF=0.526 | BB-IDF=0.737 | TextRank=0.421
- Top-10 TF-IDF: cak, costo, excedente, valoración, económico, valor, kuélap, sociologia, visita, consumidor
- Top-10 BB-IDF: costo, valoración, cak, visita, consumidor, valor, complejo, económico, excedente, viaje
- Top-10 TextRank: económico, valor, valoración, visita, costo, natural, perú, demanda, kuélap, viaje

### Doc 30: redes sociales y turismo.pdf
- Gold (autor): redes, social, turismo, venta
- F1@10: TF-IDF=0.286 | BB-IDF=0.429 | TextRank=0.429
- Top-10 TF-IDF: venta, red, social, operador, turístico, contribución, servicio, contenido, variable, chachapoyas
- Top-10 BB-IDF: venta, red, turístico, social, operador, contenido, chachapoyas, contribución, turismo, servicio
- Top-10 TextRank: social, red, venta, turístico, chachapoyas, operador, turismo, servicio, variable, contribución

### Doc 9: Innovación MYPES en turismo.pdf
- Gold (autor): competitividad, innovación, myp, sector, turismo
- F1@10: TF-IDF=0.400 | BB-IDF=0.533 | TextRank=0.667
- Top-10 TF-IDF: innovación, empresa, myp, figura, chachapoyas, hostal, competitividad, estrellas, cliente, satisfecho
- Top-10 BB-IDF: empresa, innovación, chachapoyas, myp, figura, turismo, competitividad, hostal, totalmente, detallar
- Top-10 TextRank: chachapoyas, empresa, innovación, figura, turismo, sector, hostal, competitividad, myp, organización

### Doc 13: POTENCIAL TURISTÍCO DEL DISTRITO DE CHUQUIBAMBA, PROVINCIA CHACHAPOYAS, DEPARTAMENTO DE LA LIBERT.pdf
- Gold (autor): categorización, inventario, jerarquización, potencial, recurso, turístico
- F1@10: TF-IDF=0.375 | BB-IDF=0.500 | TextRank=0.375
- Top-10 TF-IDF: recurso, chuquibamba, turístico, jerarquización, cochabamba, punto, distrito, jerarquía, pueblo, particularidades
- Top-10 BB-IDF: turístico, recurso, chuquibamba, punto, pueblo, distrito, jerarquización, turismo, jerarquía, potencial
- Top-10 TextRank: recurso, turístico, chuquibamba, distrito, turismo, pueblo, potencial, actividad, tabla, actual

### Doc 24: festividades folkloricas y actividades turisticas.pdf
- Gold (autor): actividad, festividad, folclore, folclórico, turismo, turístico
- F1@10: TF-IDF=0.500 | BB-IDF=0.625 | TextRank=0.625
- Top-10 TF-IDF: folclórico, festividad, turístico, chachapoyas, actividad, visitante, ciudad, desacuerdo, tabla, afirmación
- Top-10 BB-IDF: turístico, festividad, folclórico, chachapoyas, visitante, ciudad, actividad, desacuerdo, totalmente, turismo
- Top-10 TextRank: turístico, chachapoyas, festividad, actividad, folclórico, visitante, cultural, ciudad, turismo, investigación

## Peor desempeño relativo de BB-IDF

### Doc 27: paisajes culturales.pdf
- Gold (autor): actor, cultural, desarrollo, local, paisajes, turista, turístico
- F1@10: TF-IDF=0.471 | BB-IDF=0.353 | TextRank=0.471
- Top-10 TF-IDF: cultural, paisaje, turístico, desarrollo, chachapoyas, ciudad, turismo, paisajes, natural, patrimonio
- Top-10 BB-IDF: cultural, turístico, paisaje, desarrollo, chachapoyas, turismo, ciudad, potencial, natural, patrimonio
- Top-10 TextRank: cultural, paisaje, turístico, desarrollo, chachapoyas, turismo, natural, ciudad, local, patrimonio

### Doc 0: 2011.pdf
- Gold (autor): categoría, chachapuya, civilizado, diluvio, etnia, identidad, incario, movilidad, mítico, oral, purum, simbolismir, tradición, transformación
- F1@10: TF-IDF=0.167 | BB-IDF=0.167 | TextRank=0.167
- Top-10 TF-IDF: purum, grafía, quechua, simbolización, puya, arqueología, laguna, identidad, aymara, hombre
- Top-10 BB-IDF: purum, grafía, quechua, hombre, simbolización, chachapoyas, identidad, arqueología, sociedad, laguna
- Top-10 TextRank: purum, chachapoyas, quechua, grafía, identidad, hombre, caso, sitio, lima, laguna

### Doc 1: Chachapoyas Resort. una experiencia espléndida.pdf
- Gold (autor): cultura, estrategia, experiencia, naturaleza, turismo
- F1@10: TF-IDF=0.133 | BB-IDF=0.133 | TextRank=0.133
- Top-10 TF-IDF: año, costo, caja, tabla, financiero, servicio, precio, depreciación, fuente, turismo
- Top-10 BB-IDF: año, costo, chachapoyas, turismo, elaboración, financiero, caja, hospedaje, tabla, fuente
- Top-10 TextRank: turismo, servicio, año, chachapoyas, tabla, turista, fuente, costo, perú, proyecto

### Doc 2: Cultura turisticay conservación de recurso turistico.pdf
- Gold (autor): conservación, cultura, recurso, turístico
- F1@10: TF-IDF=0.571 | BB-IDF=0.571 | TextRank=0.571
- Top-10 TF-IDF: sholón, turístico, recurso, colcamar, conservación, cultura, nivel, poblador, correlación, distrito
- Top-10 BB-IDF: turístico, sholón, recurso, conservación, colcamar, cultura, nivel, turismo, distrito, poblador
- Top-10 TextRank: turístico, recurso, conservación, sholón, cultura, nivel, colcamar, turismo, cultural, distrito

### Doc 3: DESARROLLO DE LA ACTIVIDAD TURISTICA.pdf
- Gold (autor): actividad, amazona, cultural, economía, regional, turismo, turístico
- F1@10: TF-IDF=0.353 | BB-IDF=0.353 | TextRank=0.353
- Top-10 TF-IDF: turismo, negonotas, amazónico, docentes, turístico, región, tourism, actividad, revista, muñoz
- Top-10 BB-IDF: turismo, turístico, amazónico, revista, región, docentes, negonotas, muñoz, tourism, actividad
- Top-10 TextRank: turismo, turístico, tourism, región, actividad, revista, desarrollo, amazónico, turista, estudio

## BB-IDF más cercano a TextRank

### Doc 0: 2011.pdf
- Gold (autor): categoría, chachapuya, civilizado, diluvio, etnia, identidad, incario, movilidad, mítico, oral, purum, simbolismir, tradición, transformación
- F1@10: TF-IDF=0.167 | BB-IDF=0.167 | TextRank=0.167
- Top-10 TF-IDF: purum, grafía, quechua, simbolización, puya, arqueología, laguna, identidad, aymara, hombre
- Top-10 BB-IDF: purum, grafía, quechua, hombre, simbolización, chachapoyas, identidad, arqueología, sociedad, laguna
- Top-10 TextRank: purum, chachapoyas, quechua, grafía, identidad, hombre, caso, sitio, lima, laguna

### Doc 1: Chachapoyas Resort. una experiencia espléndida.pdf
- Gold (autor): cultura, estrategia, experiencia, naturaleza, turismo
- F1@10: TF-IDF=0.133 | BB-IDF=0.133 | TextRank=0.133
- Top-10 TF-IDF: año, costo, caja, tabla, financiero, servicio, precio, depreciación, fuente, turismo
- Top-10 BB-IDF: año, costo, chachapoyas, turismo, elaboración, financiero, caja, hospedaje, tabla, fuente
- Top-10 TextRank: turismo, servicio, año, chachapoyas, tabla, turista, fuente, costo, perú, proyecto

### Doc 2: Cultura turisticay conservación de recurso turistico.pdf
- Gold (autor): conservación, cultura, recurso, turístico
- F1@10: TF-IDF=0.571 | BB-IDF=0.571 | TextRank=0.571
- Top-10 TF-IDF: sholón, turístico, recurso, colcamar, conservación, cultura, nivel, poblador, correlación, distrito
- Top-10 BB-IDF: turístico, sholón, recurso, conservación, colcamar, cultura, nivel, turismo, distrito, poblador
- Top-10 TextRank: turístico, recurso, conservación, sholón, cultura, nivel, colcamar, turismo, cultural, distrito

### Doc 3: DESARROLLO DE LA ACTIVIDAD TURISTICA.pdf
- Gold (autor): actividad, amazona, cultural, economía, regional, turismo, turístico
- F1@10: TF-IDF=0.353 | BB-IDF=0.353 | TextRank=0.353
- Top-10 TF-IDF: turismo, negonotas, amazónico, docentes, turístico, región, tourism, actividad, revista, muñoz
- Top-10 BB-IDF: turismo, turístico, amazónico, revista, región, docentes, negonotas, muñoz, tourism, actividad
- Top-10 TextRank: turismo, turístico, tourism, región, actividad, revista, desarrollo, amazónico, turista, estudio

### Doc 5: Dialnet-GestionPublicaDelTurismoEnLaSatisfaccionDeLosTuris-9864976.pdf
- Gold (autor): arqueológico, calidad, complejo, gestión, kuelap, mercadeo, público, satisfacción, servicio, turismo, turista
- F1@10: TF-IDF=0.762 | BB-IDF=0.762 | TextRank=0.762
- Top-10 TF-IDF: satisfacción, turismo, gestión, kuelap, turista, público, arqueológico, complejo, pseudo, turístico
- Top-10 BB-IDF: turismo, satisfacción, kuelap, gestión, turista, turístico, público, complejo, arqueológico, visitante
- Top-10 TextRank: turismo, gestión, satisfacción, turista, público, kuelap, turístico, arqueológico, visitante, complejo

## BB-IDF supera a TextRank

### Doc 4: DISEÑO DE UN ECOLODGE VIVENCIAL-CHACHAPOYAS.pdf
- Gold (autor): ecolodge, material, tradicional, turismo, vernácular
- F1@10: TF-IDF=0.267 | BB-IDF=0.267 | TextRank=0.133
- Top-10 TF-IDF: ecolodge, vernácular, arquitectura, vivencial, diseño, tabla, proyecto, caso, chachapoyas, criterio
- Top-10 BB-IDF: ecolodge, vivencial, vernácular, diseño, arquitectura, proyecto, chachapoyas, criterio, tabla, caso
- Top-10 TextRank: diseño, proyecto, ecolodge, natural, chachapoyas, tabla, caso, arquitectura, análisis, zona

### Doc 6: EVALUACIÓN DE LA CULTURA TURÍSTICA Y SU INFLUENC.pdf
- Gold (autor): actitud, comportamiento, conocimiento, cultura, sostenible, turismo, turístico
- F1@10: TF-IDF=0.588 | BB-IDF=0.588 | TextRank=0.471
- Top-10 TF-IDF: turístico, turismo, cultura, desarrollo, población, sostenible, recuperado, actitud, porcentaje, ciudad
- Top-10 BB-IDF: turístico, turismo, desarrollo, chachapoyas, actitud, cultura, sostenible, porcentaje, ciudad, población
- Top-10 TextRank: turístico, turismo, población, desarrollo, cultura, chachapoyas, sostenible, ciudad, actividad, comunidad

### Doc 8: Gestion_Municipal_y_Desarrollo_Turistico_de_la_ciu.pdf
- Gold (autor): diseño, municipal, organizacional, planificación, políticas, públicas
- F1@10: TF-IDF=0.375 | BB-IDF=0.375 | TextRank=0.250
- Top-10 TF-IDF: municipal, turístico, desarrollo, gestión, correlación, organizacional, dimensión, planificación, variable, bilateral
- Top-10 BB-IDF: turístico, municipal, desarrollo, gestión, planificación, dimensión, correlación, turismo, organizacional, chachapoyas
- Top-10 TextRank: turístico, desarrollo, municipal, gestión, turismo, variable, público, dimensión, correlación, planificación

### Doc 13: POTENCIAL TURISTÍCO DEL DISTRITO DE CHUQUIBAMBA, PROVINCIA CHACHAPOYAS, DEPARTAMENTO DE LA LIBERT.pdf
- Gold (autor): categorización, inventario, jerarquización, potencial, recurso, turístico
- F1@10: TF-IDF=0.375 | BB-IDF=0.500 | TextRank=0.375
- Top-10 TF-IDF: recurso, chuquibamba, turístico, jerarquización, cochabamba, punto, distrito, jerarquía, pueblo, particularidades
- Top-10 BB-IDF: turístico, recurso, chuquibamba, punto, pueblo, distrito, jerarquización, turismo, jerarquía, potencial
- Top-10 TextRank: recurso, turístico, chuquibamba, distrito, turismo, pueblo, potencial, actividad, tabla, actual

### Doc 17: TURISMO Y LAS CONDICIONES SOCIOECONÓMICAS.pdf
- Gold (autor): acceso, básico, ingreso, pobreza, relación, servicio, turismo
- F1@10: TF-IDF=0.235 | BB-IDF=0.235 | TextRank=0.118
- Top-10 TF-IDF: hogar, porcentaje, jefe, turismo, variable, figura, turístico, kuelap, pobreza, visita
- Top-10 BB-IDF: hogar, turístico, turismo, porcentaje, jefe, kuelap, efecto, pobreza, variable, visita
- Top-10 TextRank: turismo, hogar, turístico, variable, porcentaje, figura, nivel, resultado, turista, chachapoyas

### Doc 19: Turismo sostenioble.pdf
- Gold (autor): comunidad, desarrollo, infraestructura, nativo, sostenible, turismo
- F1@10: TF-IDF=0.625 | BB-IDF=0.750 | TextRank=0.625
- Top-10 TF-IDF: amazonía, sostenible, desarrollo, prohominum, turismo, aeroportuario, comunidad, infraestructura, humanas, perpetuar
- Top-10 BB-IDF: sostenible, amazonía, desarrollo, turismo, infraestructura, comunidad, peruano, artículo, nativo, biodiversidad
- Top-10 TextRank: desarrollo, sostenible, turismo, comunidad, amazonía, infraestructura, económico, local, social, peruano

### Doc 20: VALORACIÓN ECONÓMICA DEL COMPLEJO ARQUEOLÓGICO DE KUELAP.pdf
- Gold (autor): arqueológico, complejo, consumidor, costo, económico, excedente, individual, valoración, viaje
- F1@10: TF-IDF=0.526 | BB-IDF=0.737 | TextRank=0.421
- Top-10 TF-IDF: cak, costo, excedente, valoración, económico, valor, kuélap, sociologia, visita, consumidor
- Top-10 BB-IDF: costo, valoración, cak, visita, consumidor, valor, complejo, económico, excedente, viaje
- Top-10 TextRank: económico, valor, valoración, visita, costo, natural, perú, demanda, kuélap, viaje
