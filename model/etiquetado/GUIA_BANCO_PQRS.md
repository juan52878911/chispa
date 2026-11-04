# Guía de etiquetado del banco de PQRS

The bench has 450 PQRS. They come from four public entities (datos.gov.co), already anonymized: names, ID numbers, phones, emails, addresses and runs of 7 or more digits appear as `[NOMBRE]`, `[DOC]`, `[TEL]`, `[CORREO]`, `[DIR]`, `[PLACA]` or `[NUM]`. They are served from kindling and never leave it. Do not copy, photograph or paste the texts anywhere.

## How to work

- **Blind.** Each person labels on their own. Don't discuss cases with the other person, and don't look for the original label or the model's answer: you can't see either one.
- **Fields.** Each PQRS has three fields: `tipo`, `area` and `plazo`. Answer all three.
- **Keyboard.**
  - Keys `1`–`9` choose an option in the active field, marked ▶.
  - `Tab` moves to the next field.
  - `←` and `→` move between PQRS.
  - Answers save on their own.
- **Two marks:**
  - `?` marks a case as **dudosa**. When in doubt, pick the most likely option anyway and mark it.
  - `p` marks **dato personal visible**: a name, document or address the anonymization missed. Mark it and carry on labeling.
- **Pace.** About 30 s per PQRS. With 450 PQRS that is about 4 h per person, split across several sittings.

## Fields

**tipo**: what the person wants. Judge by the content, not by the word they use: someone who writes «queja» but only asks for a document is making a petición.

| Option | When |
|---|---|
| peticion | Asks for information, a document, a procedure or an action. |
| queja | Complains about the conduct or the treatment of an official or of the service. |
| reclamo | Demands that a fault, a charge or a breached right be corrected or compensated. |
| sugerencia | Proposes an improvement, without asking for anything for themselves. |
| denuncia | Reports an irregularity, a crime or corruption. |
| felicitacion | Thanks or praises. |
| no_es_pqrs | Spam, a test, text with no meaning, or an internal procedure. |

If there are several, choose the one that sets the response. A complaint that ends with a concrete request for a refund is a reclamo.

**area**: the topical mailbox that should handle it, regardless of the entity: salud, vivienda_obras, hacienda_pagos, tramites_documentos, convivencia_seguridad, atencion_funcionarios, talento_humano, otra. If two fit, choose the one that has to *act*.

**plazo**: the legal response deadline under Ley 1755 de 2015.

| Option | When |
|---|---|
| prioritario | Risk to life, health or a fundamental right (art. 20): medicines, urgent surgery, a minor at risk. It takes precedence over the rest. |
| 10_dias | Asks for documents or copies of information (art. 14.1). |
| 30_dias | A consultation: asks the entity's opinion on a matter in its charge (art. 14.2). |
| 15_dias | Everything else that asks for a response: a general petición, queja, reclamo or denuncia (art. 14). |
| sin_plazo | Felicitaciones, sugerencias that ask for nothing, and no_es_pqrs. |

## Bench quality

- **Agreement.** Cohen's κ between the two people, for each field, must be ≥ 0.7.
- **Disagreements.** These are adjudicated afterwards. The cases marked `p` are reviewed and masked before any measurement.
