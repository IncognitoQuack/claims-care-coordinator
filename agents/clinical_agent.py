"""
Clinical Agent — EHR Data Extraction & Analysis

This agent:
1. Receives raw EHR data (structured or unstructured)
2. Uses LLM to extract clinical indicators, severity, and medical necessity
3. Writes each finding to the Medical Necessity Ledger
4. Produces a structured clinical summary for the Policy Agent

The key insight: this agent doesn't just extract data — it *interprets* it
through the lens of medical necessity for insurance authorization.
"""

import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from models.schemas import (
    AgentSource,
    EHRData,
    Severity,
    DiagnosisCode,
    LabResult,
)
from memory.ledger import MedicalNecessityLedger


CLINICAL_SYSTEM_PROMPT = """You are a Clinical Analysis Agent specializing in extracting 
medical necessity indicators from Electronic Health Records (EHR) for insurance prior 
authorization purposes.

Your job is to:
1. Identify the key clinical findings that support medical necessity
2. Flag critical lab values and their clinical significance
3. Map symptoms to diagnosis codes
4. Assess the urgency and medical necessity of the requested procedure
5. Identify any conditions that might qualify for insurance exception clauses

STRICT REQUIREMENT: You must respond in valid JSON format only. No markdown, no explanation outside JSON.
Ensure every field is present and all strings are properly escaped.
CRITICAL: Double-check for missing commas between list items and object properties, especially in long responses.
"""


class ClinicalAgent:
    def __init__(self, client: AsyncOpenAI, ledger: MedicalNecessityLedger):
        self.client = client
        self.ledger = ledger
        self.model = "nvidia/nemotron-3-super-120b-a12b:free"

    async def analyze_ehr(self, ehr: EHRData) -> dict:
        """
        Main entry point: analyze EHR data and write findings to the ledger.
        Returns the complete clinical analysis.
        """
        await self.ledger.write(
            source=AgentSource.CLINICAL,
            event_type="SCAN_START",
            message=f"Initiating EHR scan for {ehr.patient_name} ({ehr.patient_age}{ehr.patient_sex})",
            data={
                "patient": ehr.patient_name,
                "chief_complaint": ehr.chief_complaint,
            },
            tags=["EHR_SCAN", "INIT"],
        )

        symptom_analysis = await self._analyze_symptoms(ehr)
        lab_analysis = await self._analyze_labs(ehr)
        necessity_assessment = await self._assess_medical_necessity(
            ehr,
            symptom_analysis=symptom_analysis,
            lab_analysis=lab_analysis,
        )

        diagnosis_codes = []
        for dc in getattr(ehr, "diagnosis_codes", []) or []:
            try:
                diagnosis_codes.append(dc.model_dump())
            except Exception:
                diagnosis_codes.append(getattr(dc, "__dict__", str(dc)))

        critical_labs = []
        for lab in getattr(ehr, "labs", []) or []:
            try:
                flag = getattr(lab, "flag", None)
                if flag in (Severity.HIGH, Severity.CRITICAL):
                    critical_labs.append(lab.model_dump())
            except Exception:
                continue

        requested_procedure = {}
        if getattr(ehr, "requested_procedure", None):
            try:
                requested_procedure = ehr.requested_procedure.model_dump()
            except Exception:
                requested_procedure = getattr(ehr.requested_procedure, "__dict__", {})

        primary_dx = "Pending"
        if getattr(ehr, "diagnosis_codes", None):
            try:
                primary_dx = ehr.diagnosis_codes[0].description
            except Exception:
                primary_dx = "Pending"

        policy_hints = necessity_assessment.get("policy_search_hints", [])
        recommended_hint = policy_hints[0] if policy_hints else "standard review"

        await self.ledger.write(
            source=AgentSource.LEDGER,
            event_type="CLINICAL_CONTEXT_COMPLETE",
            message=(
                f"Clinical context ready for policy search. "
                f"Primary Dx: {primary_dx}. "
                f"Necessity level: {necessity_assessment.get('necessity_level', 'UNKNOWN')}. "
                f"Recommended search parameters: {recommended_hint}."
            ),
            data={
                "diagnosis_codes": diagnosis_codes,
                "critical_labs": critical_labs,
                "requested_procedure": requested_procedure,
                "necessity_level": necessity_assessment.get("necessity_level", "STANDARD"),
                "policy_search_hints": policy_hints,
                "exception_indicators": necessity_assessment.get("exception_indicators", []),
                "acuity": necessity_assessment.get("acuity", "routine"),
                "symptom_analysis": symptom_analysis,
                "lab_analysis": lab_analysis,
            },
            tags=["LEDGER_WRITE", "CONTEXT_SHARED", "CLINICAL_COMPLETE"],
            severity=Severity.CRITICAL,
        )

        return {
            "symptom_analysis": symptom_analysis,
            "lab_analysis": lab_analysis,
            "necessity_assessment": necessity_assessment,
        }

    def _safe_json_parse(self, text: str) -> dict:
        """
        Robustly extract and parse JSON from model output.

        Handles:
        - markdown code fences
        - extra prose before/after JSON
        - trailing commas
        - Python literals (True/False/None)
        - common delimiter-related issues in long lists
        """
        if not text or not str(text).strip():
            return {"error": "Empty response from model"}

        cleaned = str(text).strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        repaired = cleaned

        repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")

        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)

        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

        repaired = re.sub(r'"\s*,\s*([}\]])', r'"\1', repaired)

        repaired = re.sub(r'(\})(\s*)(")', r'\1,\2\3', repaired)
        repaired = re.sub(r'(\])(\s*)(")', r'\1,\2\3', repaired)
        repaired = re.sub(r'(")(\s*)(\{)', r'\1,\2\3', repaired)
        repaired = re.sub(r'(")(\s*)(\[)', r'\1,\2\3', repaired)

        repaired = re.sub(r'(\d)(\s*)(")', r'\1,\2\3', repaired)
        repaired = re.sub(r'(")(\s*)(\d)', r'\1,\2\3', repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse JSON from model output",
                "parse_error": str(e),
                "raw_text": text,
            }

    def _safe_model_dump(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                pass
        if isinstance(value, (dict, list, str, int, float, bool)):
            return value
        if hasattr(value, "__dict__"):
            return {
                k: v for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
        return str(value)

    def _severity_from_acuity(self, acuity: str) -> Severity:
        acuity = str(acuity or "").lower()
        if acuity == "emergent":
            return Severity.CRITICAL
        if acuity == "urgent":
            return Severity.HIGH
        return Severity.NORMAL

    async def _call_json_model(
        self,
        prompt: str,
        fallback: dict,
        max_tokens: int = 1500,
    ) -> dict:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
        except Exception as e:
            result = dict(fallback)
            result["error"] = f"Model call failed: {e}"
            return result

        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            result = dict(fallback)
            result["error"] = "Malformed model response"
            return result

        parsed = self._safe_json_parse(content)
        if not isinstance(parsed, dict):
            result = dict(fallback)
            result["error"] = "Parsed output was not a JSON object"
            result["raw"] = content
            return result

        if parsed.get("error"):
            result = dict(fallback)
            result.update(parsed)
            return result

        return parsed

    async def _analyze_symptoms(self, ehr: EHRData) -> dict:
        """Use LLM to analyze symptoms in clinical context."""
        diagnosis_codes = [
            self._safe_model_dump(dc) for dc in (getattr(ehr, "diagnosis_codes", []) or [])
        ]

        prompt = f"""Analyze these clinical findings for insurance prior authorization:

Patient: {ehr.patient_name}, {ehr.patient_age}{ehr.patient_sex}
Chief Complaint: {ehr.chief_complaint}
Symptoms: {json.dumps(getattr(ehr, "symptoms", []) or [], ensure_ascii=False)}
Diagnosis Codes: {json.dumps(diagnosis_codes, ensure_ascii=False)}
Clinical Notes: {ehr.clinical_notes or 'None provided'}

Respond with JSON:
{{
    "symptom_clusters": [
        {{
            "cluster_name": "string - clinical grouping name",
            "symptoms": ["list of symptoms in this cluster"],
            "clinical_significance": "string - why this cluster matters for auth",
            "supporting_dx_codes": ["ICD-10 codes that align"]
        }}
    ],
    "primary_condition": "string - the main condition being treated",
    "condition_category": "string - e.g., autoimmune, cardiac, oncologic, neurologic",
    "acuity": "routine | urgent | emergent",
    "classification_criteria_met": ["list of specific diagnostic criteria met, e.g., ACR criteria for lupus"]
}}"""

        fallback = {
            "symptom_clusters": [],
            "primary_condition": "Unknown",
            "condition_category": "Unclassified",
            "acuity": "routine",
            "classification_criteria_met": [],
        }

        result = await self._call_json_model(prompt, fallback)

        result.setdefault("symptom_clusters", [])
        result.setdefault("primary_condition", "Unknown")
        result.setdefault("condition_category", "Unclassified")
        result.setdefault("acuity", "routine")
        result.setdefault("classification_criteria_met", [])

        if not isinstance(result["symptom_clusters"], list):
            result["symptom_clusters"] = []
        if not isinstance(result["classification_criteria_met"], list):
            result["classification_criteria_met"] = []

        await self.ledger.write(
            source=AgentSource.CLINICAL,
            event_type="SYMPTOM_ANALYSIS",
            message=(
                f"Identified {len(result.get('symptom_clusters', []))} symptom cluster(s). "
                f"Primary condition: {result.get('primary_condition', 'Unknown')}. "
                f"Category: {result.get('condition_category', 'Unknown')}. "
                f"Acuity: {result.get('acuity', 'routine')}."
            ),
            data=result,
            tags=[
                "SYMPTOMS",
                str(result.get("condition_category", "Unknown")).upper(),
                str(result.get("acuity", "routine")).upper(),
            ],
            severity=self._severity_from_acuity(result.get("acuity", "routine")),
        )

        return result

    async def _analyze_labs(self, ehr: EHRData) -> dict:
        """Analyze lab results for clinical significance."""
        labs = getattr(ehr, "labs", []) or []
        if not labs:
            result = {
                "critical_findings": [],
                "lab_pattern": "No lab data provided",
                "disease_activity_markers": [],
                "quantitative_thresholds_met": [],
            }

            await self.ledger.write(
                source=AgentSource.CLINICAL,
                event_type="LAB_ANALYSIS",
                message="No lab data provided; lab analysis skipped.",
                data=result,
                tags=["LABS", "NO_DATA"],
                severity=Severity.NORMAL,
            )
            return result

        diagnosis_codes = [
            self._safe_model_dump(dc) for dc in (getattr(ehr, "diagnosis_codes", []) or [])
        ]
        lab_payload = [self._safe_model_dump(lab) for lab in labs]
        requested_procedure = self._safe_model_dump(getattr(ehr, "requested_procedure", None))

        prompt = f"""Analyze these lab results for insurance prior authorization medical necessity:

Patient: {ehr.patient_name}, {ehr.patient_age}{ehr.patient_sex}
Diagnosis: {json.dumps(diagnosis_codes, ensure_ascii=False)}
Requested Procedure: {json.dumps(requested_procedure, ensure_ascii=False)}

Lab Results:
{json.dumps(lab_payload, indent=2, ensure_ascii=False)}

Respond with JSON:
{{
    "critical_findings": [
        {{
            "lab_name": "string",
            "value": "string",
            "clinical_significance": "string - what this means for the patient",
            "supports_procedure": true,
            "insurance_relevance": "string - how this helps justify the procedure to insurance"
        }}
    ],
    "lab_pattern": "string - what the overall lab picture suggests",
    "disease_activity_markers": ["list of labs showing active disease"],
    "quantitative_thresholds_met": ["list of specific numeric thresholds that insurance policies commonly use, e.g., 'ANA >= 1:320'"]
}}"""

        fallback = {
            "critical_findings": [],
            "lab_pattern": "Unable to determine lab pattern",
            "disease_activity_markers": [],
            "quantitative_thresholds_met": [],
        }

        result = await self._call_json_model(prompt, fallback)

        result.setdefault("critical_findings", [])
        result.setdefault("lab_pattern", "Unable to determine lab pattern")
        result.setdefault("disease_activity_markers", [])
        result.setdefault("quantitative_thresholds_met", [])

        if not isinstance(result["critical_findings"], list):
            result["critical_findings"] = []
        if not isinstance(result["disease_activity_markers"], list):
            result["disease_activity_markers"] = []
        if not isinstance(result["quantitative_thresholds_met"], list):
            result["quantitative_thresholds_met"] = []

        critical_count = len(result.get("critical_findings", []))
        critical_labs = [
            f.get("lab_name", "")
            for f in result.get("critical_findings", [])
            if isinstance(f, dict) and f.get("supports_procedure")
        ]

        await self.ledger.write(
            source=AgentSource.CLINICAL,
            event_type="LAB_ANALYSIS",
            message=(
                f"Lab analysis complete. {critical_count} critical finding(s). "
                f"Disease activity markers: {', '.join(result.get('disease_activity_markers', ['none']))}. "
                f"Labs supporting procedure: {', '.join(critical_labs) if critical_labs else 'none'}."
            ),
            data=result,
            tags=["LABS", "CRITICAL_VALUES"] if critical_count > 0 else ["LABS"],
            severity=Severity.CRITICAL if critical_count > 0 else Severity.NORMAL,
        )

        return result

    async def _assess_medical_necessity(
        self,
        ehr: EHRData,
        symptom_analysis: Optional[dict] = None,
        lab_analysis: Optional[dict] = None,
    ) -> dict:
        """
        Assess medical necessity and generate hints for the Policy Agent.
        Also writes structured context into the shared ledger.
        """
        symptom_analysis = symptom_analysis or {}
        lab_analysis = lab_analysis or {}

        diagnosis_codes = [
            self._safe_model_dump(dc) for dc in (getattr(ehr, "diagnosis_codes", []) or [])
        ]
        lab_payload = [self._safe_model_dump(lab) for lab in (getattr(ehr, "labs", []) or [])]
        requested_procedure = self._safe_model_dump(getattr(ehr, "requested_procedure", None))
        prior_treatments = getattr(ehr, "prior_treatments", []) or []

        prompt = f"""You are assessing medical necessity for a prior authorization request.
Your assessment will be shared with a Policy Agent that searches insurance policy documents.
You need to identify what kind of policy clauses the Policy Agent should look for.

Patient: {ehr.patient_name}, {ehr.patient_age}{ehr.patient_sex}
Chief Complaint: {ehr.chief_complaint}
Symptoms: {json.dumps(getattr(ehr, "symptoms", []) or [], ensure_ascii=False)}
Diagnosis Codes: {json.dumps(diagnosis_codes, ensure_ascii=False)}
Lab Results: {json.dumps(lab_payload, ensure_ascii=False)}
Requested Procedure: {json.dumps(requested_procedure, ensure_ascii=False)}
Prior Treatments: {json.dumps(prior_treatments, ensure_ascii=False)}
Clinical Notes: {ehr.clinical_notes or 'None'}
Symptom Analysis: {json.dumps(symptom_analysis, ensure_ascii=False)}
Lab Analysis: {json.dumps(lab_analysis, ensure_ascii=False)}

Respond with JSON:
{{
    "necessity_level": "MEDICALLY_NECESSARY | RECOMMENDED | ELECTIVE",
    "acuity": "routine | urgent | emergent",
    "justification": "string - 2-3 sentence medical necessity justification",
    "policy_search_hints": [
        "string - specific types of policy clauses the Policy Agent should prioritize searching for",
        "e.g., 'autoimmune disease exception clauses', 'expedited review for active disease', 'multi-morbidity fast-track'"
    ],
    "exception_indicators": [
        "string - specific clinical facts that might trigger policy exceptions",
        "e.g., 'ANA >= 1:320 with confirmatory antibodies', 'BNP > 400 pg/mL'"
    ],
    "documentation_requirements": [
        "string - what documentation the claim should include"
    ],
    "risk_if_denied": "string - clinical risk if the procedure is denied or delayed"
}}"""

        fallback = {
            "necessity_level": "RECOMMENDED",
            "acuity": symptom_analysis.get("acuity", "routine"),
            "justification": "Insufficient structured output from model; manual review recommended.",
            "policy_search_hints": ["standard medical necessity review"],
            "exception_indicators": [],
            "documentation_requirements": [],
            "risk_if_denied": "Potential delay in clinically indicated care.",
        }

        result = await self._call_json_model(prompt, fallback)

        result.setdefault("necessity_level", "RECOMMENDED")
        result.setdefault("acuity", symptom_analysis.get("acuity", "routine"))
        result.setdefault("justification", "Clinical review completed.")
        result.setdefault("policy_search_hints", ["standard medical necessity review"])
        result.setdefault("exception_indicators", [])
        result.setdefault("documentation_requirements", [])
        result.setdefault("risk_if_denied", "Potential delay in care.")

        if not isinstance(result["policy_search_hints"], list):
            result["policy_search_hints"] = ["standard medical necessity review"]
        if not isinstance(result["exception_indicators"], list):
            result["exception_indicators"] = []
        if not isinstance(result["documentation_requirements"], list):
            result["documentation_requirements"] = []

        await self.ledger.write(
            source=AgentSource.CLINICAL,
            event_type="NECESSITY_ASSESSMENT",
            message=(
                f"Medical necessity: {result.get('necessity_level', 'UNKNOWN')}. "
                f"Acuity: {result.get('acuity', 'routine')}. "
                f"Policy search hints for Policy Agent: {', '.join(result.get('policy_search_hints', []))}."
            ),
            data=result,
            tags=[
                "NECESSITY",
                str(result.get("necessity_level", "UNKNOWN")),
                str(result.get("acuity", "ROUTINE")).upper(),
            ],
            severity=(
                Severity.CRITICAL
                if result.get("necessity_level") == "MEDICALLY_NECESSARY"
                else Severity.HIGH
            ),
        )

        await self.ledger.set_context("clinical_necessity_level", result.get("necessity_level"))
        await self.ledger.set_context("policy_search_hints", result.get("policy_search_hints", []))
        await self.ledger.set_context("exception_indicators", result.get("exception_indicators", []))
        await self.ledger.set_context(
            "clinical_risk_if_denied",
            result.get("risk_if_denied", "Potential delay in care."),
        )
        await self.ledger.set_context(
            "documentation_requirements",
            result.get("documentation_requirements", []),
        )
        await self.ledger.set_context(
            "clinical_acuity",
            result.get("acuity", "routine"),
        )

        return result