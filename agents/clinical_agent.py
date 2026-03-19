"""
Clinical Agent — EHR Data Extraction & Analysis

This agent:
1. Receives raw EHR data (structured or unstructured)
2. Uses Claude to extract clinical indicators, severity, and medical necessity
3. Writes each finding to the Medical Necessity Ledger
4. Produces a structured clinical summary for the Policy Agent

The key insight: this agent doesn't just extract data — it *interprets* it
through the lens of medical necessity for insurance authorization.
"""

import json
from anthropic import AsyncAnthropic
from models.schemas import (
    AgentSource, EHRData, Severity, DiagnosisCode, LabResult
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

You must respond in valid JSON format only. No markdown, no explanation outside JSON.
"""


class ClinicalAgent:
    def __init__(self, client: AsyncAnthropic, ledger: MedicalNecessityLedger):
        self.client = client
        self.ledger = ledger
        self.model = "claude-sonnet-4-20250514"

    async def analyze_ehr(self, ehr: EHRData) -> dict:
        """
        Main entry point: analyze EHR data and write findings to the ledger.
        Returns the complete clinical analysis.
        """
        # Step 1: Log start
        await self.ledger.write(
            source=AgentSource.CLINICAL,
            event_type="SCAN_START",
            message=f"Initiating EHR scan for {ehr.patient_name} ({ehr.patient_age}{ehr.patient_sex})",
            data={"patient": ehr.patient_name, "chief_complaint": ehr.chief_complaint},
            tags=["EHR_SCAN", "INIT"],
        )

        # Step 2: Extract and analyze symptoms
        symptom_analysis = await self._analyze_symptoms(ehr)

        # Step 3: Analyze lab results
        lab_analysis = await self._analyze_labs(ehr)

        # Step 4: Assess medical necessity with Claude
        necessity_assessment = await self._assess_medical_necessity(ehr)

        # Step 5: Write comprehensive context to ledger for the Policy Agent
        await self.ledger.write(
            source=AgentSource.LEDGER,
            event_type="CLINICAL_CONTEXT_COMPLETE",
            message=(
                f"Clinical context ready for policy search. "
                f"Primary Dx: {ehr.diagnosis_codes[0].description if ehr.diagnosis_codes else 'Pending'}. "
                f"Necessity level: {necessity_assessment.get('necessity_level', 'UNKNOWN')}. "
                f"Recommended search parameters: {necessity_assessment.get('policy_search_hints', 'standard review')}."
            ),
            data={
                "diagnosis_codes": [dc.model_dump() for dc in ehr.diagnosis_codes],
                "critical_labs": [
                    lab.model_dump() for lab in ehr.labs if lab.flag in (Severity.HIGH, Severity.CRITICAL)
                ],
                "requested_procedure": ehr.requested_procedure.model_dump() if ehr.requested_procedure else {},
                "necessity_level": necessity_assessment.get("necessity_level", "STANDARD"),
                "policy_search_hints": necessity_assessment.get("policy_search_hints", []),
                "exception_indicators": necessity_assessment.get("exception_indicators", []),
                "acuity": necessity_assessment.get("acuity", "routine"),
            },
            tags=["LEDGER_WRITE", "CONTEXT_SHARED", "CLINICAL_COMPLETE"],
            severity=Severity.CRITICAL,
        )

        return {
            "symptom_analysis": symptom_analysis,
            "lab_analysis": lab_analysis,
            "necessity_assessment": necessity_assessment,
        }

    async def _analyze_symptoms(self, ehr: EHRData) -> dict:
        """Use Claude to analyze symptoms in clinical context."""
        prompt = f"""Analyze these clinical findings for insurance prior authorization:

Patient: {ehr.patient_name}, {ehr.patient_age}{ehr.patient_sex}
Chief Complaint: {ehr.chief_complaint}
Symptoms: {json.dumps(ehr.symptoms)}
Diagnosis Codes: {json.dumps([dc.model_dump() for dc in ehr.diagnosis_codes])}
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

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=CLINICAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(match.group()) if match else {"error": "Parse failed", "raw": result_text}

        # Write findings to ledger
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
            tags=["SYMPTOMS", result.get("condition_category", "").upper(), result.get("acuity", "").upper()],
            severity=Severity.HIGH if result.get("acuity") in ("urgent", "emergent") else Severity.NORMAL,
        )

        return result

    async def _analyze_labs(self, ehr: EHRData) -> dict:
        """Analyze lab results for clinical significance."""
        if not ehr.labs:
            return {"findings": [], "summary": "No lab data provided"}

        prompt = f"""Analyze these lab results for insurance prior authorization medical necessity:

Patient: {ehr.patient_name}, {ehr.patient_age}{ehr.patient_sex}
Diagnosis: {json.dumps([dc.model_dump() for dc in ehr.diagnosis_codes])}
Requested Procedure: {ehr.requested_procedure.model_dump() if ehr.requested_procedure else 'None'}

Lab Results:
{json.dumps([lab.model_dump() for lab in ehr.labs], indent=2)}

Respond with JSON:
{{
    "critical_findings": [
        {{
            "lab_name": "string",
            "value": "string",
            "clinical_significance": "string - what this means for the patient",
            "supports_procedure": true/false,
            "insurance_relevance": "string - how this helps justify the procedure to insurance"
        }}
    ],
    "lab_pattern": "string - what the overall lab picture suggests",
    "disease_activity_markers": ["list of labs showing active disease"],
    "quantitative_thresholds_met": ["list of specific numeric thresholds that insurance policies commonly use, e.g., 'ANA >= 1:320'"]
}}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=CLINICAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(match.group()) if match else {"error": "Parse failed"}

        critical_count = len(result.get("critical_findings", []))
        critical_labs = [f.get("lab_name", "") for f in result.get("critical_findings", []) if f.get("supports_procedure")]

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

    async def _assess_medical_necessity(self, ehr: EHRData) -> dict:
        """
        The key function: assess medical necessity AND generate hints for the Policy Agent.
        This is what gets written to shared memory to change the Policy Agent's behavior.
        """
        prompt = f"""You are assessing medical necessity for a prior authorization request.
Your assessment will be shared with a Policy Agent that searches insurance policy documents.
You need to identify what kind of policy clauses the Policy Agent should look for.

Patient: {ehr.patient_name}, {ehr.patient_age}{ehr.patient_sex}
Chief Complaint: {ehr.chief_complaint}
Symptoms: {json.dumps(ehr.symptoms)}
Diagnosis Codes: {json.dumps([dc.model_dump() for dc in ehr.diagnosis_codes])}
Lab Results: {json.dumps([lab.model_dump() for lab in ehr.labs])}
Requested Procedure: {json.dumps(ehr.requested_procedure.model_dump() if ehr.requested_procedure else None)}
Prior Treatments: {json.dumps(ehr.prior_treatments)}
Clinical Notes: {ehr.clinical_notes or 'None'}

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

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=CLINICAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            result = json.loads(match.group()) if match else {"error": "Parse failed"}

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
                result.get("necessity_level", "UNKNOWN"),
                result.get("acuity", "ROUTINE").upper(),
            ],
            severity=Severity.CRITICAL
            if result.get("necessity_level") == "MEDICALLY_NECESSARY"
            else Severity.HIGH,
        )

        # Store structured hints in ledger context for Policy Agent
        await self.ledger.set_context("clinical_necessity_level", result.get("necessity_level"))
        await self.ledger.set_context("policy_search_hints", result.get("policy_search_hints", []))
        await self.ledger.set_context("exception_indicators", result.get("exception_indicators", []))

        return result
