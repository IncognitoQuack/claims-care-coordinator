from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class AgentSource(str, Enum):
    CLINICAL = "clinical"
    POLICY = "policy"
    LEDGER = "ledger"
    SYSTEM = "system"


class Severity(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class LedgerEntry(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: AgentSource
    event_type: str
    message: str
    data: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    severity: Severity = Severity.NORMAL


class LabResult(BaseModel):
    name: str
    value: str
    unit: str = ""
    flag: Severity = Severity.NORMAL
    reference_range: str = ""


class DiagnosisCode(BaseModel):
    code: str
    system: str = "ICD-10-CM"
    description: str


class Procedure(BaseModel):
    code: str
    system: str = "CPT"
    name: str
    reason: str


class EHRData(BaseModel):
    patient_name: str
    patient_age: int
    patient_sex: str
    chief_complaint: str
    symptoms: list[str] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    diagnosis_codes: list[DiagnosisCode] = Field(default_factory=list)
    requested_procedure: Optional[Procedure] = None
    prior_treatments: list[str] = Field(default_factory=list)
    clinical_notes: str = ""


class PolicySection(BaseModel):
    section_id: str
    title: str
    text: str
    relevance_score: float = 0.0
    is_exception_clause: bool = False


class PolicyData(BaseModel):
    plan_name: str
    member_id: str
    policy_document: str  # full text
    matched_sections: list[PolicySection] = Field(default_factory=list)


class AuthorizationResult(BaseModel):
    status: str  # APPROVED, DENIED, PENDING_REVIEW
    pathway: str
    reasoning: str
    policy_sections_cited: list[str]
    estimated_processing_time: str
    confidence_score: float
    admin_cost_savings: str = ""


class ClaimRequest(BaseModel):
    ehr_data: EHRData
    policy_document: str
    plan_name: str = "Sample Insurance Plan"
    member_id: str = "DEMO-001"


class SSEEvent(BaseModel):
    event: str
    source: AgentSource
    message: str
    data: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    severity: Severity = Severity.NORMAL
