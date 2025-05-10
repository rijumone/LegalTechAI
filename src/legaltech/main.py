from enum import Enum
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Union
import uuid
import markitdown


class StatusEnum(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"


class FormatEnum(str, Enum):
    PDF = "pdf"
    DOCX = "docx"


class Case(BaseModel):
    case_id: str
    title: str
    description: Optional[str] = None
    created_at: datetime
    status: StatusEnum

    def create_workflow(self, template_id: str) -> "Workflow":
        """Create a new workflow in 'draft' state."""
        workflow_id = str(uuid.uuid4())
        return Workflow(
            workflow_id=workflow_id,
            case_id=self.case_id,
            template_id=template_id,
            state=StatusEnum.DRAFT,
            created_at=datetime.now()
        )


class Workflow(BaseModel):
    workflow_id: str
    case_id: str
    template_id: str
    state: StatusEnum
    created_at: datetime

    def upload_documents(self, document_ids: List[str]) -> None:
        """Upload supporting documents to the workflow."""
        # In a real app, this would link documents to the workflow
        pass

    def add_knowledge(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add user-provided knowledge to the workflow."""
        # In a real app, this would store knowledge in a separate table
        pass

    def generate_document(self) -> "Document":
        """Generate a document using the template and knowledge."""
        # Placeholder for AI-generated content
        return Document(
            document_id=str(uuid.uuid4()),
            workflow_id=self.workflow_id,
            content="Generated content using template and knowledge",
            format=FormatEnum.PDF,
            created_at=datetime.now()
        )

    def review_edit(self, component: str, new_value: str) -> None:
        """Allow users to edit a component in the generated document."""
        # In a real app, this would update the document content
        pass

    def regenerate_component(self, component: str, prompt: str) -> str:
        """Regenerate a component using a custom AI prompt."""
        # Placeholder for AI regeneration logic
        return f"Regenerated {component} based on prompt: {prompt}"

    def publish(self) -> "Document":
        """Finalize and export the document."""
        if self.state == StatusEnum.DRAFT:
            self.state = StatusEnum.PUBLISHED
        return Document(
            document_id=str(uuid.uuid4()),
            workflow_id=self.workflow_id,
            content="Finalized document content",
            format=FormatEnum.PDF,
            created_at=datetime.now()
        )


class Template(BaseModel):
    template_id: str
    name: str
    description: Optional[str] = None
    # List of placeholder names (e.g., ["{{Plaintiff Name}}"])
    components: List[str]


class Knowledge(BaseModel):
    knowledge_id: str
    workflow_id: str
    document_id: str
    content: str  # Raw text extracted from documents
    metadata: Optional[Dict] = None  # Extracted structured data


class Document(BaseModel):
    document_id: str
    workflow_id: str
    content: str
    format: FormatEnum
    created_at: datetime

    def _import(self, filepath) -> None:
        """Given the local path to a document, run OCR on it, convert to markdown, and extract text."""

        


if __name__ == "__main__":
    # Create a case
    case = Case(
        case_id=str(uuid.uuid4()),
        title="Sample Case",
        description="This is a sample case.",
        created_at=datetime.now(),
        status=StatusEnum.DRAFT
    )

    # Create a workflow
    workflow = case.create_workflow(template_id="template_123")

    # Upload documents and add knowledge
    workflow.upload_documents(document_ids=["doc_456"])
    workflow.add_knowledge(
        content="User-provided knowledge", metadata={"key": "value"})

    # Generate and publish a document
    document = workflow.generate_document()
    published_document = workflow.publish()

    print(f"Published Document: {published_document}")
