from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class Role(StrEnum):
    ADMIN = "admin"
    ANALYST = "analyst"
    READER = "reader"


class Action(StrEnum):
    VIEW_DOCUMENT = "view_document"
    UPLOAD_DOCUMENT = "upload_document"
    ASK_QUESTION = "ask_question"
    VIEW_EXTRACTION = "view_extraction"
    VIEW_AUDIT = "view_audit"
    DELETE_DOCUMENT = "delete_document"
    RUN_EVALUATION = "run_evaluation"


@dataclass(frozen=True, slots=True)
class UserPrincipal:
    id: UUID
    username: str
    role: Role


_ROLE_ACTIONS: dict[Role, frozenset[Action]] = {
    Role.ADMIN: frozenset(Action),
    Role.ANALYST: frozenset(
        {
            Action.VIEW_DOCUMENT,
            Action.UPLOAD_DOCUMENT,
            Action.ASK_QUESTION,
            Action.VIEW_EXTRACTION,
        }
    ),
    Role.READER: frozenset({Action.VIEW_DOCUMENT, Action.ASK_QUESTION, Action.VIEW_EXTRACTION}),
}


def role_allows(role: Role, action: Action) -> bool:
    return action in _ROLE_ACTIONS[role]
