from evidencedesk_api.security import hash_password, verify_password


def test_password_is_argon2_hashed_and_verifiable() -> None:
    password = "EvidenceDemo-Analyst-2026!"

    encoded = hash_password(password)

    assert encoded != password
    assert encoded.startswith("$argon2")
    assert verify_password(password, encoded) is True
    assert verify_password("wrong-password", encoded) is False
