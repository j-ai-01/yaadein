from yaadein.redact import redact


def test_redacts_aws_access_key():
    out = redact("creds: AKIAIOSFODNN7EXAMPLE done")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "done" in out


def test_redacts_key_value_assignments():
    out = redact("api_key = sk_live_abc123 and password: hunter2")
    assert "sk_live_abc123" not in out
    assert "hunter2" not in out


def test_redacts_bearer_tokens():
    out = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out


def test_redacts_private_key_blocks():
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKC\n-----END RSA PRIVATE KEY-----"
    out = redact(f"here {block} there")
    assert "MIIEowIBAAKC" not in out
    assert "there" in out


def test_redacts_github_and_openai_style_tokens():
    out = redact("use ghp_16C7e42F292c6912E7710c838347Ae178B4a and sk-proj-abcdefghij1234567890")
    assert "ghp_16C7e42F292c6912E7710c838347Ae178B4a" not in out
    assert "sk-proj-abcdefghij1234567890" not in out


def test_redacts_long_high_entropy_tokens():
    out = redact("token was g9X2kQ7vZp4mW8rT1nB5cY3hL6jD0aFs")
    assert "g9X2kQ7vZp4mW8rT1nB5cY3hL6jD0aFs" not in out


def test_leaves_normal_prose_and_paths_alone():
    text = "The user prefers pytest; config lives in /Users/jai/workplace/rag-pipeline/config.py"
    assert redact(text) == text


def test_redacts_snake_case_prefixed_labels():
    out = redact("aws_secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    assert "wJalrXUtnFEMI" not in out


def test_redacts_suffixed_labels():
    out = redact("db_password=hunter2 and stripe_secret_key=abc123xyz")
    assert "hunter2" not in out
    assert "abc123xyz" not in out


def test_redacts_unlabeled_stripe_style_keys():
    out = redact("charge it with sk_live_abc123def456 quickly")
    assert "sk_live_abc123def456" not in out


def test_comma_joined_pairs_redacted_independently():
    out = redact("password=hunter2,api_key=abc123xyz")
    assert "hunter2" not in out
    assert "abc123xyz" not in out
    assert "api_key" in out  # second label survives
