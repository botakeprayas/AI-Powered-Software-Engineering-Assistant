"""Input validation helpers."""
import re

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email(address):
    """Check that a string looks like an email address."""
    return bool(EMAIL_PATTERN.match(address))


def validate_age(age):
    """Reject ages outside the range 0 to 150."""
    if not isinstance(age, int) or age < 0 or age > 150:
        raise ValueError("age must be an integer between 0 and 150")
    return age