import secrets
# Generate a URL-safe text string, containing 32 random bytes
api_key = secrets.token_urlsafe(32)
print(f"{api_key}")
