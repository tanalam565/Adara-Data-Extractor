import secrets
# Generate a URL-safe text string, containing 32 random bytes
api_key = secrets.token_urlsafe(32)
print(f"sk_live_{api_key}")
