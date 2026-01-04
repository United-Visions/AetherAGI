"""
AetherMind SDK Exceptions
"""


class AetherMindError(Exception):
    """Base exception for AetherMind SDK"""
    pass


class AuthenticationError(AetherMindError):
    """Raised when API key is invalid or missing"""
    pass


class RateLimitError(AetherMindError):
    """Raised when rate limit is exceeded"""
    pass


class ValidationError(AetherMindError):
    """Raised when request validation fails"""
    pass


class NetworkError(AetherMindError):
    """Raised when network request fails"""
    pass
