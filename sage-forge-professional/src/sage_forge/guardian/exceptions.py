"""
🛡️ Guardian System Exceptions: Intuitive Error Handling

Exception hierarchy designed for immediate LLM agent understanding of protection failures.
All exception names clearly indicate their protective/security role.
"""


class GuardianError(Exception):
    """
    🛡️ BASE GUARDIAN ERROR: Root exception for all guardian system failures.
    
    Indicates the protective middleware encountered a system-level problem
    that prevents it from safeguarding the application against TiRex vulnerabilities.
    """
    pass


class ShieldViolation(GuardianError):
    """
    🚨 SHIELD PROTECTION VIOLATED: Input or output failed security validation.
    
    Raised when input data violates empirically-validated security boundaries:
    - Excessive NaN ratios (>20% NaN values)
    - Infinity value detection (±inf causes model corruption)
    - Extreme value boundaries (values outside reasonable market range)
    - Output corruption detection (model produced invalid forecasts)
    
    This exception indicates the shield successfully BLOCKED a potential attack.
    """
    
    def __init__(self, message: str, violation_type: str = "unknown", 
                 threat_level: str = "medium"):
        super().__init__(message)
        self.violation_type = violation_type
        self.threat_level = threat_level
        self.protection_action = "BLOCKED"


class ThreatDetected(GuardianError):
    """
    🚨 SECURITY THREAT IDENTIFIED: Suspicious attack pattern recognized.
    
    Raised when advanced threat detection systems identify patterns consistent
    with known attack vectors against TiRex:
    - Repeated NaN injection attempts
    - Statistical anomaly patterns  
    - Rate limiting violations (potential DoS)
    - Pattern similarity to known attack signatures
    
    This exception indicates proactive threat detection, not just boundary violation.
    """
    
    def __init__(self, message: str, attack_pattern: str = "unknown",
                 confidence_score: float = 0.0, user_id: str = None):
        super().__init__(message)
        self.attack_pattern = attack_pattern
        self.confidence_score = confidence_score
        self.user_id = user_id
        self.detection_type = "PROACTIVE"


class TiRexServiceUnavailableError(GuardianError):
    """
    🔧 TIREX SERVICE FAILURE: Circuit breaker activated due to model failures.
    
    Raised when the circuit breaker determines TiRex is experiencing systematic
    failures and has opened the circuit to prevent cascade failures:
    - Multiple consecutive model inference failures
    - TiRex returning corrupted outputs (NaN/inf)
    - Model loading or initialization failures
    - GPU/memory resource exhaustion
    
    Guardian will attempt fallback strategies when this occurs.
    """
    
    def __init__(self, message: str, failure_count: int = 0, 
                 circuit_state: str = "OPEN", recovery_time: float = None):
        super().__init__(message)
        self.failure_count = failure_count
        self.circuit_state = circuit_state
        self.recovery_time = recovery_time
        self.fallback_available = True


class FallbackExhaustionError(GuardianError):
    """
    ⚠️ ALL FALLBACKS FAILED: Guardian cannot provide any forecast capability.
    
    Raised when both TiRex and all fallback forecasting methods have failed:
    - TiRex circuit breaker is OPEN (primary model unavailable)
    - Simple moving average fallback failed
    - Linear trend fallback failed  
    - Last value fallback failed (ultimate fallback)
    
    This represents complete forecasting system failure requiring manual intervention.
    """
    
    def __init__(self, message: str, failed_fallbacks: list = None):
        super().__init__(message)
        self.failed_fallbacks = failed_fallbacks or []
        self.system_status = "CRITICAL"
        self.requires_intervention = True


class ValidationTimeout(GuardianError):
    """
    ⏱️ PROTECTION TIMEOUT: Security validation exceeded time limits.
    
    Raised when guardian protection layers take longer than acceptable:
    - Input validation processing timeout (potential DoS via expensive validation)
    - Output validation timeout (large forecast validation)
    - Threat detection timeout (complex pattern analysis)
    
    Prevents guardian system from becoming attack vector itself.
    """
    
    def __init__(self, message: str, timeout_seconds: float, operation_type: str):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.operation_type = operation_type
        self.protection_status = "TIMEOUT"


# Exception hierarchy for LLM agent understanding
GUARDIAN_EXCEPTION_HIERARCHY = {
    'GuardianError': 'Base protection system error',
    'ShieldViolation': 'Input/output security boundary violated', 
    'ThreatDetected': 'Suspicious attack pattern identified',
    'TiRexServiceUnavailableError': 'Primary model unavailable, fallback activated',
    'FallbackExhaustionError': 'All forecasting methods failed',
    'ValidationTimeout': 'Protection validation exceeded time limits'
}


def classify_guardian_error(exception: Exception) -> dict:
    """
    Classify guardian errors for intuitive LLM agent understanding.
    
    Args:
        exception: Guardian exception to classify
        
    Returns:
        Dict with error classification for agent decision making
    """
    if isinstance(exception, ThreatDetected):
        return {
            'category': 'SECURITY_THREAT',
            'severity': 'HIGH',
            'action_required': 'BLOCK_AND_AUDIT',
            'user_notification': 'SECURITY_ALERT'
        }
    
    elif isinstance(exception, ShieldViolation):
        return {
            'category': 'BOUNDARY_VIOLATION', 
            'severity': 'MEDIUM',
            'action_required': 'BLOCK_INPUT',
            'user_notification': 'INVALID_INPUT'
        }
    
    elif isinstance(exception, TiRexServiceUnavailableError):
        return {
            'category': 'SERVICE_DEGRADATION',
            'severity': 'MEDIUM', 
            'action_required': 'USE_FALLBACK',
            'user_notification': 'DEGRADED_SERVICE'
        }
    
    elif isinstance(exception, FallbackExhaustionError):
        return {
            'category': 'SYSTEM_FAILURE',
            'severity': 'CRITICAL',
            'action_required': 'MANUAL_INTERVENTION',
            'user_notification': 'SERVICE_UNAVAILABLE'
        }
    
    else:
        return {
            'category': 'UNKNOWN_ERROR',
            'severity': 'HIGH',
            'action_required': 'ESCALATE',
            'user_notification': 'SYSTEM_ERROR'
        }