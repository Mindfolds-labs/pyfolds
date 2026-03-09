from pyfolds.serialization.security_levels import SecurityLevel, get_security_config


def test_parse_defaults_cache():
    assert SecurityLevel.parse("basic") is SecurityLevel.BASIC
    a = get_security_config("high")
    b = get_security_config(SecurityLevel.HIGH)
    assert a is b
    assert a.merkle and a.encryption and a.trust_block
    assert not get_security_config("basic").ecc
