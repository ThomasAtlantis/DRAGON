from dataclasses import dataclass


@dataclass
class SystemProfile:
    rtt: float
    latency_dec_rem: float
    latency_dec_loc: float
    acceptance_rate_rem: float
    acceptance_rate_loc: float

def in_range(x, r):
    return r[0] < x <= r[1]

def strategy(profile: SystemProfile):
    delta_z: float
    if profile.latency_dec_loc == 0:
        delta_z = (1 - profile.acceptance_rate_rem) * profile.rtt
    elif in_range(profile.latency_dec_loc, (
            0, 
            profile.latency_dec_rem - profile.rtt,
        )):
        delta_z = (1 - profile.acceptance_rate_rem) * profile.rtt
    elif in_range(profile.latency_dec_loc, (
            profile.latency_dec_rem - profile.rtt, 
            profile.latency_dec_rem,
        )):
        delta_z = (1 - profile.acceptance_rate_loc) * (profile.latency_dec_rem - profile.latency_dec_loc) + \
            (profile.acceptance_rate_loc - profile.acceptance_rate_rem) * profile.rtt
    elif in_range(profile.latency_dec_loc, (
            profile.latency_dec_rem,
            profile.latency_dec_rem + profile.rtt,
        )):
        delta_z = (1 - profile.acceptance_rate_rem) * (profile.latency_dec_rem - profile.latency_dec_loc) + \
            (profile.acceptance_rate_loc - profile.acceptance_rate_rem) * profile.rtt
    else:
        delta_z = (profile.acceptance_rate_loc - 1) * profile.rtt
    return delta_z
