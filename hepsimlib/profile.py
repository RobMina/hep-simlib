"""The experiment profile: the few settings in which one experiment's use of the package differs from another's.

Everything else (the manifest schema, the hashes, the load checks, the domain accounting, the validation vocabulary,
the component banks) is the same for every experiment. A profile holds:

    subsystems        allowed manifest subsystems (None: any lower-case identifier)
    dtype_env         environment variable naming the run's device float, read by dtypes.compute_dtype(None)
    default_dtype     the device float when that variable is unset
    cache_env         environment variable of the library download cache directory
    cache_dir         the cache directory when that variable is unset
    release_url       template of a release-asset URL, with {tag} and {file} (None: no default URL)
    bank_subsystem    the manifest subsystem of component banks; bank names start with bank_prefix
    bank_reasons      default reasons for the bank manifest fields a caller does not give
    bank_validation_reason   why a bank with a complete manifest is still not admissible

One profile is current per process. An experiment package installs its own once, when it is imported:

    from hepsimlib import profile
    profile.configure(profile.Profile(name="myexp", subsystems=("calo", "tracker"), dtype_env="MYEXP_FLOAT"))

and :func:`using` sets one for a block (tests). Standard library only.
"""
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType

IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z")
COMPUTE = ("float32", "float64")

BANK_REASONS = MappingProxyType(dict(
    source="the capture products the bank was built from are not given",
    generator="physics-list, material and cut hashes are not recorded yet",
    release="not given to save_library()", seeds="not given to save_library()",
    split="not given to save_library()", config_hash="no build configuration manifest"))
BANK_VALIDATION_REASON = ("a complete manifest is necessary, not sufficient (docs/component-training.md); the energy "
                          "accounting stays unrecorded until the bank carries an energy-accounting report hash")


@dataclass(frozen=True, eq=False)
class Profile:
    name: str = "generic"
    subsystems: tuple = None
    dtype_env: str = "HEPSIMLIB_DEVICE_FLOAT"
    default_dtype: str = "float64"
    cache_env: str = "HEPSIMLIB_LIBRARY_CACHE"
    cache_dir: str = "~/.cache/hepsimlib/libraries"
    release_url: str = None
    bank_subsystem: str = "component"
    bank_prefix: str = "component."
    bank_reasons: MappingProxyType = field(default=BANK_REASONS)
    bank_validation_reason: str = BANK_VALIDATION_REASON

    def __post_init__(self):
        problems = []
        if self.subsystems is not None:
            subs = tuple(self.subsystems)
            if not subs or any(not isinstance(s, str) or not IDENTIFIER.match(s) for s in subs):
                problems.append("subsystems must be None or lower-case identifiers")
            object.__setattr__(self, "subsystems", subs)
        if self.default_dtype not in COMPUTE:
            problems.append(f"default_dtype must be one of {COMPUTE}")
        if not self.dtype_env or not self.cache_env or not self.cache_dir:
            problems.append("dtype_env, cache_env and cache_dir are required")
        if self.release_url is not None and not {"{tag}", "{file}"} <= set(re.findall(r"\{\w+\}", self.release_url)):
            problems.append("release_url must contain {tag} and {file}")
        if not self.allows(self.bank_subsystem) or not self.bank_prefix.startswith(self.bank_subsystem + "."):
            problems.append("bank_subsystem must be an allowed subsystem and bank_prefix start with it and a dot")
        if not self.bank_prefix.endswith("."):
            problems.append("bank_prefix must end with a dot")
        reasons = dict(self.bank_reasons)
        if set(reasons) != set(BANK_REASONS) or not all(isinstance(v, str) and v.strip() for v in reasons.values()):
            problems.append(f"bank_reasons needs a reason for each of {sorted(BANK_REASONS)}")
        if problems:
            raise ValueError(f"profile {self.name!r}: " + "; ".join(problems))
        object.__setattr__(self, "bank_reasons", MappingProxyType(reasons))

    def allows(self, subsystem):
        """True for a lower-case identifier that the profile's subsystem list (if any) contains."""
        return (isinstance(subsystem, str) and bool(IDENTIFIER.match(subsystem))
                and (self.subsystems is None or subsystem in self.subsystems))


GENERIC = Profile()
_LOCK = threading.Lock()
_CURRENT = GENERIC


def current():
    """The profile in force."""
    return _CURRENT


def configure(profile):
    """Make ``profile`` current for the process; returns the previous one."""
    global _CURRENT
    if not isinstance(profile, Profile):
        raise TypeError("configure() takes a hepsimlib.profile.Profile")
    with _LOCK:
        previous, _CURRENT = _CURRENT, profile
    return previous


@contextmanager
def using(profile):
    """Set ``profile`` for the duration of a block (not thread-local: for tests and single-threaded tools)."""
    previous = configure(profile)
    try:
        yield profile
    finally:
        configure(previous)
