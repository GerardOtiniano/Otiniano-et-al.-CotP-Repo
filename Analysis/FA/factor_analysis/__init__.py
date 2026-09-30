"""Factor analysis tools."""
from .fa_em import fa_em
from .eof import run_fa
from .fa_wrappers import fa_proxy_wrapper, fa_prod_wrapper
from .import_prod import load_prod

__all__ = ["fa_em", "run_fa", "fa_proxy_wrapper", "fa_prod_wrapper", "load_prod"]
