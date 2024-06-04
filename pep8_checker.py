import io
import pycodestyle
from typing import Set


def check_compliance(code_snippet: str) -> Set[str]:
    """Check PEP8 compliance and list unique codes of violations using pycodestyle."""
    code_file = io.StringIO(code_snippet)
    style_guide = pycodestyle.StyleGuide(quiet=True)
    report = style_guide.init_report(pycodestyle.StandardReport)

    checker = pycodestyle.Checker(filename='(none)', lines=code_file.readlines(), report=report)
    checker.check_all()

    code_file.close()

    violations = {error[2] for error in report._deferred_print}
    return violations
