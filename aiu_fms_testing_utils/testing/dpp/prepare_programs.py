import re
from aiu_fms_testing_utils.testing.dpp.program_models import ProgramInfo
from aiu_fms_testing_utils.utils.paged import ProgramCriteria


def _parse_program_limit(limit_str: str) -> tuple[int, str | None]:
    """Parses a program limit string into a numeric value and comparison operator.

    Accepts either a plain integer (defaults to ">=" for backward compatibility)
    or a string with a comparison operator prefix (e.g., ">=10", "<5", "==8").

    Args:
        limit_str: String representation of the limit, either a number or
                   operator+number (e.g., "10", ">=10", "<5").

    Returns:
        Tuple containing:
            - limit_val: The numeric limit value.
            - limit_type: The comparison operator string (">=", "<=", "<", ">", "==").

    Raises:
        ValueError: If the limit string format is invalid."""

    matcher = re.compile(r"^(<|>|<=|>=|==)(\d+)")

    # Default limit to min to maintain backwards compat
    try:
        limit_type = ">="
        limit_val = int(limit_str)
    except ValueError:
        limit_type = None
        match = matcher.fullmatch(limit_str)
        if match is None:
            raise ValueError("Program not well formatted, wrong limit type")
        limit_type = match.group(1)
        limit_val = int(match.group(2))
    return limit_val, limit_type


def get_programs_to_test(
    programs: list[str], program_criteria_list: list[ProgramCriteria]
) -> list[ProgramInfo]:
    """Parses program specifications into ProgramInfo objects for testing.

    Converts command-line program specifications into structured ProgramInfo objects.
    Supports three formats:
    - Empty list: Tests all programs with any valid prompt.
    - "program_id": Tests specific program with any valid prompt.
    - "program_id:batch_constraint,prompt_constraint": Tests program with specific constraints.

    Args:
        programs: List of program specification strings from command line.
        program_criteria_list: List of ProgramCriteria objects defining available programs.

    Returns:
        List of ProgramInfo objects representing programs to test with their constraints."""

    programs_to_test = []
    for program_str in programs:
        enforce_prompt_split = program_str.split(":")
        program_id = enforce_prompt_split[0]
        if len(enforce_prompt_split) == 1:
            programs_to_test.append(
                ProgramInfo(program_id, 0, ">=", 0, ">=")
            )  # this will always satisfy
        else:
            enforce_batch_size, enforce_prompt_length = (
                _ for _ in enforce_prompt_split[1].split(",")
            )

            # Default limit to min to maintain backwards compat
            enforce_batch_size_val, enforce_batch_size_type = _parse_program_limit(
                enforce_batch_size
            )
            enforce_prompt_length_val, enforce_prompt_length_type = (
                _parse_program_limit(enforce_prompt_length)
            )

            programs_to_test.append(
                ProgramInfo(
                    program_id,
                    enforce_batch_size_val,
                    enforce_batch_size_type,
                    enforce_prompt_length_val,
                    enforce_prompt_length_type,
                )
            )

    if len(programs_to_test) == 0:
        programs_to_test = [
            ProgramInfo(str(p.program_id), 0, ">=", 0, ">=")
            for p in program_criteria_list
        ]

    return programs_to_test
