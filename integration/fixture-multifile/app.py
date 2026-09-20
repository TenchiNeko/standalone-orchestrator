from formatter import format_greeting


def greeting(name: str, excited: bool = False) -> str:
    return format_greeting(name, excited=excited)
