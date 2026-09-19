# Optional TypeSafe Jev reference integration

The project-local `.venv` pins `typesafe-sdk==0.7.0`, installed from the official Python package. The vendored TypeSafe skill is at `.typesafe-skill/skills/typesafe-ai/SKILL.md`, from skill repository revision `65a39f393687675ce170e6094757de20370365b9`. The adapter uses `TYPESAFE_API_KEY` and `JEV_MODEL` (default `jev-1.13.0`) and asks only narrow failure-category judgments.

No TypeSafe credential was present during the initial build, so live Jev calls
were not made. The standard v2 install is fully local and does not require
`TYPESAFE_API_KEY`; install the `typesafe` optional dependency only for an
explicit comparison. `off` remains the default and `shadow` never changes
routing. NanoJev is the local decision-provider experiment and is documented
in `NANOJEV.md`. No private code, credentials, browser data, or production
logs are sent by the default path.
