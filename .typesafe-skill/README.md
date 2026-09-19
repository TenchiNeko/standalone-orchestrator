# TypeSafe Agent Skills

Agent skills for building with [TypeSafe](https://typesafe.ai): typed decisions and probabilities from System One models.

You can [read SKILL.md on GitHub](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md) or fetch its [raw Markdown](https://raw.githubusercontent.com/typesafe-ai/skills/main/skills/typesafe-ai/SKILL.md).

## Install

### Claude Code plugin

Run in your terminal:

```bash
claude plugin marketplace add typesafe-ai/skills
claude plugin install typesafe@typesafe-ai
```

### Other agents via skills.sh

```bash
npx skills add typesafe-ai/skills --skill typesafe-ai
```

Select your agent when prompted. Installation is project-local by default; add `-g` to install globally.

See the [installation guide](https://docs.typesafe.ai/agent-skill#installation) for a prompt to copy to your agent, manual installation, and updates.

## Use

Ask your agent, for example:

> Use TypeSafe to route incoming support tickets by department, with human review for uncertain decisions.

In Claude Code, you can explicitly invoke the plugin skill with `/typesafe:typesafe-ai`.

| Skill | Purpose |
|---|---|
| [typesafe-ai](skills/typesafe-ai/SKILL.md) | Design TypeSafe workflows, find current docs and cookbooks, and compose typed judgments in code |

## License

[MIT](LICENSE).
