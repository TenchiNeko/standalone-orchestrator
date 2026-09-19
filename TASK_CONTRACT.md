# Task contracts

A task JSON must provide a nonempty `goal`, explicit `workspace`, permitted files/actions, and at least one required criterion. Criteria are verified against evidence references and the source hash that produced them. A missing, contradictory, or stale criterion blocks completion. Required tests are distinct from advisory checks; a successful model narrative is never evidence.
