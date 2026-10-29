# Export metadata note

Reviewer attribution in the exported v7 pretraining review is normalized to “independent implementation review”. This is a presentation-only metadata change; experiment source, data, models, ledgers, metrics, test results, and their hashes are unchanged. Original immutable run files are retained without modification.

The original `artifact_inventory.json` remains provenance for the original run and therefore records the original review-file hash. For this export, use the package/archive SHA256 manifest and verifier, which record the exported review bytes. `EXPORT_HASH_MAPPING.json` explicitly maps the original review hash to its exported hash. These exports do not claim that the review metadata is byte-identical to the original run.
