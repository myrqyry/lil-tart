
### Pipeline load cancellation

The shared pipelines resolve a repo-relative manifest path against a configured model
base and hand the resulting bytes to a runtime that cannot be cancelled once it has
started compiling. Cancellation therefore has to be proven at the transfer, not at the
compile: this case streams a real local asset through the shared
`createHttpAssetResolver`, aborts after the first chunk, and observes that the
resolver surfaces a cancellation and that far fewer bytes arrived than the asset holds.

The asset is generated under the gitignored `static-models/` directory rather than
committed, so the case needs no multi-megabyte fixture in the repository. The probe
path and its exact byte size live in `probeAsset.ts`, and the browser asserts against
that size, so shrinking or regenerating the asset cannot silently weaken the case.
