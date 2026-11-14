# Network dependency qualification checkpoint

## Combined HIGH follow-up (local, unpublished)
- LiteLLM 1.95.0 → 1.100.4; mandatory boto3 1.44.3, botocore 1.44.3, jmespath 1.1.0, s3transfer 0.19.3 added. Manifest floor >=1.100.4, no invented upper bound; later vulnerable branches are not excluded by the floor alone.
- Transitive fsspec 2026.4.0 → 2026.6.0; no direct requirement added. Other package records, sources/markers and Python contract unchanged apart from project floor metadata.
- Combined frozen all-extras Python 3.12, pip check, Ruff lint/format, configured Pyright, exact pytest -n auto coverage gate and real LiteLLM mock_response/fsspec memory smoke passed: 155 tests, 85.13% coverage.
- Python socket guard selftested; three blocked boundaries observed (two intentional selftest, one additional unlabelled attempt), not zero attempts. Local LiteLLM cost map used; no OS/native/subprocess sandbox or live provider/store acceptance claim.
- Four-file candidate includes manifest, lock, changelog and this existing checkpoint; independent review and budget acceptance required before separate staging/commit/push authorization. No release/version or remote alert-closure claim. Original checkout preserved; Chroma risk and other dependency exclusions unchanged.
- Evidence: /home/voytas/.hermes/artifacts/github-security-next-implementation/ctm-high-combined (hash-sealed packet).

## Historical network qualification

Status: local qualification; publication pending independent review and separate approval.

- Scope: uv.lock, CHANGELOG.md, and this checkpoint only; no staging, commit or push.
- Locked upgrades: aiohttp 3.14.3 → 3.14.4; urllib3 2.7.0 → 2.8.0; mandatory yarl 1.23.0 → 1.25.1.
- Results: 155 passed, 4 warnings; 85.13% coverage with pytest -n auto --cov=src --cov-fail-under=85 --disable-warnings -q.
- Existing frozen all-extras installation, pip check, Ruff lint/format and Pyright receipts passed.
- CTM xdist guard selftest: 5 passed; controller and all 4 automatic workers loaded the artifact guard through PYTHONPATH. No resource cap or installed .pth change was imposed.
- CTM exact suite: zero observed calls to patched network/provider/store methods; Python guards are not an OS sandbox and do not comprehensively cover native code, arbitrary subprocesses or other APIs.
- DRM exclusions: test_redis_working_memory_ttl_expiry, test_redis_store_recovers_after_restart, test_semantic_roundtrip_with_real_chroma. Historical DRM guards observed blocked DNS attempts; no zero-attempt claim for DRM.
- No live-provider/store acceptance or exploit-prevention claim. Existing unpatched Chroma risk decisions remain unchanged.
- Source, manifests, Python support, CI, security settings, provider/runtime configuration and roadmap are unchanged; no release-version bump is required for this unreleased lock-only slice.
- Earlier launcher failure receipts were overwritten; their historical trace is unavailable.
- Evidence: external drm-ctm-network and drm-ctm-network-closeout review packets; publication requires exact-byte independent review, then separate staging/commit/push authorization and remote verification.
